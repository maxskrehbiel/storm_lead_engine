"""The offline storm source: a synthetic storm season shipped as package data.

This is the only data source that uses the synthetic generator. The pipeline
receives its grids exactly as it would receive real MRMS grids.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from ._types import FloatGrid
from .config import BBox
from .errors import InputDataError
from .grid import KM_PER_DEG_LAT, KM_PER_DEG_LON_AT_EQUATOR, HailGrid, snap_bbox
from .history import parse_day
from .synthetic import SyntheticStorm, rasterize_storm

PACKAGE = "storm_lead_engine"
DEFAULT_FIXTURE = "fixtures/synthetic_season.json"


@dataclass(frozen=True)
class StormSeason:
    """A synthetic storm season loaded from a fixture.

    Attributes:
        name: Human-readable name.
        description: What the fixture contains and why.
        origin: ``(lat, lon)`` of the fictional town centre; storm tracks are relative to it.
        region: Area rasterized for every storm day.
        as_of: Date the season is evaluated at.
        storms: Storm definitions in absolute coordinates.
    """

    name: str
    description: str
    origin: tuple[float, float]
    region: BBox
    as_of: date
    storms: tuple[SyntheticStorm, ...]


def bundled_fixture_text() -> str:
    """Text of the storm season shipped inside the package."""
    return resources.files(PACKAGE).joinpath(DEFAULT_FIXTURE).read_text(encoding="utf-8")


def km_to_latlon(
    origin: tuple[float, float], east_km: float, north_km: float
) -> tuple[float, float]:
    """Offset a ``(lat, lon)`` origin by kilometres east and north (local flat approximation)."""
    km_per_deg_lon = KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(origin[0]))
    return (
        round(origin[0] + north_km / KM_PER_DEG_LAT, 6),
        round(origin[1] + east_km / km_per_deg_lon, 6),
    )


def _parse_storm(raw: dict[str, Any], origin: tuple[float, float]) -> SyntheticStorm:
    core = raw.get("core_at_km")
    return SyntheticStorm(
        storm_id=str(raw["id"]),
        date=str(raw["date"]),
        path=tuple(km_to_latlon(origin, float(e), float(n)) for e, n in raw["path_km"]),
        width_km=float(raw["width_km"]),
        peak_in=float(raw["peak_in"]),
        pulses=float(raw.get("pulses", 2.0)),
        seed=int(raw.get("seed", 0)),
        core_at=km_to_latlon(origin, float(core[0]), float(core[1])) if core else None,
    )


def load_season(path: str | Path | None = None) -> StormSeason:
    """Load a synthetic storm-season fixture.

    Storm tracks in the file are ``[east_km, north_km]`` offsets from ``origin``,
    so a fixture reads as geometry ("1.5 km south of town") and moves with one
    coordinate change.

    Args:
        path: Fixture JSON; None loads the season bundled with the package.

    Returns:
        The season with storms converted to absolute coordinates.

    Raises:
        InputDataError: If the file cannot be read, does not declare itself synthetic,
            or is malformed.
    """
    source = str(path) if path else DEFAULT_FIXTURE
    try:
        text = Path(path).read_text(encoding="utf-8") if path else bundled_fixture_text()
    except OSError as exc:
        raise InputDataError(f"cannot read {source}: {exc.strerror or exc}") from exc
    try:
        d = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputDataError(f"{source}: not valid JSON: {exc}") from exc
    if not isinstance(d, dict) or d.get("synthetic") is not True:
        raise InputDataError(f'{source}: a fixture must declare "synthetic": true')
    try:
        return _build_season(d)
    except (KeyError, TypeError, ValueError) as exc:
        raise InputDataError(f"{source}: malformed fixture: {exc}") from exc


def _build_season(d: dict[str, Any]) -> StormSeason:
    origin = (float(d["origin"][0]), float(d["origin"][1]))
    half_east, half_north = (float(v) for v in d["region_half_km"])
    sw = km_to_latlon(origin, -half_east, -half_north)
    ne = km_to_latlon(origin, half_east, half_north)
    return StormSeason(
        name=str(d["name"]),
        description=str(d.get("description", "")),
        origin=origin,
        region=snap_bbox(BBox(sw[1], sw[0], ne[1], ne[0])),
        as_of=parse_day(str(d["as_of"])),
        storms=tuple(_parse_storm(s, origin) for s in d["storms"]),
    )


def season_day_grids(season: StormSeason) -> dict[str, HailGrid]:
    """Rasterize every storm of a season; storms on the same day are pixel-maxed.

    Each storm draws its texture from its own seeded generator, so adding or
    reordering storms never changes the others.

    Args:
        season: A loaded season.

    Returns:
        Storm-day grids keyed by ``YYYY-MM-DD``.
    """
    lattice = HailGrid.zeros(season.region)
    by_day: dict[str, list[FloatGrid]] = defaultdict(list)
    for storm in season.storms:
        rng = np.random.default_rng(storm.seed)
        by_day[storm.date].append(rasterize_storm(storm, lattice, rng))
    return {
        day: HailGrid.from_mm(np.maximum.reduce(mms), lattice.bounds)
        for day, mms in sorted(by_day.items())
    }
