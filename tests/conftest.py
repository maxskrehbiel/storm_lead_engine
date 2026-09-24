"""Shared fixtures; network access is blocked in every test not marked ``integration``."""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from storm_lead_engine.config import BBox
from storm_lead_engine.fixture import StormSeason, load_season, season_day_grids
from storm_lead_engine.grid import HailGrid
from storm_lead_engine.pipeline import RunResult, run_pipeline
from storm_lead_engine.synthetic import generate_properties, generate_providers

SEED = 7
N_PROPERTIES = 400
GridFactory = Callable[..., HailGrid]


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail loudly if an offline test tries to open a socket."""
    if "integration" not in request.keywords:

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("network access attempted in an offline test")

        monkeypatch.setattr(socket.socket, "connect", refuse)
    yield


@pytest.fixture(scope="session")
def season() -> StormSeason:
    return load_season()


@pytest.fixture(scope="session")
def day_grids(season: StormSeason) -> dict[str, HailGrid]:
    return season_day_grids(season)


@pytest.fixture(scope="session")
def demo_result(season: StormSeason, day_grids: dict[str, HailGrid]) -> RunResult:
    rng = np.random.default_rng(SEED)
    props = generate_properties(N_PROPERTIES, season.origin, season.as_of.year, rng)
    provs = generate_providers(season.origin, rng)
    return run_pipeline(props, provs, day_grids, season.as_of, season.name, synthetic=True)


def _make_grid(values: list[list[float]], west: float = 0.0, north: float = 1.0) -> HailGrid:
    arr = np.asarray(values, dtype=np.float32)
    h, w = arr.shape
    bounds = BBox(west, round(north - h * 0.01, 6), round(west + w * 0.01, 6), north)
    return HailGrid(arr, bounds)


@pytest.fixture
def make_grid() -> GridFactory:
    """Factory: inch values on a 0.01-degree lattice with the NW corner at (north, west)."""
    return _make_grid
