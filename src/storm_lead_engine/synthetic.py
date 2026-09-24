"""Seeded generators for everything that is not public data: storms, a fictional town, providers.

Every generator takes an explicit ``numpy.random.Generator``, and every
modelling assumption is a field on a frozen spec dataclass.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ._types import FloatArray, FloatGrid
from .grid import KM_PER_DEG_LAT, KM_PER_DEG_LON_AT_EQUATOR, MM_PER_INCH, HailGrid
from .properties import Property
from .providers import MAX_STARS, Provider


@dataclass(frozen=True)
class SyntheticStorm:
    """A storm described by its track, not its pixels.

    Attributes:
        storm_id: Identifier within the fixture.
        date: Local storm day, ``YYYY-MM-DD``.
        path: ``(lat, lon)`` vertices of the storm track.
        width_km: Full swath width.
        peak_in: Largest stone anywhere in the swath.
        pulses: Number of along-track intensity pulses (hail streaks).
        seed: Seed for this storm's texture noise.
        core_at: ``(lat, lon)`` where a pulse peaks; None for a random phase.
    """

    storm_id: str
    date: str
    path: tuple[tuple[float, float], ...]
    width_km: float
    peak_in: float
    pulses: float = 2.0
    seed: int = 0
    core_at: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if len(self.path) < 2:
            raise ValueError(f"{self.storm_id}: path needs at least 2 vertices")
        if self.width_km <= 0 or self.peak_in <= 0:
            raise ValueError(f"{self.storm_id}: width_km and peak_in must be positive")


@dataclass(frozen=True)
class StormShape:
    """Shape of the synthetic swath model.

    Hail at a cell = peak x cross-track profile x along-track taper x pulse x texture.

    Attributes:
        taper_start: Fraction of the track over which hail ramps up.
        taper_end: Fraction of the track over which hail dies out.
        pulse_depth: Pulse amplitude; the pulse factor spans ``1 - 2*depth`` to 1.
        texture_sd: Standard deviation of the multiplicative noise.
        texture_min: Lower clip of the noise factor.
        texture_max: Upper clip of the noise factor.
        blur_width: Box-blur width (odd, in cells) used to smooth the noise.
        blur_passes: Number of blur passes (two box passes approximate a Gaussian).
        floor_in: Values below this are set to 0 (no meaningful hail).
        quantum_mm: Output is rounded to this step, so platform-level floating-point
            differences in the math above can never change a stored value.
    """

    taper_start: float = 0.15
    taper_end: float = 0.20
    pulse_depth: float = 0.28
    texture_sd: float = 0.15
    texture_min: float = 0.5
    texture_max: float = 1.5
    blur_width: int = 5
    blur_passes: int = 2
    floor_in: float = 0.25
    quantum_mm: float = 0.01


def box_blur(a: FloatArray, k: int) -> FloatArray:
    """Separable box blur with edge padding.

    Args:
        a: 2-D array.
        k: Odd window width in cells.

    Returns:
        Blurred array with the same shape as ``a``.

    Raises:
        ValueError: If ``k`` is not a positive odd integer.
    """
    if k < 1 or k % 2 == 0:
        raise ValueError("blur width must be a positive odd integer")
    p = k // 2
    out = np.pad(a, p, mode="edge")
    c = np.cumsum(out, axis=0)
    c = np.concatenate([np.zeros((1, c.shape[1])), c], axis=0)
    out = (c[k:] - c[:-k]) / k
    c = np.cumsum(out, axis=1)
    c = np.concatenate([np.zeros((c.shape[0], 1)), c], axis=1)
    result: FloatArray = (c[:, k:] - c[:, :-k]) / k
    return result


def _smoothstep(x: FloatArray) -> FloatArray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def project_onto_path(
    x: FloatArray, y: FloatArray, pts: list[tuple[float, float]]
) -> tuple[FloatArray, FloatArray]:
    """Project points onto a polyline in a local kilometre plane.

    Args:
        x: Easting of each point, km.
        y: Northing of each point, km.
        pts: Polyline vertices as ``(x, y)`` km.

    Returns:
        ``(distance_km, along_fraction)``: distance from each point to the nearest
        spot on the polyline, and where that spot falls along it (0 = start, 1 = end).
    """
    seg_len = [math.dist(a, b) for a, b in itertools.pairwise(pts)]
    total = sum(seg_len)
    best_d = np.full(x.shape, np.inf)
    best_s = np.zeros(x.shape)
    cum = 0.0
    for (ax, ay), (bx, by), length in zip(pts, pts[1:], seg_len, strict=False):
        abx, aby = bx - ax, by - ay
        t = np.clip(((x - ax) * abx + (y - ay) * aby) / (length**2), 0.0, 1.0)
        d = np.hypot(x - (ax + t * abx), y - (ay + t * aby))
        closer = d < best_d
        best_d = np.where(closer, d, best_d)
        best_s = np.where(closer, (cum + t * length) / total, best_s)
        cum += length
    return best_d, best_s


def rasterize_storm(
    storm: SyntheticStorm,
    lattice: HailGrid,
    rng: np.random.Generator,
    shape: StormShape | None = None,
) -> FloatGrid:
    """Render one storm onto a lattice as MRMS-style millimetres.

    The in-grid maximum is rescaled to equal ``storm.peak_in`` exactly, so the
    fixture's numbers are ground truth that tests can assert against.

    Args:
        storm: Storm definition.
        lattice: Grid whose shape and bounds define the output cells.
        rng: Random generator for the texture noise.
        shape: Swath model constants; defaults to :class:`StormShape`.

    Returns:
        MESH in millimetres with the lattice's shape.
    """
    sh = shape or StormShape()
    lat0 = lattice.bounds.center[0]
    kx = KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat0))
    xs, ys = np.meshgrid(lattice.col_longitudes() * kx, lattice.row_latitudes() * KM_PER_DEG_LAT)

    pts = [(lon * kx, lat * KM_PER_DEG_LAT) for lat, lon in storm.path]
    dist, along_s = project_onto_path(xs, ys, pts)

    cross = np.clip(1.0 - (dist / (storm.width_km / 2)) ** 2, 0.0, None)
    along = _smoothstep(along_s / sh.taper_start) * _smoothstep((1.0 - along_s) / sh.taper_end)
    if storm.core_at is not None:
        core = (
            np.array([[storm.core_at[1] * kx]]),
            np.array([[storm.core_at[0] * KM_PER_DEG_LAT]]),
        )
        _, s_core = project_onto_path(core[0], core[1], pts)
        phase = -2 * math.pi * storm.pulses * float(s_core[0, 0])
    else:
        phase = float(rng.uniform(0, 2 * math.pi))
    pulse = (
        1 - sh.pulse_depth + sh.pulse_depth * np.cos(2 * math.pi * storm.pulses * along_s + phase)
    )

    noise: FloatArray = rng.standard_normal(xs.shape)
    for _ in range(sh.blur_passes):
        noise = box_blur(noise, sh.blur_width)
    noise = (noise - noise.mean()) / (noise.std() or 1.0)
    texture = np.clip(1.0 + sh.texture_sd * noise, sh.texture_min, sh.texture_max)

    raw = cross * along * pulse * texture
    peak = float(raw.max())
    if peak <= 0:
        return np.zeros(lattice.shape, dtype=np.float32)
    inches = raw / peak * storm.peak_in
    inches[inches < sh.floor_in] = 0.0
    mm: FloatGrid = (np.round(inches * MM_PER_INCH / sh.quantum_mm) * sh.quantum_mm).astype(
        np.float32
    )
    return mm


# Invented, deliberately generic names. Addresses carry no city, state or ZIP.
STREET_NAMES = (
    "Bluestem", "Sideoats", "Switchgrass", "Meadowlark", "Kestrel", "Sandhill", "Cottonwood",
    "Yucca", "Windmill", "Harvest", "Prairie Rose", "Buffalo Grass", "Sunflower", "Plover",
)  # fmt: skip
AVENUE_NAMES = tuple(
    f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')} Ave"
    for n in range(1, 17)
)


@dataclass(frozen=True)
class TownSpec:
    """Assumptions behind the fictional town. Change a field, get a different town.

    Attributes:
        block_lat_deg: Street spacing north-south (~500 m).
        block_lon_deg: Avenue spacing east-west (~500 m).
        n_streets: East-west streets in the core grid.
        n_avenues: North-south avenues in the core grid.
        lots_per_street_block: Lots per side per block along a street.
        lots_per_avenue_block: Lots per side per block along an avenue.
        street_setback_deg: Latitude offset of a lot from its street centreline.
        avenue_setback_deg: Longitude offset of a lot from its avenue centreline.
        core_share: Fraction of properties on the gridded core.
        rural_extent: Rural lots reach this multiple of the core half-size.
        max_properties: Largest town this spec can generate.
        rural_attempts_per_lot: Random draws allowed per rural lot before giving up.
        section_line_deg: Spacing of rural roads (~1 mile).
        rural_number_base: Smallest rural house number.
        rural_number_scale: House-number growth per degree east or west of centre.
        rural_number_jitter: Random spread added to rural house numbers.
        jitter_deg: Standard deviation of geocode jitter.
        core_year_base: Year built at the town centre.
        core_year_span: Extra years added from centre to the core's edge.
        core_year_sd: Spread of core year built.
        core_year_min: Earliest core year built.
        rural_year_mean: Mean rural year built.
        rural_year_sd: Spread of rural year built.
        rural_year_min: Earliest rural year built.
        reroof_probability: Chance a property has a recorded roof replacement.
        reroof_min_age_yrs: A roof is not replaced before the house is this old.
        reroof_earliest_year: Earliest year a replacement can be on record.
        core_value_base: Typical core assessed value.
        rural_value_base: Typical rural assessed value.
        value_sigma: Log-normal spread of assessed value.
        value_per_year: Relative value gain per year of newer construction.
        value_reference_year: Year at which the newer-construction premium is zero.
        value_min: Lower clip of assessed value.
        value_max: Upper clip of assessed value.
        core_owner_occupied: Owner-occupancy rate in the core.
        rural_owner_occupied: Owner-occupancy rate on acreages.
        core_type_mix: ``(type, cumulative probability)`` for the core.
        rural_type_mix: ``(type, cumulative probability)`` for acreages.
    """

    block_lat_deg: float = 0.0045
    block_lon_deg: float = 0.0058
    n_streets: int = 13
    n_avenues: int = 14
    lots_per_street_block: int = 7
    lots_per_avenue_block: int = 5
    street_setback_deg: float = 0.00035
    avenue_setback_deg: float = 0.00045
    core_share: float = 0.75
    rural_extent: float = 2.2
    max_properties: int = 6000
    rural_attempts_per_lot: int = 50
    section_line_deg: float = 0.0145
    rural_number_base: int = 1000
    rural_number_scale: float = 30_000.0
    rural_number_jitter: int = 40
    jitter_deg: float = 0.00004
    core_year_base: float = 1932.0
    core_year_span: float = 58.0
    core_year_sd: float = 11.0
    core_year_min: int = 1900
    rural_year_mean: float = 1994.0
    rural_year_sd: float = 14.0
    rural_year_min: int = 1950
    reroof_probability: float = 0.42
    reroof_min_age_yrs: int = 10
    reroof_earliest_year: int = 2004
    core_value_base: float = 135_000.0
    rural_value_base: float = 245_000.0
    value_sigma: float = 0.33
    value_per_year: float = 0.007
    value_reference_year: int = 1960
    value_min: float = 40_000.0
    value_max: float = 1_500_000.0
    core_owner_occupied: float = 0.70
    rural_owner_occupied: float = 0.86
    core_type_mix: tuple[tuple[str, float], ...] = (
        ("single_family", 0.84),
        ("multi_family", 0.94),
        ("commercial", 1.0),
    )
    rural_type_mix: tuple[tuple[str, float], ...] = (("single_family", 0.97), ("commercial", 1.0))


@dataclass(frozen=True)
class _Lot:
    lat: float
    lon: float
    address: str
    rural: bool
    radius: float  # 0 at the town centre, 1 at the core's corner


def _core_lots(center: tuple[float, float], spec: TownSpec) -> list[_Lot]:
    clat, clon = center
    half_lat = (spec.n_streets // 2) * spec.block_lat_deg
    half_lon = (spec.n_avenues // 2) * spec.block_lon_deg
    lots: list[_Lot] = []

    def radius(lat: float, lon: float) -> float:
        return min(1.0, math.hypot((lat - clat) / half_lat, (lon - clon) / half_lon) / math.sqrt(2))

    for j in range(spec.n_streets):
        lat = clat + (j - spec.n_streets // 2) * spec.block_lat_deg
        street = f"{STREET_NAMES[j % len(STREET_NAMES)]} St"
        for i in range(spec.n_avenues - 1):
            lon0 = clon + (i - spec.n_avenues // 2) * spec.block_lon_deg
            for k in range(1, spec.lots_per_street_block + 1):
                for side in (0, 1):
                    lot_lat = lat + (spec.street_setback_deg if side else -spec.street_setback_deg)
                    lot_lon = lon0 + k * spec.block_lon_deg / (spec.lots_per_street_block + 1)
                    number = (i + 1) * 100 + 2 * k + side
                    lots.append(
                        _Lot(
                            lot_lat, lot_lon, f"{number} {street}", False, radius(lot_lat, lot_lon)
                        )
                    )
    for i in range(spec.n_avenues):
        lon = clon + (i - spec.n_avenues // 2) * spec.block_lon_deg
        avenue = AVENUE_NAMES[i % len(AVENUE_NAMES)]
        for j in range(spec.n_streets - 1):
            lat0 = clat + (j - spec.n_streets // 2) * spec.block_lat_deg
            for k in range(1, spec.lots_per_avenue_block + 1):
                for side in (0, 1):
                    lot_lon = lon + (spec.avenue_setback_deg if side else -spec.avenue_setback_deg)
                    lot_lat = lat0 + k * spec.block_lat_deg / (spec.lots_per_avenue_block + 1)
                    number = (j + 1) * 100 + 2 * k + side
                    lots.append(
                        _Lot(
                            lot_lat, lot_lon, f"{number} {avenue}", False, radius(lot_lat, lot_lon)
                        )
                    )
    return lots


def _rural_lots(
    n: int, center: tuple[float, float], spec: TownSpec, rng: np.random.Generator
) -> list[_Lot]:
    clat, clon = center
    half_lat = (spec.n_streets // 2) * spec.block_lat_deg
    half_lon = (spec.n_avenues // 2) * spec.block_lon_deg
    lots: list[_Lot] = []
    used: set[str] = set()
    for _ in range(spec.rural_attempts_per_lot * n):
        if len(lots) == n:
            break
        lat = clat + float(rng.uniform(-spec.rural_extent, spec.rural_extent)) * half_lat
        lon = clon + float(rng.uniform(-spec.rural_extent, spec.rural_extent)) * half_lon
        if abs(lat - clat) <= half_lat and abs(lon - clon) <= half_lon:
            continue  # acreages sit outside the gridded core
        road = int(abs(lat - clat) / spec.section_line_deg) + 1
        number = int(
            spec.rural_number_base
            + abs(lon - clon) * spec.rural_number_scale
            + rng.integers(0, spec.rural_number_jitter)
        )
        address = f"{number} {'N' if lat > clat else 'S'} Rural Rd {road}"
        if address in used:
            continue
        used.add(address)
        lots.append(_Lot(lat, lon, address, True, 1.0))
    if len(lots) < n:
        raise ValueError(f"placed only {len(lots)} of {n} rural lots; lower n or widen the town")
    return lots


def _pick(mix: tuple[tuple[str, float], ...], u: float) -> str:
    return next((name for name, cumulative in mix if u < cumulative), mix[-1][0])


def _make_property(
    idx: int, lot: _Lot, sp: TownSpec, as_of_year: int, rng: np.random.Generator
) -> Property:
    jitter = rng.normal(0, sp.jitter_deg, size=2)
    if lot.rural:
        built = rng.normal(sp.rural_year_mean, sp.rural_year_sd)
        year_built = int(np.clip(built, sp.rural_year_min, as_of_year - 1))
    else:
        built = sp.core_year_base + sp.core_year_span * lot.radius + rng.normal(0, sp.core_year_sd)
        year_built = int(np.clip(built, sp.core_year_min, as_of_year - 1))
    roof_year = None
    if rng.random() < sp.reroof_probability:
        lo = max(year_built + sp.reroof_min_age_yrs, sp.reroof_earliest_year)
        if lo <= as_of_year - 1:
            roof_year = int(rng.integers(lo, as_of_year))
    base = sp.rural_value_base if lot.rural else sp.core_value_base
    premium = 1 + sp.value_per_year * (year_built - sp.value_reference_year)
    value = base * math.exp(float(rng.normal(0, sp.value_sigma))) * premium
    occupied = sp.rural_owner_occupied if lot.rural else sp.core_owner_occupied
    mix = sp.rural_type_mix if lot.rural else sp.core_type_mix
    return Property(
        property_id=f"SYN-{idx:05d}",
        lat=round(lot.lat + float(jitter[0]), 6),
        lon=round(lot.lon + float(jitter[1]), 6),
        year_built=year_built,
        address=lot.address,
        roof_year=roof_year,
        assessed_value=float(np.clip(round(value, -2), sp.value_min, sp.value_max)),
        owner_occupied=bool(rng.random() < occupied),
        property_type=_pick(mix, float(rng.random())),
    )


def generate_properties(
    n: int,
    center: tuple[float, float],
    as_of_year: int,
    rng: np.random.Generator,
    spec: TownSpec | None = None,
) -> list[Property]:
    """Generate a fictional town of ``n`` properties.

    Most lots sit on a gridded core (older, denser, more rentals); the rest are
    rural acreages (newer, pricier, mostly owner-occupied). IDs are ``SYN-00001``
    onward in north-to-south, west-to-east reading order.

    Args:
        n: Number of properties.
        center: ``(lat, lon)`` of the town centre.
        as_of_year: Year the data is "as of"; nothing is built or re-roofed after it.
        rng: Random generator.
        spec: Town assumptions; defaults to :class:`TownSpec`.

    Returns:
        The synthetic properties.

    Raises:
        ValueError: If ``n`` is outside 1..``spec.max_properties`` or the lots run out.
    """
    sp = spec or TownSpec()
    if not 1 <= n <= sp.max_properties:
        raise ValueError(f"n must be between 1 and {sp.max_properties}, got {n}")
    core = _core_lots(center, sp)
    n_core = min(len(core), round(n * sp.core_share))
    picks = rng.choice(len(core), size=n_core, replace=False)
    lots = [core[int(i)] for i in picks] + _rural_lots(n - n_core, center, sp, rng)
    lots.sort(key=lambda lot: (-round(lot.lat, 5), lot.lon))

    return [_make_property(i, lot, sp, as_of_year, rng) for i, lot in enumerate(lots, start=1)]


@dataclass(frozen=True)
class ProviderSpec:
    """Assumptions behind the fictional service providers.

    Attributes:
        count: Number of providers.
        edge_cases: ``(rating, review_count)`` pairs used first, in order. The first two are
            the canonical shrinkage case: 5.0 stars from 3 reviews versus 4.8 from 400.
        min_distance_km: Closest a provider base sits to the town centre.
        max_distance_km: Farthest a provider base sits from the town centre.
        service_radii_km: Radii to choose from.
        log_reviews_mean: Mean of log review count for random providers.
        log_reviews_sd: Spread of log review count for random providers.
        rating_mean: Mean rating for random providers.
        rating_sd: Rating spread for random providers.
        rating_min: Lower clip of random ratings.
    """

    count: int = 14
    edge_cases: tuple[tuple[float | None, int], ...] = (
        (5.0, 3), (4.8, 400), (4.9, 14), (4.6, 160), (4.2, 65),
        (3.8, 230), (4.95, 42), (None, 0), (4.7, 7), (4.4, 310),
    )  # fmt: skip
    min_distance_km: float = 3.0
    max_distance_km: float = 45.0
    service_radii_km: tuple[float, ...] = (40.0, 60.0, 80.0)
    log_reviews_mean: float = 3.6
    log_reviews_sd: float = 1.0
    rating_mean: float = 4.45
    rating_sd: float = 0.3
    rating_min: float = 2.8


def generate_providers(
    center: tuple[float, float], rng: np.random.Generator, spec: ProviderSpec | None = None
) -> list[Provider]:
    """Generate fictional providers around ``center``, named ``Synthetic Provider NN``.

    Args:
        center: ``(lat, lon)`` of the town centre.
        rng: Random generator.
        spec: Provider assumptions; defaults to :class:`ProviderSpec`.

    Returns:
        The synthetic providers.
    """
    sp = spec or ProviderSpec()
    clat, clon = center
    km_per_deg_lon = KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(clat))
    out: list[Provider] = []
    for i in range(sp.count):
        rating: float | None
        if i < len(sp.edge_cases):
            rating, count = sp.edge_cases[i]
        else:
            count = int(rng.lognormal(sp.log_reviews_mean, sp.log_reviews_sd))
            drawn = float(
                np.clip(rng.normal(sp.rating_mean, sp.rating_sd), sp.rating_min, MAX_STARS)
            )
            rating = round(drawn, 1) if count else None
        dist = float(rng.uniform(sp.min_distance_km, sp.max_distance_km))
        bearing = float(rng.uniform(0, 2 * math.pi))
        out.append(
            Provider(
                provider_id=f"PRV-{i + 1:02d}",
                name=f"Synthetic Provider {i + 1:02d}",
                lat=round(clat + dist * math.cos(bearing) / KM_PER_DEG_LAT, 5),
                lon=round(clon + dist * math.sin(bearing) / km_per_deg_lon, 5),
                service_radius_km=float(rng.choice(sp.service_radii_km)),
                review_count=count,
                rating=rating,
            )
        )
    return out
