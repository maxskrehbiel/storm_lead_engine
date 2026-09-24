"""Synthetic storms, town and providers: determinism and known ground truth."""

import numpy as np
import pytest

from storm_lead_engine.grid import MM_PER_INCH, HailGrid
from storm_lead_engine.properties import PROPERTY_TYPES
from storm_lead_engine.synthetic import (
    ProviderSpec,
    StormShape,
    SyntheticStorm,
    TownSpec,
    box_blur,
    generate_properties,
    generate_providers,
    project_onto_path,
    rasterize_storm,
)

CENTER = (0.0, 0.0)


def _storm(**kw: object) -> SyntheticStorm:
    base: dict[str, object] = {
        "storm_id": "T",
        "date": "2024-06-01",
        "path": ((-0.1, -0.3), (0.1, 0.3)),
        "width_km": 8.0,
        "peak_in": 2.0,
        "seed": 3,
    }
    base.update(kw)
    return SyntheticStorm(**base)  # type: ignore[arg-type]


@pytest.fixture
def lattice(season) -> HailGrid:
    return HailGrid.zeros(season.region)


def test_rasterized_peak_equals_spec_and_is_deterministic(lattice) -> None:
    storm = _storm()
    mm_a = rasterize_storm(storm, lattice, np.random.default_rng(1))
    mm_b = rasterize_storm(storm, lattice, np.random.default_rng(1))
    assert np.array_equal(mm_a, mm_b)
    assert mm_a.max() / MM_PER_INCH == pytest.approx(2.0, abs=1e-5)
    nonzero = mm_a[mm_a > 0] / MM_PER_INCH
    assert nonzero.min() >= StormShape().floor_in - 1e-3  # floor, then 0.01 mm quantization
    step = StormShape().quantum_mm
    assert np.allclose(mm_a / step, np.round(mm_a / step), atol=1e-2)


def test_core_at_places_the_maximum_near_the_core(lattice) -> None:
    storm = _storm(core_at=(0.0, 0.0), pulses=2.0)
    mm = rasterize_storm(storm, lattice, np.random.default_rng(0))
    row, col = np.unravel_index(int(np.argmax(mm)), mm.shape)
    lat, lon = lattice.row_latitudes()[row], lattice.col_longitudes()[col]
    assert abs(lat) < 0.08 and abs(lon) < 0.10


def test_storm_outside_lattice_renders_nothing(lattice) -> None:
    far = _storm(path=((10.0, 10.0), (10.5, 11.0)))
    assert rasterize_storm(far, lattice, np.random.default_rng(0)).max() == 0


def test_storm_validation() -> None:
    with pytest.raises(ValueError, match="2 vertices"):
        _storm(path=((39.0, -102.0),))
    with pytest.raises(ValueError, match="positive"):
        _storm(width_km=0.0)


def test_box_blur_keeps_shape_and_constants() -> None:
    a = np.full((6, 9), 2.5)
    out = box_blur(a, 5)
    assert out.shape == a.shape
    assert np.allclose(out, 2.5)
    with pytest.raises(ValueError, match="odd"):
        box_blur(a, 4)


def test_project_onto_path_distance_and_fraction() -> None:
    x = np.array([[5.0, 0.0, 12.0]])
    y = np.array([[3.0, 0.0, 0.0]])
    d, s = project_onto_path(x, y, [(0.0, 0.0), (10.0, 0.0)])
    assert d.tolist() == [[3.0, 0.0, 2.0]]
    assert s.tolist() == [[0.5, 0.0, 1.0]]


def test_generate_properties_is_seeded_and_valid() -> None:
    a = generate_properties(500, CENTER, 2025, np.random.default_rng(11))
    b = generate_properties(500, CENTER, 2025, np.random.default_rng(11))
    c = generate_properties(500, CENTER, 2025, np.random.default_rng(12))
    assert a == b
    assert a != c
    assert len(a) == 500
    assert len({p.property_id for p in a}) == 500
    assert len({p.address for p in a}) == 500
    assert all(p.property_type in PROPERTY_TYPES for p in a)
    assert all(p.year_built < 2025 for p in a)
    assert all(p.roof_year is None or p.year_built <= p.roof_year < 2025 for p in a)
    assert all(abs(p.lat - CENTER[0]) < 0.1 and abs(p.lon - CENTER[1]) < 0.1 for p in a)
    rural = [p for p in a if "Rural" in p.address]
    assert 0.2 < len(rural) / len(a) < 0.3


def test_generate_properties_validates_capacity() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="between 1 and"):
        generate_properties(0, CENTER, 2025, rng)
    with pytest.raises(ValueError, match="between 1 and"):
        generate_properties(TownSpec().max_properties + 1, CENTER, 2025, rng)
    # rural lots must fall outside the core but inside rural_extent: impossible at 1.0
    with pytest.raises(ValueError, match="rural lots"):
        generate_properties(100, CENTER, 2025, rng, TownSpec(rural_extent=1.0))


def test_generate_properties_at_capacity() -> None:
    spec = TownSpec()
    props = generate_properties(spec.max_properties, CENTER, 2025, np.random.default_rng(4))
    assert len({p.address for p in props}) == spec.max_properties


def test_generate_providers_includes_edge_cases() -> None:
    spec = ProviderSpec(count=12)
    provs = generate_providers(CENTER, np.random.default_rng(5), spec)
    assert len(provs) == 12
    assert (provs[0].rating, provs[0].review_count) == (5.0, 3)
    assert (provs[1].rating, provs[1].review_count) == (4.8, 400)
    unrated = [p for p in provs if p.review_count == 0]
    assert unrated and all(p.rating is None for p in unrated)
    assert all(p.name.startswith("Synthetic Provider") for p in provs)
