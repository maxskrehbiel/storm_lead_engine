"""Footprint tracing: exact rings, holes, orientation and area."""

import itertools

import numpy as np
import pytest

from storm_lead_engine.swaths import (
    directed_edges,
    feature_collection,
    footprints,
    mask_to_multipolygon,
    signed_area,
    trace_rings,
)


def _ring_area_lonlat(ring: list[list[float]]) -> float:
    s = 0.0
    for (x1, y1), (x2, y2) in itertools.pairwise(ring):
        s += x1 * y2 - x2 * y1
    return s / 2


def test_single_cell_is_one_counter_clockwise_square(make_grid) -> None:
    g = make_grid([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    geom = mask_to_multipolygon(g.inches >= 1, g)
    assert geom["type"] == "MultiPolygon"
    assert len(geom["coordinates"]) == 1
    (outer,) = geom["coordinates"][0]
    assert len(outer) == 5 and outer[0] == outer[-1]
    assert sorted({tuple(p) for p in outer}) == [
        (0.01, 0.98),
        (0.01, 0.99),
        (0.02, 0.98),
        (0.02, 0.99),
    ]
    assert _ring_area_lonlat(outer) > 0  # RFC 7946: exterior rings counter-clockwise


def test_donut_has_one_clockwise_hole(make_grid) -> None:
    g = make_grid([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    geom = mask_to_multipolygon(g.inches >= 1, g)
    assert len(geom["coordinates"]) == 1
    outer, hole = geom["coordinates"][0]
    assert _ring_area_lonlat(outer) > 0
    assert _ring_area_lonlat(hole) < 0
    assert abs(_ring_area_lonlat(outer)) == pytest.approx(9e-4)
    assert abs(_ring_area_lonlat(hole)) == pytest.approx(1e-4)


def test_island_inside_a_hole_is_its_own_polygon(make_grid) -> None:
    ring = [[1] * 5, [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 1], [1] * 5]
    g = make_grid(ring)
    geom = mask_to_multipolygon(g.inches >= 1, g)
    polys = sorted(geom["coordinates"], key=len)
    assert [len(p) for p in polys] == [1, 2]  # island (no hole), frame (one hole)


def test_diagonal_cells_trace_as_two_rings() -> None:
    mask = np.array([[True, False], [False, True]])
    rings = trace_rings(directed_edges(mask))
    assert len(rings) == 2
    assert all(signed_area(r) == 1.0 for r in rings)


def test_footprint_area_counts_and_nesting(day_grids) -> None:
    grid = day_grids["2022-07-08"]
    fps = footprints(grid, (1.0, 1.5, 2.0), "EV-20220708")
    assert [f.threshold_in for f in fps] == [1.0, 1.5, 2.0]
    assert fps[0].n_cells >= fps[1].n_cells >= fps[2].n_cells > 0
    per_row = grid.cell_areas_km2()[:, None]
    expected = float(((grid.inches >= 1.0) * per_row).sum())
    assert fps[0].area_km2 == pytest.approx(expected)
    # a 0.01-degree cell near the equator is ~1.106 km x ~1.113 km
    assert fps[0].area_km2 == pytest.approx(fps[0].n_cells * 1.231, rel=0.002)


def test_unreached_thresholds_are_skipped(make_grid) -> None:
    fps = footprints(make_grid([[1.2]]), (1.0, 1.5), "x")
    assert [f.threshold_in for f in fps] == [1.0]
    fc = feature_collection(fps, {"synthetic_data": True})
    assert fc["type"] == "FeatureCollection"
    assert fc["properties"] == {"synthetic_data": True}
    assert fc["features"][0]["properties"]["n_cells"] == 1
    assert "properties" not in feature_collection(fps)
