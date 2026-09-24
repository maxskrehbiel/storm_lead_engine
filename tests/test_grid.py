"""HailGrid geometry, sentinel handling and sampling."""

import numpy as np
import pytest

from storm_lead_engine.config import BBox
from storm_lead_engine.grid import MM_PER_INCH, HailGrid, snap_bbox


def test_snap_bbox_expands_to_lattice() -> None:
    snapped = snap_bbox(BBox(0.004, -0.004, 0.019, 0.013))
    assert snapped.as_tuple() == (0.0, -0.01, 0.02, 0.02)
    # already aligned boxes are unchanged despite float noise
    assert snap_bbox(BBox(-0.54, -0.46, 0.54, 0.46)).as_tuple() == (-0.54, -0.46, 0.54, 0.46)


def test_from_mm_turns_sentinels_into_zero() -> None:
    mm = np.array([[-3.0, -999.0], [np.nan, 25.4 * 2]])
    g = HailGrid.from_mm(mm, BBox(0.0, 0.0, 0.02, 0.02))
    assert g.inches.tolist() == [[0.0, 0.0], [0.0, pytest.approx(2.0)]]
    assert g.inches.dtype == np.float32


@pytest.mark.parametrize("bad", [np.array([1.0, 2.0]), np.array([[-1.0]]), np.array([[np.inf]])])
def test_rejects_invalid_arrays(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        HailGrid(bad, BBox(0.0, 0.0, 0.01, 0.01))


def test_zeros_shape_and_resolution() -> None:
    g = HailGrid.zeros(BBox(-0.7, -0.45, 0.7, 0.45))
    assert g.shape == (90, 140)
    assert g.res_lat == pytest.approx(0.01)
    assert g.res_lon == pytest.approx(0.01)


def test_cell_of_edges_and_outside(make_grid) -> None:
    g = make_grid([[0, 0, 0], [0, 0, 0]])  # rows cover lat 0.98-1.00, cols lon 0.00-0.03
    assert g.cell_of(1.0, 0.0) == (0, 0)  # NW corner
    assert g.cell_of(0.98, 0.03) == (1, 2)  # SE corner clamps into the grid
    assert g.cell_of(0.995, 0.015) == (0, 1)
    assert g.cell_of(2.0, 0.0) is None


def test_anchor_cell_prefers_worst_neighbour_and_keeps_own_on_ties(make_grid) -> None:
    g = make_grid([[0, 0, 0], [0, 1, 0], [0, 0, 3]])
    centre = (0.985, 0.015)
    assert g.anchor_cell(*centre, neighborhood=1) == (2, 2)
    assert g.anchor_cell(*centre, neighborhood=0) == (1, 1)
    assert g.sample(*centre, neighborhood=1) == pytest.approx(3.0)
    flat = make_grid([[0, 0], [0, 0]])
    assert flat.anchor_cell(0.995, 0.005, neighborhood=1) == (0, 0)
    assert flat.sample(5.0, 0.005) == 0.0


def test_pixel_max_requires_shared_lattice(make_grid) -> None:
    a = make_grid([[1, 0], [0, 2]])
    b = make_grid([[0, 3], [1, 0]])
    assert HailGrid.pixel_max([a, b]).inches.tolist() == [[1, 3], [1, 2]]
    shifted = make_grid([[1, 0], [0, 2]], west=1.0)
    with pytest.raises(ValueError, match="different lattices"):
        HailGrid.pixel_max([a, shifted])
    with pytest.raises(ValueError, match="at least one"):
        HailGrid.pixel_max([])


def test_cell_area_shrinks_toward_the_pole() -> None:
    g = HailGrid.zeros(BBox(0.0, 30.0, 0.01, 60.0))
    areas = g.cell_areas_km2()
    assert areas[0] < areas[-1]  # row 0 is the northern edge
    assert areas[-1] == pytest.approx(1.1057 * 1.1132 * np.cos(np.radians(30.005)), rel=1e-3)


def test_max_and_mm_constant() -> None:
    g = HailGrid.from_mm(np.array([[MM_PER_INCH * 1.75]]), BBox(0.0, 0.0, 0.01, 0.01))
    assert g.max_in == pytest.approx(1.75)
