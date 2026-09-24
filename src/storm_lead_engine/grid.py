"""The hail raster: a north-up grid of hail size in inches on the MRMS 0.01-degree lattice.

Raw MRMS MESH is in millimetres, with negative sentinels (-3 = no radar coverage,
-999 = missing). Ingest converts everything to a clean :class:`HailGrid`, so the
synthetic fixture and real data meet at the same type.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._types import FloatArray, FloatGrid
from .config import BBox

MM_PER_INCH = 25.4
MRMS_RES_DEG = 0.01
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_AT_EQUATOR = 111.320
_SNAP_DIGITS = 6


def snap_bbox(bbox: BBox, res: float = MRMS_RES_DEG) -> BBox:
    """Expand a box outward to the nearest lattice edges.

    Snapping means every read of the same region lands on the same rows and
    columns, so grids from different days stack cell-for-cell.

    Args:
        bbox: Box to snap.
        res: Lattice spacing in degrees.

    Returns:
        The smallest lattice-aligned box that contains ``bbox``.
    """

    def down(x: float) -> float:
        return round(math.floor(round(x / res, _SNAP_DIGITS)) * res, _SNAP_DIGITS)

    def up(x: float) -> float:
        return round(math.ceil(round(x / res, _SNAP_DIGITS)) * res, _SNAP_DIGITS)

    return BBox(down(bbox.west), down(bbox.south), up(bbox.east), up(bbox.north))


@dataclass
class HailGrid:
    """Hail size in inches. Row 0 is the northern edge, column 0 the western edge.

    Attributes:
        inches: 2-D float32 array of non-negative hail sizes.
        bounds: Outer edges of the grid.
    """

    inches: FloatGrid
    bounds: BBox

    def __post_init__(self) -> None:
        arr = np.asarray(self.inches, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError("HailGrid needs a non-empty 2-D array")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("HailGrid values must be finite and >= 0 (use from_mm for raw MRMS)")
        self.inches = arr

    @classmethod
    def from_mm(cls, mm: npt.ArrayLike, bounds: BBox) -> HailGrid:
        """Build a grid from raw MRMS millimetres; sentinels and NaNs become 0.

        Args:
            mm: 2-D array of MESH values in millimetres.
            bounds: Outer edges of the array.

        Returns:
            The grid in inches.
        """
        arr = np.asarray(mm, dtype=np.float32)
        clean = np.where(np.isfinite(arr) & (arr > 0), arr, np.float32(0.0))
        return cls((clean / MM_PER_INCH).astype(np.float32), bounds)

    @classmethod
    def zeros(cls, bounds: BBox, res: float = MRMS_RES_DEG) -> HailGrid:
        """Return an all-zero grid covering ``bounds`` at spacing ``res``."""
        h = round((bounds.north - bounds.south) / res)
        w = round((bounds.east - bounds.west) / res)
        return cls(np.zeros((h, w), dtype=np.float32), bounds)

    @staticmethod
    def pixel_max(grids: Iterable[HailGrid]) -> HailGrid:
        """Cell-wise maximum of grids that share a lattice.

        Args:
            grids: One or more grids with identical shape and bounds.

        Returns:
            A grid holding the maximum of each cell.

        Raises:
            ValueError: If no grids are given or their lattices differ.
        """
        items = list(grids)
        if not items:
            raise ValueError("pixel_max needs at least one grid")
        first = items[0]
        for g in items[1:]:
            if not first.same_lattice(g):
                raise ValueError("grids are on different lattices; snap the bbox first")
        stacked = np.maximum.reduce([g.inches for g in items])
        return HailGrid(stacked, first.bounds)

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape as ``(rows, cols)``."""
        rows, cols = self.inches.shape
        return int(rows), int(cols)

    @property
    def res_lat(self) -> float:
        """Cell height in degrees of latitude."""
        return (self.bounds.north - self.bounds.south) / self.shape[0]

    @property
    def res_lon(self) -> float:
        """Cell width in degrees of longitude."""
        return (self.bounds.east - self.bounds.west) / self.shape[1]

    @property
    def max_in(self) -> float:
        """Largest hail size anywhere on the grid, inches."""
        return float(self.inches.max())

    def same_lattice(self, other: HailGrid, tol: float = 1e-6) -> bool:
        """Return True if ``other`` has the same shape and bounds (within ``tol`` degrees)."""
        return self.shape == other.shape and all(
            abs(a - b) <= tol
            for a, b in zip(self.bounds.as_tuple(), other.bounds.as_tuple(), strict=True)
        )

    def cell_of(self, lat: float, lon: float) -> tuple[int, int] | None:
        """Return the ``(row, col)`` containing a point, or None if it is outside the grid."""
        if not self.bounds.contains(lat, lon):
            return None
        h, w = self.shape
        row = min(int((self.bounds.north - lat) / self.res_lat), h - 1)
        col = min(int((lon - self.bounds.west) / self.res_lon), w - 1)
        return row, col

    def row_latitudes(self) -> FloatArray:
        """Latitude of each row's cell centres."""
        rows = np.arange(self.shape[0], dtype=np.float64)
        return np.asarray(self.bounds.north - (rows + 0.5) * self.res_lat, dtype=np.float64)

    def col_longitudes(self) -> FloatArray:
        """Longitude of each column's cell centres."""
        cols = np.arange(self.shape[1], dtype=np.float64)
        return np.asarray(self.bounds.west + (cols + 0.5) * self.res_lon, dtype=np.float64)

    def cell_areas_km2(self) -> FloatArray:
        """Area of one cell in each row, km^2 (cells narrow toward the pole)."""
        lat_km = self.res_lat * KM_PER_DEG_LAT
        lon_km = self.res_lon * KM_PER_DEG_LON_AT_EQUATOR * np.cos(np.radians(self.row_latitudes()))
        return lat_km * lon_km

    def anchor_cell(self, lat: float, lon: float, neighborhood: int = 1) -> tuple[int, int] | None:
        """Return the worst (maximum) cell within ``neighborhood`` cells of a point.

        A geocoded point can sit a few hundred metres from the roof, and a radar
        cell is about 1 km across, so the (2k+1)^2 window absorbs that error. Ties,
        including an all-zero window, keep the point's own cell so the choice is stable.

        Args:
            lat: Point latitude.
            lon: Point longitude.
            neighborhood: Window half-size in cells.

        Returns:
            ``(row, col)`` of the chosen cell, or None if the point is outside the grid.
        """
        cell = self.cell_of(lat, lon)
        if cell is None:
            return None
        row, col = cell
        h, w = self.shape
        r0, r1 = max(0, row - neighborhood), min(h, row + neighborhood + 1)
        c0, c1 = max(0, col - neighborhood), min(w, col + neighborhood + 1)
        sub = self.inches[r0:r1, c0:c1]
        if sub.max() <= self.inches[row, col]:
            return row, col
        lr, lc = np.unravel_index(int(np.argmax(sub)), sub.shape)
        return r0 + int(lr), c0 + int(lc)

    def sample(self, lat: float, lon: float, neighborhood: int = 0) -> float:
        """Return the maximum hail (inches) within ``neighborhood`` cells; 0 if outside."""
        anchor = self.anchor_cell(lat, lon, neighborhood)
        if anchor is None:
            return 0.0
        return float(self.inches[anchor])
