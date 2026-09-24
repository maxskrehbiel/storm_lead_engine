"""Hail-swath footprints: threshold masks traced into exact, cell-aligned GeoJSON polygons.

Footprints are not smoothed or buffered, so the map shows exactly the cells the
scoring used, and a footprint's area is exact (cell count x cell area).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._types import BoolGrid
from .grid import HailGrid

Vertex = tuple[int, int]  # (row, col) on the lattice of cell corners
Ring = list[Vertex]


@dataclass(frozen=True)
class Footprint:
    """Where hail reached at least ``threshold_in`` during one event (or ever).

    Attributes:
        label: Event id, or ``"max-ever"`` for the all-time footprint.
        threshold_in: Hail-size threshold, inches.
        n_cells: Number of grid cells inside.
        area_km2: Area inside, km^2.
        geometry: GeoJSON MultiPolygon in ``[lon, lat]``.
    """

    label: str
    threshold_in: float
    n_cells: int
    area_km2: float
    geometry: dict[str, Any]

    def to_feature(self) -> dict[str, Any]:
        """Return the footprint as a GeoJSON Feature."""
        return {
            "type": "Feature",
            "properties": {
                "label": self.label,
                "threshold_in": self.threshold_in,
                "n_cells": self.n_cells,
                "area_km2": round(self.area_km2, 2),
            },
            "geometry": self.geometry,
        }


def directed_edges(mask: BoolGrid) -> dict[Vertex, list[Vertex]]:
    """Boundary edges of the true region, oriented so the region is on the left.

    Every true cell contributes each side that borders a false cell or the grid edge.

    Args:
        mask: 2-D boolean grid.

    Returns:
        Start vertex to the end vertices of edges leaving it.
    """
    padded = np.pad(mask, 1, constant_values=False)
    core = padded[1:-1, 1:-1]
    edges: dict[Vertex, list[Vertex]] = defaultdict(list)
    # Each open side becomes one directed edge; the true cell stays on the left.
    # South side, heading east.
    for r, c in zip(*np.nonzero(core & ~padded[2:, 1:-1]), strict=True):
        edges[(int(r) + 1, int(c))].append((int(r) + 1, int(c) + 1))
    # East side, heading north.
    for r, c in zip(*np.nonzero(core & ~padded[1:-1, 2:]), strict=True):
        edges[(int(r) + 1, int(c) + 1)].append((int(r), int(c) + 1))
    # North side, heading west.
    for r, c in zip(*np.nonzero(core & ~padded[:-2, 1:-1]), strict=True):
        edges[(int(r), int(c) + 1)].append((int(r), int(c)))
    # West side, heading south.
    for r, c in zip(*np.nonzero(core & ~padded[1:-1, :-2]), strict=True):
        edges[(int(r), int(c))].append((int(r) + 1, int(c)))
    return dict(edges)


def _turn_rank(d_in: Vertex, d_out: Vertex) -> int:
    # Prefer left, then straight, then right (the row axis points south). Hugging
    # the region keeps two cells that touch only at a corner as two rings, not one
    # ring that pinches itself.
    left = (-d_in[1], d_in[0])
    right = (d_in[1], -d_in[0])
    if d_out == left:
        return 0
    if d_out == d_in:
        return 1
    return 2 if d_out == right else 3


def trace_rings(edges: dict[Vertex, list[Vertex]]) -> list[Ring]:
    """Chain directed edges into closed, simplified rings.

    Args:
        edges: Output of :func:`directed_edges`.

    Returns:
        Closed rings (first vertex repeated at the end) without collinear vertices.
    """
    remaining = {k: list(v) for k, v in edges.items()}
    rings: list[Ring] = []
    while remaining:
        start = next(iter(remaining))
        ring = [start]
        prev, cur = start, remaining[start].pop()
        if not remaining[start]:
            del remaining[start]
        while cur != start:
            ring.append(cur)
            options = remaining[cur]
            d_in = (cur[0] - prev[0], cur[1] - prev[1])
            options.sort(key=lambda nxt: _turn_rank(d_in, (nxt[0] - cur[0], nxt[1] - cur[1])))
            nxt = options.pop(0)
            if not options:
                del remaining[cur]
            prev, cur = cur, nxt
        ring.append(start)
        rings.append(_simplify(ring))
    return rings


def _simplify(ring: Ring) -> Ring:
    pts = ring[:-1]
    n = len(pts)
    keep = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            keep.append(b)
    return [*keep, keep[0]]


def signed_area(ring: Ring) -> float:
    """Shoelace area in ``(x=col, y=-row)`` space; positive means counter-clockwise.

    Args:
        ring: Closed ring of lattice vertices.

    Returns:
        Signed area in cell units.
    """
    s = 0.0
    for (r1, c1), (r2, c2) in itertools.pairwise(ring):
        s += c1 * (-r2) - c2 * (-r1)
    return s / 2


def _inside(point: tuple[float, float], ring: Ring) -> bool:
    # Ray casting. The probe is a half-integer cell centre and ring edges sit on
    # integer lattice lines, so the ray never grazes a vertex.
    pr, pc = point
    inside = False
    for (r1, c1), (r2, c2) in itertools.pairwise(ring):
        if (r1 > pr) != (r2 > pr) and pc < c1 + (pr - r1) * (c2 - c1) / (r2 - r1):
            inside = not inside
    return inside


def _hole_probe(ring: Ring) -> tuple[float, float]:
    # Centre of the (false) cell to the right of the hole ring's first edge.
    (r1, c1), (r2, c2) = ring[0], ring[1]
    dr = (r2 > r1) - (r2 < r1)
    dc = (c2 > c1) - (c2 < c1)
    return (r1 + dr / 2 + dc / 2, c1 + dc / 2 - dr / 2)


def mask_to_multipolygon(mask: BoolGrid, grid: HailGrid) -> dict[str, Any]:
    """Convert a boolean mask on a grid's lattice to a GeoJSON MultiPolygon.

    Outer rings run counter-clockwise and holes clockwise, as RFC 7946 asks.
    Each hole goes to the smallest outer ring that contains it; every hole in a
    mask has one, since a hole is by definition enclosed by true cells.

    Args:
        mask: Boolean grid with the same shape as ``grid``.
        grid: Supplies bounds and cell size.

    Returns:
        A GeoJSON MultiPolygon geometry in ``[lon, lat]``.
    """
    rings = trace_rings(directed_edges(mask.astype(bool)))
    outers = [r for r in rings if signed_area(r) > 0]
    holes = [r for r in rings if signed_area(r) < 0]
    polygons: list[list[Ring]] = [[o] for o in outers]
    areas = [signed_area(o) for o in outers]
    for hole in holes:
        probe = _hole_probe(hole)
        owners = [i for i, o in enumerate(outers) if _inside(probe, o)]
        polygons[min(owners, key=lambda i: areas[i])].append(hole)

    b = grid.bounds

    def to_lonlat(ring: Ring) -> list[list[float]]:
        return [
            [round(b.west + c * grid.res_lon, 5), round(b.north - r * grid.res_lat, 5)]
            for r, c in ring
        ]

    return {
        "type": "MultiPolygon",
        "coordinates": [[to_lonlat(r) for r in poly] for poly in polygons],
    }


def footprints(grid: HailGrid, thresholds: tuple[float, ...], label: str) -> list[Footprint]:
    """Build one footprint per threshold, skipping thresholds nothing reaches.

    Args:
        grid: Hail grid.
        thresholds: Sizes in inches.
        label: Label stored on each footprint.

    Returns:
        Footprints in threshold order.
    """
    row_area = grid.cell_areas_km2()[:, None]
    out = []
    for t in thresholds:
        mask = grid.inches >= t
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(
            Footprint(
                label=label,
                threshold_in=t,
                n_cells=n,
                area_km2=float((mask * row_area).sum()),
                geometry=mask_to_multipolygon(mask, grid),
            )
        )
    return out


def feature_collection(
    fps: list[Footprint], properties: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Wrap footprints in a GeoJSON FeatureCollection with optional top-level properties."""
    fc: dict[str, Any] = {"type": "FeatureCollection", "features": [f.to_feature() for f in fps]}
    if properties:
        fc["properties"] = properties
    return fc
