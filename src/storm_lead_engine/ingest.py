"""The public storm source: NOAA MRMS MESH read from the Iowa State University archive mirror.

Needs network access and the ``mrms`` extra (rasterio). Returns the same
``{"YYYY-MM-DD": HailGrid}`` mapping as the offline fixture.
"""

from __future__ import annotations

import gzip
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .config import BBox
from .errors import MissingDependencyError
from .grid import HailGrid, snap_bbox

logger = logging.getLogger(__name__)

_LISTING = re.compile(r'href="((?:MRMS|MESH)_Max_1440min_00\.50_(\d{8})-(\d{6})\.grib2\.gz)"')
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60
HTTP_CLIENT_ERRORS = range(400, 500)


@dataclass(frozen=True)
class MrmsSource:
    """Where and how to read MRMS ``MESH_Max_1440min`` (24-hour maximum hail size).

    A storm day ``D`` is read from the file stamped 12:00 UTC on ``D+1``. Its
    window, 12:00 UTC on ``D`` to 12:00 UTC on ``D+1``, is the convective day the
    NOAA Storm Prediction Center uses for storm reports. An evening storm that
    crosses 00 UTC stays inside one day, and consecutive days never overlap.

    Attributes:
        base_url: Archive root.
        product: Product directory name.
        target_stamp: Preferred file time, ``HHMMSS`` UTC.
        max_offset_hours: Farthest a substitute file may be from ``target_stamp``.
        timeout_s: HTTP timeout per request.
        retries: Attempts per request (client errors are never retried).
        backoff_s: Base delay between attempts (grows linearly).
        user_agent: HTTP User-Agent header.
    """

    base_url: str = "https://mtarchive.geol.iastate.edu"
    product: str = "MESH_Max_1440min"
    target_stamp: str = "120000"
    max_offset_hours: float = 2.0
    timeout_s: float = 120.0
    retries: int = 3
    backoff_s: float = 1.5
    user_agent: str = "storm_lead_engine/0.1 (+https://github.com/maxskrehbiel/storm_lead_engine)"

    def day_dir_url(self, utc_day: date) -> str:
        """Directory URL holding one UTC day's files."""
        return f"{self.base_url}/{utc_day:%Y/%m/%d}/mrms/ncep/{self.product}/"


def parse_listing(html: str) -> list[tuple[str, str]]:
    """Extract ``(filename, HHMMSS)`` pairs from an archive directory page.

    File names switched from an ``MRMS_`` to a ``MESH_`` prefix around 2023; both match.

    Args:
        html: Directory listing HTML.

    Returns:
        Unique files sorted by time stamp.
    """
    seen: dict[str, str] = {}
    for m in _LISTING.finditer(html):
        seen.setdefault(m.group(1), m.group(3))
    return sorted(seen.items(), key=lambda kv: kv[1])


def _seconds(hhmmss: str) -> int:
    hours, minutes, seconds = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6])
    return hours * SECONDS_PER_HOUR + minutes * SECONDS_PER_MINUTE + seconds


def pick_convective_day_file(files: list[tuple[str, str]], source: MrmsSource) -> str | None:
    """Choose the file stamped closest to the target time, within the allowed offset.

    Returns None rather than substituting a file whose window would miss most of the day.

    Args:
        files: ``(filename, HHMMSS)`` pairs for one UTC day.
        source: Target time and tolerance.

    Returns:
        The chosen file name, or None.
    """
    if not files:
        return None
    target = _seconds(source.target_stamp)
    name, stamp = min(files, key=lambda f: (abs(_seconds(f[1]) - target), f[1]))
    if abs(_seconds(stamp) - target) > source.max_offset_hours * SECONDS_PER_HOUR:
        return None
    return name


def http_get(url: str, source: MrmsSource) -> bytes:
    """GET a URL, retrying transient failures with linear backoff.

    A 4xx response (for example a day missing from the archive) fails at once;
    there is no sleep after the final attempt.

    Args:
        url: Address to fetch.
        source: Timeout, retry and header settings.

    Returns:
        Response body.

    Raises:
        ConnectionError: On a 4xx response, or when every attempt fails.
    """
    last: Exception | None = None
    for attempt in range(1, source.retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": source.user_agent})
            with urllib.request.urlopen(req, timeout=source.timeout_s) as r:
                body: bytes = r.read()
                return body
        except urllib.error.HTTPError as exc:
            if exc.code in HTTP_CLIENT_ERRORS:
                raise ConnectionError(f"GET {url}: HTTP {exc.code}") from exc
            last = exc
        except OSError as exc:
            last = exc
        if attempt < source.retries:
            time.sleep(source.backoff_s * attempt)
    raise ConnectionError(f"GET {url} failed after {source.retries} tries: {last}")


def decode_mesh_window(raw: bytes, bbox: BBox) -> HailGrid:
    """Decode an uncompressed MRMS GRIB2 file and cut out a lattice-aligned window.

    Any single-band raster GDAL can open works, which lets tests use an
    in-memory GeoTIFF instead of a download.

    Args:
        raw: File bytes.
        bbox: Region to cut; snapped outward to the lattice.

    Returns:
        The window as a grid in inches.

    Raises:
        MissingDependencyError: If rasterio is not installed.
    """
    try:
        from rasterio.io import MemoryFile
        from rasterio.windows import Window
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MissingDependencyError(
            'reading real MRMS GRIB2 needs rasterio: pip install -e ".[mrms]"'
        ) from exc

    snapped = snap_bbox(bbox)
    with MemoryFile(raw) as mf, mf.open() as ds:
        t = ds.transform
        col_off = round((snapped.west - t.c) / t.a)
        row_off = round((t.f - snapped.north) / -t.e)
        width = round((snapped.east - snapped.west) / t.a)
        height = round((snapped.north - snapped.south) / -t.e)
        window = Window(col_off, row_off, width, height)
        arr = ds.read(1, window=window, boundless=True, fill_value=-999)
        west = t.c + col_off * t.a
        north = t.f + row_off * t.e
    bounds = BBox(
        round(west, 6),
        round(north + height * t.e, 6),
        round(west + width * t.a, 6),
        round(north, 6),
    )
    return HailGrid.from_mm(arr, bounds)


def mrms_day_grid(day: date, bbox: BBox, cache_dir: Path, source: MrmsSource) -> HailGrid | None:
    """Read one convective storm day of MESH over a region, caching the window as ``.npz``.

    Args:
        day: Storm day.
        bbox: Region to cut.
        cache_dir: Directory for cached windows (never committed).
        source: Archive settings.

    Returns:
        The day's grid, or None if the archive has no suitable file.
    """
    snapped = snap_bbox(bbox)
    w, s, e, n = snapped.as_tuple()
    cache = cache_dir / f"mesh_{day:%Y%m%d}_{w:.2f}_{s:.2f}_{e:.2f}_{n:.2f}.npz"
    if cache.exists():
        logger.debug("MRMS %s served from cache %s", day, cache.name)
        with np.load(cache) as z:
            return HailGrid(z["inches"], BBox(*(float(v) for v in z["bounds"])))

    next_day = day + timedelta(days=1)
    logger.info("reading MRMS MESH for storm day %s", day)
    listing = http_get(source.day_dir_url(next_day), source).decode("utf-8", "ignore")
    name = pick_convective_day_file(parse_listing(listing), source)
    if name is None:
        return None
    raw = gzip.decompress(http_get(source.day_dir_url(next_day) + name, source))
    grid = decode_mesh_window(raw, snapped)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, inches=grid.inches, bounds=np.array(grid.bounds.as_tuple()))
    return grid


def fetch_mrms_day_grids(
    days: list[date], bbox: BBox, cache_dir: Path, source: MrmsSource | None = None
) -> tuple[dict[str, HailGrid], list[str]]:
    """Read several storm days, keeping days that fail separate from quiet days.

    A missing day is reported, never read as "no hail". A silent outage must
    not look like a calm day.

    Args:
        days: Storm days to read.
        bbox: Region to cut.
        cache_dir: Directory for cached windows.
        source: Archive settings; defaults to :class:`MrmsSource`.

    Returns:
        ``(grids, missing_days)``.
    """
    src = source or MrmsSource()
    grids: dict[str, HailGrid] = {}
    missing: list[str] = []
    for d in sorted(set(days)):
        try:
            grid = mrms_day_grid(d, bbox, cache_dir, src)
        # Parsing a downloaded file: network, gzip and GDAL errors are OSErrors, and a
        # malformed raster surfaces as ValueError.
        except (OSError, ValueError) as exc:
            logger.warning("MRMS %s unavailable: %s", d, exc)
            grid = None
        if grid is None:
            missing.append(d.isoformat())
        else:
            grids[d.isoformat()] = grid
    return grids, missing
