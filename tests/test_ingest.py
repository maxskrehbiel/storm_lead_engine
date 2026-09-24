"""The MRMS source: listing parse, file choice, HTTP policy and decode (network mocked)."""

import gzip
import importlib.util
import io
import urllib.error
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from storm_lead_engine import ingest
from storm_lead_engine.config import BBox
from storm_lead_engine.grid import MM_PER_INCH
from storm_lead_engine.ingest import (
    MrmsSource,
    fetch_mrms_day_grids,
    http_get,
    parse_listing,
    pick_convective_day_file,
)

needs_rasterio = pytest.mark.skipif(
    importlib.util.find_spec("rasterio") is None, reason="install the mrms extra"
)

FAST = MrmsSource(backoff_s=0.0, retries=2)


LISTING = """
<a href="MESH_Max_1440min_00.50_20240826-113000.grib2.gz">x</a>
<a href="MESH_Max_1440min_00.50_20240826-120000.grib2.gz">x</a>
<a href="MESH_Max_1440min_00.50_20240826-120000.grib2.gz">dup</a>
<a href="MRMS_Max_1440min_00.50_20210705-000000.grib2.gz">old prefix</a>
"""


def test_parse_listing_handles_both_prefixes_and_duplicates() -> None:
    files = parse_listing(LISTING)
    assert [f[1] for f in files] == ["000000", "113000", "120000"]
    assert files[0][0].startswith("MRMS_")


def test_pick_convective_day_file() -> None:
    src = MrmsSource()
    files = parse_listing(LISTING)
    assert pick_convective_day_file(files, src) == "MESH_Max_1440min_00.50_20240826-120000.grib2.gz"
    near = [("a", "103000"), ("b", "230000")]
    assert pick_convective_day_file(near, src) == "a"
    assert pick_convective_day_file([("late", "230000")], src) is None
    assert pick_convective_day_file([], src) is None


def _geotiff_bytes(mm: np.ndarray, west: float, north: float) -> bytes:
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    h, w = mm.shape
    with MemoryFile() as mf:
        with mf.open(
            driver="GTiff", height=h, width=w, count=1, dtype="float32",
            transform=from_origin(west, north, 0.01, 0.01),
        ) as ds:  # fmt: skip
            ds.write(mm.astype("float32"), 1)
        return bytes(mf.read())


def _fake_archive(monkeypatch: pytest.MonkeyPatch, mm: np.ndarray) -> list[str]:
    calls: list[str] = []
    tif = gzip.compress(_geotiff_bytes(mm, west=-103.0, north=40.0))

    def fake_get(url: str, source: MrmsSource) -> bytes:
        calls.append(url)
        return LISTING.encode() if url.endswith("/") else tif

    monkeypatch.setattr(ingest, "http_get", fake_get)
    return calls


@needs_rasterio
def test_decode_window_converts_and_aligns() -> None:
    mm = np.full((20, 30), -3.0)  # no coverage everywhere...
    mm[5, 10] = 2 * MM_PER_INCH  # ...except one 2-inch cell at lat 39.945, lon -102.895
    raw = _geotiff_bytes(mm, west=-103.0, north=40.0)
    grid = ingest.decode_mesh_window(raw, BBox(-102.95, 39.9, -102.85, 39.98))
    assert grid.shape == (8, 10)
    assert grid.bounds.as_tuple() == pytest.approx((-102.95, 39.9, -102.85, 39.98))
    assert grid.max_in == pytest.approx(2.0)
    assert grid.cell_of(39.945, -102.895) == tuple(
        int(i) for i in np.unravel_index(int(np.argmax(grid.inches)), grid.shape)
    )


@needs_rasterio
def test_mrms_day_grid_reads_next_day_file_and_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mm = np.zeros((20, 30))
    mm[3, 4] = 1.5 * MM_PER_INCH
    calls = _fake_archive(monkeypatch, mm)
    bbox = BBox(-103.0, 39.8, -102.7, 40.0)
    grids, missing = fetch_mrms_day_grids([date(2024, 8, 25)], bbox, tmp_path, FAST)
    assert missing == []
    assert grids["2024-08-25"].max_in == pytest.approx(1.5)
    assert calls[0].endswith("/2024/08/26/mrms/ncep/MESH_Max_1440min/")
    assert calls[1].endswith("20240826-120000.grib2.gz")
    n_calls = len(calls)
    again, _ = fetch_mrms_day_grids([date(2024, 8, 25)], bbox, tmp_path, FAST)
    assert len(calls) == n_calls  # served from the .npz cache
    assert np.array_equal(again["2024-08-25"].inches, grids["2024-08-25"].inches)


def test_unavailable_days_are_reported_not_zeroed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_listing(url: str, source: MrmsSource) -> bytes:
        return b"<html>no files</html>"

    def offline(url: str, source: MrmsSource) -> bytes:
        raise ConnectionError("offline")

    bbox = BBox(-103.0, 39.8, -102.7, 40.0)
    monkeypatch.setattr(ingest, "http_get", no_listing)
    grids, missing = fetch_mrms_day_grids([date(2024, 8, 25)], bbox, tmp_path, FAST)
    assert grids == {} and missing == ["2024-08-25"]
    monkeypatch.setattr(ingest, "http_get", offline)
    grids, missing = fetch_mrms_day_grids([date(2024, 8, 27)], bbox, tmp_path, FAST)
    assert grids == {} and missing == ["2024-08-27"]


def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)
    return sleeps


def test_http_get_retries_transient_errors_without_a_final_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    def boom(*_a: object, **_k: object) -> None:
        attempts.append(1)
        raise OSError("down")

    sleeps = _record_sleeps(monkeypatch)
    monkeypatch.setattr(ingest.urllib.request, "urlopen", boom)
    source = MrmsSource(retries=3, backoff_s=0.5)
    with pytest.raises(ConnectionError, match="after 3 tries"):
        http_get("https://example.invalid/x", source)
    assert len(attempts) == 3
    assert sleeps == [0.5, 1.0]  # between attempts only


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid/x", code, "err", {}, io.BytesIO())  # type: ignore[arg-type]


def test_http_get_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def not_found(*_a: object, **_k: object) -> None:
        attempts.append(1)
        raise _http_error(404)

    sleeps = _record_sleeps(monkeypatch)
    monkeypatch.setattr(ingest.urllib.request, "urlopen", not_found)
    with pytest.raises(ConnectionError, match="HTTP 404"):
        http_get("https://example.invalid/x", MrmsSource(retries=3))
    assert len(attempts) == 1 and sleeps == []


def test_http_get_retries_server_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def unavailable(*_a: object, **_k: object) -> None:
        attempts.append(1)
        raise _http_error(503)

    _record_sleeps(monkeypatch)
    monkeypatch.setattr(ingest.urllib.request, "urlopen", unavailable)
    with pytest.raises(ConnectionError, match="after 2 tries"):
        http_get("https://example.invalid/x", FAST)
    assert len(attempts) == 2


def test_http_get_returns_body(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b"ok"

    monkeypatch.setattr(ingest.urllib.request, "urlopen", lambda *a, **k: Response())
    assert http_get("https://example.invalid/x", FAST) == b"ok"


@pytest.mark.integration
def test_live_mrms_day(tmp_path: Path) -> None:
    """Reads one real convective day from the public archive (network)."""
    bbox = BBox(-98.0, 38.0, -97.0, 39.0)
    grids, missing = fetch_mrms_day_grids([date(2024, 8, 25)], bbox, tmp_path)
    assert missing == []
    grid = grids["2024-08-25"]
    assert grid.shape == (100, 100)
    assert float(grid.inches.min()) >= 0.0
