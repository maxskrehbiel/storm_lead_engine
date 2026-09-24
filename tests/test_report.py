"""Output files: CSV schemas, GeoJSON, summary and the HTML map."""

import csv
import dataclasses
import json
from pathlib import Path

import folium
import numpy as np
import pytest

from storm_lead_engine.report import (
    HAIL_RAMP,
    LEAD_COLUMNS,
    PROVIDER_COLUMNS,
    SCORE_BINS,
    colorize_hail,
    csv_safe,
    png_bytes,
    render_map,
    score_color,
    stable_ids,
    to_web_mercator,
    write_leads_csv,
    write_outputs,
)

ONE_MB = 1_000_000


def test_write_outputs(demo_result, tmp_path: Path) -> None:
    paths = write_outputs(demo_result, tmp_path, prefix="synthetic_", top=25)
    assert {p.name for p in paths.values()} == {
        "synthetic_leads_top25.csv",
        "synthetic_providers_ranked.csv",
        "synthetic_swath_footprints.geojson",
        "synthetic_run_summary.json",
        "synthetic_storm_map.html",
    }
    assert all(p.stat().st_size < ONE_MB for p in paths.values())

    with paths["leads"].open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == LEAD_COLUMNS
    assert len(rows) == 25
    assert [int(r["rank"]) for r in rows] == list(range(1, 26))
    assert float(rows[0]["score"]) >= float(rows[-1]["score"])

    with paths["providers"].open(encoding="utf-8") as f:
        prov = list(csv.DictReader(f))
    assert tuple(prov[0]) == PROVIDER_COLUMNS
    assert len(prov) == len(demo_result.providers)

    fc = json.loads(paths["footprints"].read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection"
    assert fc["properties"]["synthetic_data"] is True
    labels = {f["properties"]["label"] for f in fc["features"]}
    assert "max-ever" in labels and "EV-20220708" in labels

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["synthetic_data"] is True

    page = paths["map"].read_text(encoding="utf-8")
    assert "SYNTHETIC DEMO DATA" in page
    assert "<title>Storm lead map (synthetic demo)</title>" in page
    assert "Bayesian-shrunk" in page
    assert "tile.openstreetmap" not in page  # no basemap, no tile requests

    for path in paths.values():
        data = path.read_bytes()
        assert b"\r\n" not in data and data.endswith(b"\n"), path.name


def test_outputs_are_byte_stable(demo_result, tmp_path: Path) -> None:
    first = write_outputs(demo_result, tmp_path / "a", prefix="synthetic_", top=10)
    second = write_outputs(demo_result, tmp_path / "b", prefix="synthetic_", top=10)
    for kind, path in first.items():
        assert path.read_bytes() == second[kind].read_bytes(), kind


def test_stable_ids_renumbers_in_order_of_appearance() -> None:
    a, b = "0f" * 16, "a1" * 16
    page = f"map_{a} geo_json_{b} L.map('map_{a}') not_an_id_{'ab' * 8}"
    assert stable_ids(page) == f"map_0001 geo_json_0002 L.map('map_0001') not_an_id_{'ab' * 8}"


def test_real_data_maps_carry_no_synthetic_banner(demo_result, tmp_path: Path) -> None:
    real = dataclasses.replace(demo_result, synthetic=False, providers=(), leads=())
    paths = write_outputs(real, tmp_path)
    page = paths["map"].read_text(encoding="utf-8")
    assert "SYNTHETIC DEMO DATA" not in page
    assert "<title>Storm lead map</title>" in page
    assert paths["leads"].name == "leads.csv"
    assert paths["leads"].read_text(encoding="utf-8").count("\n") == 1  # header only


def test_colorize_hail_is_clear_below_the_ramp(make_grid) -> None:
    img = colorize_hail(make_grid([[0.0, 0.1, HAIL_RAMP[0][0], 3.0]]))
    assert img.shape == (1, 4, 4) and img.dtype == np.uint8
    assert img[0, 0, 3] == 0 and img[0, 1, 3] == 0
    assert img[0, 2, 3] == HAIL_RAMP[0][2]
    assert img[0, 3, 3] == HAIL_RAMP[-1][2]


def test_score_color_bins() -> None:
    assert score_color(0.0) == SCORE_BINS[0][1]
    assert score_color(SCORE_BINS[2][0]) == SCORE_BINS[2][1]
    assert score_color(100.0) == SCORE_BINS[-1][1]


def test_map_requires_a_figure_root(
    demo_result, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(folium.Map, "get_root", lambda self: folium.Element(""))
    with pytest.raises(TypeError, match="unexpected folium root"):
        render_map(demo_result, tmp_path / "m.html")


def test_user_text_is_escaped_in_the_map_and_neutralised_in_csv(
    demo_result, tmp_path: Path
) -> None:
    hostile = '<img src=x onerror="alert(1)">'
    first = demo_result.leads[0]
    evil_prop = dataclasses.replace(first.property, address=hostile)
    evil_lead = dataclasses.replace(first, property=evil_prop)
    result = dataclasses.replace(demo_result, leads=(evil_lead, *demo_result.leads[1:]))
    render_map(result, tmp_path / "m.html")
    page = (tmp_path / "m.html").read_text(encoding="utf-8")
    assert "<img src=x" not in page
    assert "&lt;img src=x" in page

    formula = dataclasses.replace(first.property, address="=HYPERLINK(1)")
    rows_path = tmp_path / "leads.csv"
    write_leads_csv(
        dataclasses.replace(demo_result, leads=(dataclasses.replace(first, property=formula),)),
        rows_path,
    )
    with rows_path.open(encoding="utf-8") as f:
        (row,) = list(csv.DictReader(f))
    assert row["address"] == "'=HYPERLINK(1)"


def test_csv_safe() -> None:
    assert csv_safe("12 Main St") == "12 Main St"
    assert csv_safe("-2+3") == "'-2+3"
    assert csv_safe("@SUM(A1)") == "'@SUM(A1)"


def test_web_mercator_resampling_copies_whole_rows() -> None:
    rows = np.arange(100, dtype=np.uint8).reshape(100, 1, 1).repeat(4, axis=2)
    near_equator = to_web_mercator(rows, -0.5, 0.5)
    assert near_equator[:, 0, 0].tolist() == list(range(100))  # no stretch at the equator
    high = to_web_mercator(rows, 30.0, 60.0)[:, 0, 0].tolist()
    assert high == sorted(high)  # rows stay in order
    assert set(high) <= set(range(100))  # every row is a copy of an input row
    assert high[0] == 0 and high[-1] == 99
    # Mercator stretches the north, so northern input rows are repeated more often
    assert high.count(0) >= high.count(99)


def test_png_bytes_are_a_valid_stored_png() -> None:
    import struct
    import zlib

    img = np.arange(3 * 40000 * 4, dtype=np.uint64).astype(np.uint8).reshape(3, 40000, 4)
    data = png_bytes(img)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (40000, 3)
    idat_len = struct.unpack(">I", data[33:37])[0]
    assert data[37:41] == b"IDAT"
    raw = zlib.decompress(data[41 : 41 + idat_len])  # multiple stored blocks decode cleanly
    rows = [raw[i * (1 + 40000 * 4) : (i + 1) * (1 + 40000 * 4)] for i in range(3)]
    assert all(r[0] == 0 for r in rows)
    assert b"".join(r[1:] for r in rows) == img.tobytes()
    assert data == png_bytes(img)
