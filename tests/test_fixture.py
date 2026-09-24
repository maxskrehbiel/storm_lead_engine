"""The bundled synthetic storm season: loading, validation and known ground truth."""

import json
from datetime import date
from pathlib import Path

import pytest

from storm_lead_engine.errors import InputDataError
from storm_lead_engine.fixture import (
    bundled_fixture_text,
    km_to_latlon,
    load_season,
    season_day_grids,
)


def _raw() -> dict[str, object]:
    data: dict[str, object] = json.loads(bundled_fixture_text())
    return data


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "season.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_bundled_season(season) -> None:
    assert season.origin == (0.0, 0.0)
    assert season.as_of == date(2025, 12, 31)
    assert len(season.storms) == 9
    w, s, e, n = season.region.as_tuple()
    assert (round(n - s, 2), round(e - w, 2)) == (0.92, 1.08)


def test_every_storm_day_peaks_at_its_specified_size(season, day_grids) -> None:
    peaks = {s.date: s.peak_in for s in season.storms}
    assert set(day_grids) == set(peaks)
    for day, grid in day_grids.items():
        assert grid.max_in == pytest.approx(peaks[day], abs=1e-4)


def test_km_offsets_convert_to_degrees(season) -> None:
    s03 = next(s for s in season.storms if s.storm_id == "S03")
    assert s03.path[1] == pytest.approx((0.0, 0.0))
    assert s03.core_at == pytest.approx((0.0, 0.0))
    s01 = next(s for s in season.storms if s.storm_id == "S01")
    assert s01.path[1][0] == pytest.approx(-1.5 / 110.574, abs=1e-6)
    assert km_to_latlon((0.0, 0.0), 111.32, 0.0) == pytest.approx((0.0, 1.0))


def test_fixture_must_declare_synthetic(tmp_path: Path) -> None:
    raw = _raw()
    raw["synthetic"] = False
    with pytest.raises(InputDataError, match="synthetic"):
        load_season(_write(tmp_path, raw))
    with pytest.raises(InputDataError, match="synthetic"):
        load_season(_write(tmp_path, [1, 2, 3]))


def test_missing_fixture_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputDataError, match="cannot read"):
        load_season(tmp_path / "absent.json")


def test_malformed_fixtures_are_input_errors(tmp_path: Path) -> None:
    raw = _raw()
    del raw["origin"]
    with pytest.raises(InputDataError, match="malformed"):
        load_season(_write(tmp_path, raw))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputDataError, match="not valid JSON"):
        load_season(bad)


def test_same_day_storms_are_pixel_maxed(tmp_path: Path) -> None:
    raw = _raw()
    storms = raw["storms"]
    assert isinstance(storms, list)
    first, third = dict(storms[0]), dict(storms[2])
    third["date"] = first["date"]
    raw["storms"] = [first, third]
    grids = season_day_grids(load_season(_write(tmp_path, raw)))
    assert list(grids) == ["2021-05-18"]
    assert grids["2021-05-18"].max_in == pytest.approx(2.75, abs=1e-4)
