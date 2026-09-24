"""Shared CSV helpers: optional-value parsing and LF output."""

from pathlib import Path

import pytest

from storm_lead_engine.csv_io import (
    lat_lon,
    opt_bool,
    opt_float,
    opt_int,
    read_records,
    write_records,
)
from storm_lead_engine.errors import InputDataError


def test_optional_parsers() -> None:
    assert opt_int("") is None and opt_int(" 2015 ") == 2015 and opt_int("7.0") == 7
    assert opt_float("") is None and opt_float("1.5") == 1.5
    assert [opt_bool(v) for v in ("", "Yes", "n", "1", "FALSE")] == [None, True, False, True, False]
    with pytest.raises(ValueError, match="not a boolean"):
        opt_bool("maybe")


def test_lat_lon_ranges() -> None:
    assert lat_lon({"lat": "0.5", "lon": "-179.9"}) == (0.5, -179.9)
    with pytest.raises(ValueError, match="out of range"):
        lat_lon({"lat": "0", "lon": "181"})


def test_write_records_uses_lf_and_blanks_none(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "rows.csv"
    n = write_records([{"a": 1, "b": None}, {"a": "x", "b": True}], out, ("a", "b"))
    assert n == 2
    assert out.read_bytes() == b"a,b\n1,\nx,True\n"


def test_unreadable_files_are_input_errors(tmp_path: Path) -> None:
    def parse(row: dict[str, str]) -> str:
        return row["id"]

    with pytest.raises(InputDataError, match="cannot read"):
        read_records(tmp_path / "absent.csv", ("id",), parse, str)
    latin = tmp_path / "latin.csv"
    latin.write_bytes("id\ncaf\xe9\n".encode("latin-1"))
    with pytest.raises(InputDataError, match="not UTF-8"):
        read_records(latin, ("id",), parse, str)
