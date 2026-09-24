"""Property CSV schema: validation, minimisation and round-trips."""

from pathlib import Path

import numpy as np
import pytest

from storm_lead_engine.errors import InputDataError
from storm_lead_engine.properties import Property, load_properties_csv, write_properties_csv
from storm_lead_engine.synthetic import generate_properties

HEADER = "property_id,lat,lon,year_built,roof_year,owner_occupied,property_type\n"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_round_trip(tmp_path: Path) -> None:
    props = generate_properties(50, (0.0, 0.0), 2025, np.random.default_rng(1))
    out = tmp_path / "p.csv"
    assert write_properties_csv(props, out) == 50
    assert load_properties_csv(out) == props
    assert b"\r\n" not in out.read_bytes()


def test_unknown_columns_are_dropped(tmp_path: Path) -> None:
    f = _write(
        tmp_path / "p.csv",
        "property_id,lat,lon,year_built,owner_name,notes\nX1,0.5,0.5,1975,Somebody,call\n",
    )
    (p,) = load_properties_csv(f)
    assert p == Property("X1", 0.5, 0.5, 1975)
    assert not hasattr(p, "owner_name")
    assert p.roof_install_year() == (1975, "year_built")


def test_optional_fields_parse(tmp_path: Path) -> None:
    f = _write(
        tmp_path / "p.csv",
        "property_id,lat,lon,year_built,roof_year,assessed_value,owner_occupied,property_type\n"
        "A,0.5,0.5,1980,2015,210000,yes,single_family\n"
        "B,0.5,0.5,1980,,,N,commercial\n"
        "C,0.5,0.5,1980,,,,\n",
    )
    a, b, c = load_properties_csv(f)
    assert (a.roof_year, a.assessed_value, a.owner_occupied) == (2015, 210000.0, True)
    assert a.roof_install_year() == (2015, "roof_record")
    assert (b.owner_occupied, b.property_type) == (False, "commercial")
    assert (c.owner_occupied, c.property_type) == (None, "single_family")


def test_missing_required_column(tmp_path: Path) -> None:
    f = _write(tmp_path / "p.csv", "property_id,lat,lon\nA,0.5,0.5\n")
    with pytest.raises(InputDataError, match="year_built"):
        load_properties_csv(f)


def test_every_bad_row_is_reported(tmp_path: Path) -> None:
    rows = [
        "A,0.5,0.5,1980,,",  # valid
        "B,0.5,0.5,1980,,maybe",  # bad boolean
        "C,95,0.5,1980,,",  # latitude out of range
        "D,0.5,0.5,1980,1970,",  # roof before house
        "E,0.5,0.5,1500,,",  # implausible year
        "A,0.5,0.5,1980,,",  # duplicate id
        ",0.5,0.5,1980,,",  # empty id
        "F,0.5,0.5,1980,,,castle",  # unknown type
    ]
    f = _write(tmp_path / "p.csv", HEADER + "\n".join(rows) + "\n")
    with pytest.raises(InputDataError) as err:
        load_properties_csv(f)
    msg = str(err.value)
    assert "7 invalid row(s)" in msg
    needles = (
        "line 3",
        "not a boolean",
        "out of range",
        "before year_built",
        "implausible",
        "duplicate id",
        "empty id",
        "unknown property_type",
    )
    for needle in needles:
        assert needle in msg


def test_error_list_is_truncated(tmp_path: Path) -> None:
    body = "property_id,lat,lon,year_built\n" + "".join(f"X{i},99,0,1980\n" for i in range(12))
    with pytest.raises(InputDataError, match=r"\.\.\. and 2 more"):
        load_properties_csv(_write(tmp_path / "p.csv", body))
