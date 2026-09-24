"""Provider schema validation and CSV round-trips (same reader as properties)."""

from pathlib import Path

import numpy as np
import pytest

from storm_lead_engine.errors import InputDataError
from storm_lead_engine.providers import Provider, load_providers_csv, write_providers_csv
from storm_lead_engine.synthetic import generate_providers

HEADER = "provider_id,name,lat,lon,service_radius_km,review_count,rating\n"


def test_round_trip(tmp_path: Path) -> None:
    provs = generate_providers((0.0, 0.0), np.random.default_rng(2))
    out = tmp_path / "prov.csv"
    assert write_providers_csv(provs, out) == len(provs)
    assert load_providers_csv(out) == provs


@pytest.mark.parametrize(
    ("rating", "count", "radius", "match"),
    [
        (4.5, -1, 10.0, "negative"),
        (None, 5, 10.0, "no rating"),
        (6.0, 5, 10.0, "outside 1-5"),
        (4.5, 5, 0.0, "service_radius_km"),
    ],
)
def test_validation(rating: float | None, count: int, radius: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Provider("P", "P", 0.0, 0.0, radius, count, rating)


def test_missing_column(tmp_path: Path) -> None:
    missing = tmp_path / "a.csv"
    missing.write_text("provider_id,name\nP,P\n", encoding="utf-8")
    with pytest.raises(InputDataError, match="missing required"):
        load_providers_csv(missing)


def test_every_bad_row_is_reported(tmp_path: Path) -> None:
    rows = [
        "P1,One,0.1,0.1,50,3,4.5",  # valid
        "P2,Two,0.1,0.1,50,3,9",  # rating out of range
        "P3,Three,120,0.1,50,3,4.5",  # latitude out of range
        "P1,Dup,0.1,0.1,50,3,4.5",  # duplicate id
        ",Blank,0.1,0.1,50,3,4.5",  # empty id
        "P4,Four,0.1,0.1,abc,3,4.5",  # radius not a number
    ]
    path = tmp_path / "b.csv"
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(InputDataError) as err:
        load_providers_csv(path)
    msg = str(err.value)
    assert "5 invalid row(s)" in msg
    for needle in ("line 3", "outside 1-5", "out of range", "duplicate id", "empty id", "abc"):
        assert needle in msg
