"""The property layer: a small validated schema the user supplies as CSV.

Only the columns below are read. Any other column in the input (names, contact
details, notes) is dropped at load time and can never reach an output file.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .csv_io import lat_lon, opt_bool, opt_float, opt_int, read_records, write_records

REQUIRED_COLUMNS = ("property_id", "lat", "lon", "year_built")
OPTIONAL_COLUMNS = ("address", "roof_year", "assessed_value", "owner_occupied", "property_type")
PROPERTY_TYPES = ("single_family", "multi_family", "commercial")
EARLIEST_YEAR, LATEST_YEAR = 1700, 2100


@dataclass(frozen=True)
class Property:
    """One property in the user-supplied layer.

    Attributes:
        property_id: Unique identifier.
        lat: Latitude of the structure.
        lon: Longitude of the structure.
        year_built: Construction year.
        address: Free-text street address (optional; passed through to outputs).
        roof_year: Year of the last known roof replacement, e.g. from permit records.
        assessed_value: Assessed value in dollars.
        owner_occupied: True if the owner lives there, False if absentee, None if unknown.
        property_type: One of ``PROPERTY_TYPES``.
    """

    property_id: str
    lat: float
    lon: float
    year_built: int
    address: str = ""
    roof_year: int | None = None
    assessed_value: float | None = None
    owner_occupied: bool | None = None
    property_type: str = "single_family"

    def roof_install_year(self) -> tuple[int, str]:
        """Return ``(year the current roof went on, source of that year)``."""
        if self.roof_year is not None:
            return self.roof_year, "roof_record"
        return self.year_built, "year_built"


def parse_property(row: dict[str, str]) -> Property:
    """Validate one CSV row and build a :class:`Property`.

    Args:
        row: Column name to stripped string value.

    Returns:
        The validated property.

    Raises:
        ValueError: If a value is malformed or implausible.
        KeyError: If a required column is absent from ``row``.
    """
    lat, lon = lat_lon(row)
    year_built = int(float(row["year_built"]))
    if not (EARLIEST_YEAR <= year_built <= LATEST_YEAR):
        raise ValueError(f"implausible year_built: {year_built}")
    roof_year = opt_int(row.get("roof_year", ""))
    if roof_year is not None and roof_year < year_built:
        raise ValueError(f"roof_year {roof_year} is before year_built {year_built}")
    ptype = row.get("property_type", "") or "single_family"
    if ptype not in PROPERTY_TYPES:
        raise ValueError(f"unknown property_type {ptype!r} (expected one of {PROPERTY_TYPES})")
    return Property(
        property_id=row["property_id"],
        lat=lat,
        lon=lon,
        year_built=year_built,
        address=row.get("address", ""),
        roof_year=roof_year,
        assessed_value=opt_float(row.get("assessed_value", "")),
        owner_occupied=opt_bool(row.get("owner_occupied", "")),
        property_type=ptype,
    )


def load_properties_csv(path: str | Path) -> list[Property]:
    """Load and validate a property CSV.

    Args:
        path: CSV file with at least the ``REQUIRED_COLUMNS``.

    Returns:
        Properties in file order.

    Raises:
        InputDataError: If a required column is missing or any row is invalid.
    """
    return read_records(path, REQUIRED_COLUMNS, parse_property, lambda p: p.property_id)


def write_properties_csv(props: Iterable[Property], path: str | Path) -> int:
    """Write properties to CSV in the canonical column order; returns the row count."""
    return write_records((asdict(p) for p in props), path, (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS))
