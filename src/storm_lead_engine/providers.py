"""Service providers the leads can be routed to: an aggregate rating, a location, a radius.

No review text or contact details are modelled. Ratings are 1-5 stars.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .csv_io import lat_lon, opt_float, read_records, write_records

REQUIRED_COLUMNS = ("provider_id", "name", "lat", "lon", "service_radius_km", "review_count")
OPTIONAL_COLUMNS = ("rating",)
MIN_STARS, MAX_STARS = 1.0, 5.0


@dataclass(frozen=True)
class Provider:
    """A service provider with an aggregate star rating.

    Attributes:
        provider_id: Unique identifier.
        name: Display name.
        lat: Base latitude.
        lon: Base longitude.
        service_radius_km: Distance from base the provider will travel.
        review_count: Number of reviews behind ``rating``.
        rating: Mean star rating (1-5); None when there are no reviews.
    """

    provider_id: str
    name: str
    lat: float
    lon: float
    service_radius_km: float
    review_count: int
    rating: float | None = None

    def __post_init__(self) -> None:
        if self.review_count < 0:
            raise ValueError(f"{self.provider_id}: negative review_count")
        if self.review_count > 0 and self.rating is None:
            raise ValueError(f"{self.provider_id}: has reviews but no rating")
        if self.rating is not None and not (MIN_STARS <= self.rating <= MAX_STARS):
            raise ValueError(f"{self.provider_id}: rating {self.rating} outside 1-5")
        if self.service_radius_km <= 0:
            raise ValueError(f"{self.provider_id}: service_radius_km must be > 0")


def parse_provider(row: dict[str, str]) -> Provider:
    """Validate one CSV row and build a :class:`Provider`.

    Args:
        row: Column name to stripped string value.

    Returns:
        The validated provider.

    Raises:
        ValueError: If a value is malformed or out of range.
        KeyError: If a required column is absent from ``row``.
    """
    lat, lon = lat_lon(row)
    return Provider(
        provider_id=row["provider_id"],
        name=row["name"],
        lat=lat,
        lon=lon,
        service_radius_km=float(row["service_radius_km"]),
        review_count=int(float(row["review_count"] or 0)),
        rating=opt_float(row.get("rating", "")),
    )


def load_providers_csv(path: str | Path) -> list[Provider]:
    """Load and validate a provider CSV.

    Args:
        path: CSV file with at least the ``REQUIRED_COLUMNS``.

    Returns:
        Providers in file order.

    Raises:
        InputDataError: If a required column is missing or any row is invalid.
    """
    return read_records(path, REQUIRED_COLUMNS, parse_provider, lambda p: p.provider_id)


def write_providers_csv(providers: Iterable[Provider], path: str | Path) -> int:
    """Write providers to CSV in the canonical column order; returns the row count."""
    return write_records(
        (asdict(p) for p in providers), path, (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)
    )
