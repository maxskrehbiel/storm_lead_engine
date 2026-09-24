"""Typed, frozen configuration objects: every tunable number in the engine lives here."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BBox:
    """Geographic bounding box in decimal degrees (WGS84).

    Attributes:
        west: Western longitude.
        south: Southern latitude.
        east: Eastern longitude.
        north: Northern latitude.
    """

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not (-180 <= self.west < self.east <= 180):
            raise ValueError(f"invalid longitudes: west={self.west} east={self.east}")
        if not (-90 <= self.south < self.north <= 90):
            raise ValueError(f"invalid latitudes: south={self.south} north={self.north}")

    @classmethod
    def parse(cls, text: str) -> BBox:
        """Parse a ``"west,south,east,north"`` string (GeoJSON order).

        Args:
            text: Four comma-separated numbers.

        Returns:
            The parsed box.

        Raises:
            ValueError: If there are not exactly four numbers or they are out of range.
        """
        parts = [float(p) for p in text.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must be 'west,south,east,north'")
        return cls(*parts)

    def contains(self, lat: float, lon: float) -> bool:
        """Return True if the point lies inside the box (edges included)."""
        return self.south <= lat <= self.north and self.west <= lon <= self.east

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(west, south, east, north)``."""
        return (self.west, self.south, self.east, self.north)

    @property
    def center(self) -> tuple[float, float]:
        """Centre point as ``(lat, lon)``."""
        return ((self.south + self.north) / 2, (self.west + self.east) / 2)


@dataclass(frozen=True)
class HailTiers:
    """Hail-size tiers in inches; each yields a swath footprint and an event count.

    ``severe`` (1.00 in, quarter-size) is the U.S. National Weather Service
    severe-hail criterion and a common rule-of-thumb onset of damage to aged
    asphalt shingles. ``damaging`` and ``extreme`` mark increasingly certain damage.

    Attributes:
        severe: Lowest tier, inches.
        damaging: Middle tier, inches.
        extreme: Highest tier, inches.
    """

    severe: float = 1.00
    damaging: float = 1.50
    extreme: float = 2.00

    def __post_init__(self) -> None:
        if not (0 < self.severe < self.damaging < self.extreme):
            raise ValueError("tiers must be strictly increasing and positive")

    def as_tuple(self) -> tuple[float, float, float]:
        """Return ``(severe, damaging, extreme)``."""
        return (self.severe, self.damaging, self.extreme)


@dataclass(frozen=True)
class HistoryConfig:
    """How storm days become events and how properties read the grid.

    Attributes:
        storm_day_min_in: A day is a storm day only if its grid maximum reaches this.
        event_gap_days: Storm days at most this far apart merge into one event.
        neighborhood: Geocode tolerance in cells; 1 means a 3x3 window (~1 km at MRMS).
    """

    storm_day_min_in: float = 0.75
    event_gap_days: int = 1
    neighborhood: int = 1


@dataclass(frozen=True)
class ConfidenceWeights:
    """Weights and saturation points for the 0-100 hail-confidence score.

    Each term is ``weight * clamp(value / saturation, 0, 1)``; the weights sum to 1.

    Attributes:
        severe: Weight on the count of events at or above the severe tier.
        damaging: Weight on the count of events at or above the damaging tier.
        extreme: Weight on the count of events at or above the extreme tier.
        max_size: Weight on the largest stone on the current roof.
        severe_saturation: Severe-event count that earns the full severe weight.
        damaging_saturation: Damaging-event count that earns the full damaging weight.
        extreme_saturation: Extreme-event count that earns the full extreme weight.
        max_size_floor_in: Stone size that earns nothing on the size term.
        max_size_cap_in: Stone size that earns the full size weight.
    """

    severe: float = 0.20
    damaging: float = 0.35
    extreme: float = 0.15
    max_size: float = 0.30
    severe_saturation: int = 5
    damaging_saturation: int = 3
    extreme_saturation: int = 2
    max_size_floor_in: float = 1.0
    max_size_cap_in: float = 3.0

    def __post_init__(self) -> None:
        total = self.severe + self.damaging + self.extreme + self.max_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"confidence weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class LeadWeights:
    """Property lead-score weights (summing to 1) and normalisation ranges.

    Attributes:
        hail: Weight on hail confidence.
        roof_age: Weight on roof age.
        value: Weight on assessed value.
        occupancy: Weight on owner occupancy.
        roof_age_floor_yrs: Roof age that scores 0 on the age term.
        roof_age_cap_yrs: Roof age that scores 1 on the age term.
        value_floor: Assessed value that scores 0 (square-root curve above it).
        value_cap: Assessed value that scores 1.
        absentee_occupancy_credit: Occupancy term for an absentee owner (owner-occupied = 1).
        neutral_component: Value and occupancy score when that input is missing.
        min_lead_hail_in: Hail on the current roof must reach this for a lead.
        eligible_types: Property types that can become leads.
    """

    hail: float = 0.45
    roof_age: float = 0.25
    value: float = 0.15
    occupancy: float = 0.15
    roof_age_floor_yrs: float = 5.0
    roof_age_cap_yrs: float = 25.0
    value_floor: float = 75_000.0
    value_cap: float = 750_000.0
    absentee_occupancy_credit: float = 0.5
    neutral_component: float = 0.5
    min_lead_hail_in: float = 1.00
    eligible_types: tuple[str, ...] = ("single_family",)

    def __post_init__(self) -> None:
        total = self.hail + self.roof_age + self.value + self.occupancy
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"lead weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class ProviderPriorConfig:
    """Bayesian shrinkage settings for service-provider ratings.

    Attributes:
        prior_mean: Prior mean rating; None means empirical Bayes (review-weighted mean).
        prior_weight: Pseudo-review count; a provider with this many reviews sits halfway
            between its own mean and the prior.
        fallback_prior_mean: Prior used when no provider has any reviews.
        rating_floor: Shrunk rating that maps to quality 0.
        rating_ceil: Shrunk rating that maps to quality 100.
        min_reviews_to_recommend: Fewer reviews than this and a provider is never recommended.
    """

    prior_mean: float | None = None
    prior_weight: float = 20.0
    fallback_prior_mean: float = 4.0
    rating_floor: float = 3.0
    rating_ceil: float = 5.0
    min_reviews_to_recommend: int = 5


@dataclass(frozen=True)
class EngineConfig:
    """Top-level bundle of every configuration group.

    Attributes:
        tiers: Hail-size tiers.
        history: Event grouping and sampling rules.
        confidence: Hail-confidence weights.
        weights: Lead-score weights.
        providers: Provider shrinkage settings.
    """

    tiers: HailTiers = field(default_factory=HailTiers)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    confidence: ConfidenceWeights = field(default_factory=ConfidenceWeights)
    weights: LeadWeights = field(default_factory=LeadWeights)
    providers: ProviderPriorConfig = field(default_factory=ProviderPriorConfig)
