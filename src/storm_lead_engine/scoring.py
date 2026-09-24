"""Scoring: explainable property lead scores and Bayesian-shrunk provider quality."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ConfidenceWeights, LeadWeights, ProviderPriorConfig
from .history import HailProfile
from .properties import Property
from .providers import Provider

EARTH_RADIUS_KM = 6371.0088


def ramp(x: float, lo: float, hi: float) -> float:
    """Linear ramp: 0 at or below ``lo``, 1 at or above ``hi``, straight line between."""
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def hail_confidence(prof: HailProfile, w: ConfidenceWeights) -> float:
    """Score, 0-100, how strongly the hail record implies damage to the current roof.

    Repetition carries the most weight: three separate storms at 1.5 in or more
    are three independent chances to damage the roof. The largest stone still
    counts, so a single extreme storm can score well on its own.

    Args:
        prof: The property's hail ledger.
        w: Weights and saturation points.

    Returns:
        Confidence on a 0-100 scale, rounded to 0.1.
    """
    score = (
        w.severe * ramp(prof.n_severe, 0, w.severe_saturation)
        + w.damaging * ramp(prof.n_damaging, 0, w.damaging_saturation)
        + w.extreme * ramp(prof.n_extreme, 0, w.extreme_saturation)
        + w.max_size * ramp(prof.max_in, w.max_size_floor_in, w.max_size_cap_in)
    )
    return round(100 * score, 1)


@dataclass(frozen=True)
class LeadScore:
    """A property's score, split into the points each component contributed.

    Attributes:
        property_id: Property identifier.
        eligible: True if the property is a lead.
        status: Why the property is or is not a lead.
        score: Total, 0-100.
        hail_confidence: Hail confidence, 0-100.
        hail_pts: Points from hail.
        roof_pts: Points from roof age.
        value_pts: Points from assessed value.
        occupancy_pts: Points from owner occupancy.
        roof_age: Roof age in years at the as-of date.
    """

    property_id: str
    eligible: bool
    status: str
    score: float
    hail_confidence: float
    hail_pts: float
    roof_pts: float
    value_pts: float
    occupancy_pts: float
    roof_age: int


def _status(p: Property, prof: HailProfile, w: LeadWeights) -> str:
    if not prof.in_region:
        return "outside storm region"
    if p.property_type not in w.eligible_types:
        return f"type {p.property_type} not targeted"
    if prof.max_in < w.min_lead_hail_in:
        return "hail predates current roof" if prof.excluded_pre_roof else "no qualifying hail"
    return "lead"


def score_property(
    p: Property,
    prof: HailProfile,
    weights: LeadWeights,
    confidence: ConfidenceWeights,
    as_of_year: int,
) -> LeadScore:
    """Score one property.

    ``score = 100 * (w_hail*confidence/100 + w_roof*age + w_value*value + w_occ*occupancy)``,
    each component in [0, 1]. A missing value or occupancy flag scores
    ``neutral_component`` rather than zero, because a gap in the data is not
    evidence against the property.

    Args:
        p: The property.
        prof: Its hail ledger.
        weights: Lead weights and ranges.
        confidence: Hail-confidence weights.
        as_of_year: Year roof age is measured at.

    Returns:
        The score with its component breakdown.
    """
    w = weights
    roof_age = max(0, as_of_year - prof.roof_install_year)
    conf = hail_confidence(prof, confidence)
    roof_c = ramp(roof_age, w.roof_age_floor_yrs, w.roof_age_cap_yrs)
    if p.assessed_value is None:
        value_c = w.neutral_component
    else:
        value_c = math.sqrt(ramp(p.assessed_value, w.value_floor, w.value_cap))
    if p.owner_occupied is None:
        occ_c = w.neutral_component
    else:
        occ_c = 1.0 if p.owner_occupied else w.absentee_occupancy_credit
    parts = (
        round(w.hail * conf, 1),
        round(100 * w.roof_age * roof_c, 1),
        round(100 * w.value * value_c, 1),
        round(100 * w.occupancy * occ_c, 1),
    )
    status = _status(p, prof, w)
    return LeadScore(
        property_id=p.property_id,
        eligible=status == "lead",
        status=status,
        score=round(sum(parts), 1),
        hail_confidence=conf,
        hail_pts=parts[0],
        roof_pts=parts[1],
        value_pts=parts[2],
        occupancy_pts=parts[3],
        roof_age=roof_age,
    )


def bayesian_rating(rating: float | None, n: int, prior_mean: float, prior_weight: float) -> float:
    """Posterior-mean star rating, ``(n * rating + m * C) / (n + m)``.

    ``C`` is the prior mean (belief about a provider with no reviews) and ``m``
    the prior weight in pseudo-reviews. With ``m = 20`` and ``C = 4.5``, 5.0 stars
    from 3 reviews shrinks to 4.57, while 4.8 from 400 reviews stays at 4.79.

    Args:
        rating: Observed mean rating, or None if unrated.
        n: Number of reviews.
        prior_mean: ``C``.
        prior_weight: ``m``.

    Returns:
        The shrunk rating.
    """
    if n <= 0 or rating is None:
        return prior_mean
    return (n * rating + prior_weight * prior_mean) / (n + prior_weight)


def empirical_prior(providers: list[Provider], fallback: float) -> float:
    """Review-weighted mean rating across providers (empirical Bayes prior).

    Args:
        providers: All providers.
        fallback: Returned when no provider has reviews.

    Returns:
        The prior mean.
    """
    rated = [p for p in providers if p.rating is not None and p.review_count > 0]
    total = sum(p.review_count for p in rated)
    if total == 0:
        return fallback
    return sum((p.rating or 0.0) * p.review_count for p in rated) / total


@dataclass(frozen=True)
class ScoredProvider:
    """A provider with its shrunk rating and rank.

    Attributes:
        rank: 1 = best.
        provider: The provider.
        prior_mean: Prior used for shrinkage.
        shrunk_rating: Posterior-mean rating.
        quality: Shrunk rating mapped onto 0-100.
        evidence: ``none`` (no reviews), ``thin`` (fewer than the prior weight) or ``solid``.
    """

    rank: int
    provider: Provider
    prior_mean: float
    shrunk_rating: float
    quality: float
    evidence: str


def score_providers(providers: list[Provider], cfg: ProviderPriorConfig) -> list[ScoredProvider]:
    """Rank providers by shrunk rating (ties: more reviews first, then id).

    Args:
        providers: Providers to rank.
        cfg: Shrinkage settings.

    Returns:
        Providers, best first.
    """
    prior = (
        cfg.prior_mean
        if cfg.prior_mean is not None
        else empirical_prior(providers, cfg.fallback_prior_mean)
    )
    rows = []
    for p in providers:
        shrunk = bayesian_rating(p.rating, p.review_count, prior, cfg.prior_weight)
        quality = 100 * ramp(shrunk, cfg.rating_floor, cfg.rating_ceil)
        if p.review_count == 0:
            evidence = "none"
        elif p.review_count < cfg.prior_weight:
            evidence = "thin"
        else:
            evidence = "solid"
        rows.append((p, shrunk, quality, evidence))
    rows.sort(key=lambda r: (-r[2], -r[0].review_count, r[0].provider_id))
    return [
        ScoredProvider(i, p, round(prior, 3), round(s, 3), round(q, 1), ev)
        for i, (p, s, q, ev) in enumerate(rows, start=1)
    ]


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two ``(lat, lon)`` points, km."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (
        math.sin((la2 - la1) / 2) ** 2
        + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def recommend_provider(
    lat: float, lon: float, ranked: list[ScoredProvider], cfg: ProviderPriorConfig
) -> ScoredProvider | None:
    """Return the best-ranked provider that serves a location and has enough reviews.

    Args:
        lat: Location latitude.
        lon: Location longitude.
        ranked: Output of :func:`score_providers`.
        cfg: Supplies the minimum review count.

    Returns:
        The recommended provider, or None if nobody qualifies.
    """
    for sp in ranked:
        p = sp.provider
        if p.review_count < cfg.min_reviews_to_recommend:
            continue
        if haversine_km((lat, lon), (p.lat, p.lon)) <= p.service_radius_km:
            return sp
    return None
