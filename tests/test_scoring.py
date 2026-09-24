"""Hail confidence, lead scores, and Bayesian shrinkage of provider ratings."""

import pytest

from storm_lead_engine.config import ConfidenceWeights, LeadWeights, ProviderPriorConfig
from storm_lead_engine.history import HailProfile
from storm_lead_engine.properties import Property
from storm_lead_engine.providers import Provider
from storm_lead_engine.scoring import (
    bayesian_rating,
    empirical_prior,
    hail_confidence,
    haversine_km,
    ramp,
    recommend_provider,
    score_property,
    score_providers,
)

CONF = ConfidenceWeights()
W = LeadWeights()


def _prof(**kw: object) -> HailProfile:
    base: dict[str, object] = {
        "property_id": "P",
        "in_region": True,
        "roof_install_year": 2000,
        "roof_source": "year_built",
    }
    base.update(kw)
    return HailProfile(**base)  # type: ignore[arg-type]


def _prop(**kw: object) -> Property:
    base: dict[str, object] = {
        "property_id": "P",
        "lat": 0.0,
        "lon": 0.0,
        "year_built": 2000,
        "assessed_value": 200_000.0,
        "owner_occupied": True,
    }
    base.update(kw)
    return Property(**base)  # type: ignore[arg-type]


def test_ramp_clamps() -> None:
    assert ramp(-1, 0, 10) == 0.0
    assert ramp(5, 0, 10) == 0.5
    assert ramp(50, 0, 10) == 1.0


def test_hail_confidence_bounds_and_repetition() -> None:
    assert hail_confidence(_prof(), CONF) == 0.0
    full = _prof(n_severe=5, n_damaging=3, n_extreme=2, max_in=3.0)
    assert hail_confidence(full, CONF) == 100.0
    one_big = _prof(n_severe=1, n_damaging=1, n_extreme=1, max_in=2.5)
    three_mid = _prof(n_severe=3, n_damaging=3, n_extreme=0, max_in=1.8)
    assert hail_confidence(three_mid, CONF) > hail_confidence(one_big, CONF)


def test_score_parts_sum_and_statuses() -> None:
    hailed = _prof(n_severe=2, n_damaging=1, max_in=1.7)
    s = score_property(_prop(), hailed, W, CONF, 2025)
    assert s.eligible and s.status == "lead"
    assert s.roof_age == 25 and s.roof_pts == pytest.approx(25.0)
    assert s.occupancy_pts == pytest.approx(15.0)
    assert s.score == pytest.approx(
        s.hail_pts + s.roof_pts + s.value_pts + s.occupancy_pts, abs=0.2
    )

    assert score_property(_prop(), _prof(in_region=False), W, CONF, 2025).status == (
        "outside storm region"
    )
    multi = score_property(_prop(property_type="multi_family"), hailed, W, CONF, 2025)
    assert not multi.eligible and multi.status == "type multi_family not targeted"
    assert score_property(_prop(), _prof(max_in=0.9), W, CONF, 2025).status == "no qualifying hail"
    reroofed = _prof(max_in=0.0, excluded_pre_roof=2)
    assert score_property(_prop(), reroofed, W, CONF, 2025).status == "hail predates current roof"


def test_missing_inputs_score_neutral_not_zero() -> None:
    hailed = _prof(n_severe=1, max_in=1.2)
    known = score_property(_prop(assessed_value=0.0, owner_occupied=False), hailed, W, CONF, 2025)
    unknown = score_property(_prop(assessed_value=None, owner_occupied=None), hailed, W, CONF, 2025)
    assert unknown.value_pts == pytest.approx(100 * W.value * W.neutral_component)
    assert unknown.value_pts > known.value_pts
    assert unknown.occupancy_pts == pytest.approx(100 * W.occupancy * W.neutral_component)
    assert known.occupancy_pts == pytest.approx(100 * W.occupancy * W.absentee_occupancy_credit)


def test_more_hail_never_lowers_the_score() -> None:
    p = _prop()
    light = score_property(p, _prof(n_severe=1, max_in=1.1), W, CONF, 2025)
    heavy = score_property(p, _prof(n_severe=3, n_damaging=2, max_in=2.2), W, CONF, 2025)
    assert heavy.score > light.score


def test_bayesian_rating_canonical_case() -> None:
    thin = bayesian_rating(5.0, 3, prior_mean=4.5, prior_weight=20)
    deep = bayesian_rating(4.8, 400, prior_mean=4.5, prior_weight=20)
    assert thin == pytest.approx(4.565, abs=1e-3)
    assert deep == pytest.approx(4.786, abs=1e-3)
    assert deep > thin
    assert bayesian_rating(None, 0, 4.2, 20) == 4.2
    assert bayesian_rating(5.0, 20, 4.0, 20) == pytest.approx(4.5)  # n == m: halfway


def _prov(pid: str, rating: float | None, n: int, lat: float = 0.0, radius: float = 50.0):
    return Provider(pid, pid, lat, 0.0, radius, n, rating)


def test_empirical_prior_is_review_weighted() -> None:
    provs = [_prov("a", 5.0, 1), _prov("b", 4.0, 9), _prov("c", None, 0)]
    assert empirical_prior(provs, fallback=3.0) == pytest.approx(4.1)
    assert empirical_prior([_prov("c", None, 0)], fallback=3.0) == 3.0


def test_score_providers_ranks_volume_over_thin_perfection() -> None:
    provs = [_prov("thin", 5.0, 3), _prov("deep", 4.8, 400), _prov("none", None, 0)]
    ranked = score_providers(provs, ProviderPriorConfig(prior_mean=4.5))
    assert [r.provider.provider_id for r in ranked] == ["deep", "thin", "none"]
    assert [r.evidence for r in ranked] == ["solid", "thin", "none"]
    assert ranked[2].shrunk_rating == 4.5
    assert 0 <= ranked[-1].quality <= ranked[0].quality <= 100


def test_recommend_requires_coverage_and_evidence() -> None:
    cfg = ProviderPriorConfig(prior_mean=4.5)
    provs = [
        _prov("far", 5.0, 900, lat=2.0, radius=20),  # best, but out of range
        _prov("thin", 5.0, 2),  # in range, too few reviews
        _prov("ok", 4.4, 80),
    ]
    ranked = score_providers(provs, cfg)
    pick = recommend_provider(0.0, 0.0, ranked, cfg)
    assert pick is not None and pick.provider.provider_id == "ok"
    assert recommend_provider(10.0, 10.0, ranked, cfg) is None


def test_haversine_one_degree_of_latitude() -> None:
    assert haversine_km((0.0, 0.0), (1.0, 0.0)) == pytest.approx(111.19, abs=0.05)
