"""End-to-end pipeline behaviour on the synthetic season."""

import json

import numpy as np
import pytest

from storm_lead_engine.config import EngineConfig, HistoryConfig
from storm_lead_engine.errors import NoStormDataError
from storm_lead_engine.pipeline import run_pipeline
from storm_lead_engine.synthetic import generate_properties


def test_leads_are_ranked_eligible_and_explained(demo_result) -> None:
    leads = demo_result.leads
    assert leads, "the synthetic season should produce leads"
    assert [ld.rank for ld in leads] == list(range(1, len(leads) + 1))
    scores = [ld.score.score for ld in leads]
    assert scores == sorted(scores, reverse=True)
    for ld in leads:
        assert ld.score.eligible
        assert ld.property.property_type == "single_family"
        assert ld.profile.max_in >= 1.0
        assert ld.outreach_channel == ("door_knock" if ld.property.owner_occupied else "mail")


def test_every_property_gets_a_status(demo_result) -> None:
    assert set(demo_result.scores) == {p.property_id for p in demo_result.properties}
    statuses = {s.status for s in demo_result.scores.values()}
    assert "lead" in statuses
    assert "hail predates current roof" in statuses  # re-roofed homes are filtered


def test_summary_is_serialisable_and_labelled(demo_result) -> None:
    s = json.loads(json.dumps(demo_result.summary()))
    assert s["synthetic_data"] is True
    assert len(s["events"]) == 8
    merged = next(e for e in s["events"] if e["event_id"] == "EV-20220708")
    assert merged["days"] == ["2022-07-08", "2022-07-09"]
    assert merged["peak_in"] == pytest.approx(2.75)
    sub_severe = next(e for e in s["events"] if e["event_id"] == "EV-20250717")
    assert sub_severe["footprint_km2"] == {}
    assert sum(s["status_counts"].values()) == s["properties"]
    assert s["leads"] == s["status_counts"]["lead"]


def test_pipeline_is_deterministic(season, day_grids) -> None:
    def run() -> list[tuple[str, float]]:
        rng = np.random.default_rng(99)
        props = generate_properties(200, season.origin, season.as_of.year, rng)
        result = run_pipeline(props, [], day_grids, season.as_of, "t", synthetic=True)
        return [(ld.property.property_id, ld.score.score) for ld in result.leads]

    first = run()
    assert first == run()


def test_no_providers_means_no_recommendation(season, day_grids) -> None:
    props = generate_properties(50, season.origin, season.as_of.year, np.random.default_rng(3))
    result = run_pipeline(
        props, [], day_grids, season.as_of, "t", synthetic=False, missing_days=["2024-01-01"]
    )
    assert all(ld.provider is None for ld in result.leads)
    assert result.summary()["missing_storm_days"] == ["2024-01-01"]


def test_no_storm_days_is_an_error(season, day_grids) -> None:
    cfg = EngineConfig(history=HistoryConfig(storm_day_min_in=9.0))
    with pytest.raises(NoStormDataError, match="nothing to score"):
        run_pipeline([], [], day_grids, season.as_of, "t", synthetic=True, config=cfg)
