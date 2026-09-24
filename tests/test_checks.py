"""Demo self-checks: they pass on a correct run and catch each kind of corruption."""

import dataclasses

from storm_lead_engine.checks import demo_checks
from storm_lead_engine.fixture import StormSeason


def _failed(checks) -> set[str]:
    return {c.name for c in checks if not c.passed}


def test_all_checks_pass_on_the_demo(demo_result, season, day_grids) -> None:
    checks = demo_checks(demo_result, season, day_grids)
    assert len(checks) == 8
    assert _failed(checks) == set()


def test_a_wrong_peak_is_caught(demo_result, season, day_grids) -> None:
    first = dataclasses.replace(season.storms[0], peak_in=season.storms[0].peak_in + 0.5)
    tampered = StormSeason(
        season.name, season.description, season.origin, season.region, season.as_of,
        (first, *season.storms[1:]),
    )  # fmt: skip
    assert "storm-day peaks match the fixture" in _failed(
        demo_checks(demo_result, tampered, day_grids)
    )
    assert "storm days group into the expected events" not in _failed(
        demo_checks(demo_result, tampered, day_grids)
    )


def test_bad_ranking_and_events_are_caught(demo_result, season, day_grids) -> None:
    reordered = dataclasses.replace(
        demo_result, leads=tuple(reversed(demo_result.leads)), events=demo_result.events[1:]
    )
    failed = _failed(demo_checks(reordered, season, day_grids))
    assert "leads are ranked by score" in failed
    assert "storm days group into the expected events" in failed


def test_shrinkage_check_skips_without_the_canonical_pair(demo_result, season, day_grids) -> None:
    no_providers = dataclasses.replace(demo_result, providers=())
    shrink = next(
        c for c in demo_checks(no_providers, season, day_grids) if c.name.startswith("4.8")
    )
    assert shrink.passed and shrink.detail == "pair not present"
