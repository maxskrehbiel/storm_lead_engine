"""Self-checks for the synthetic demo: the run is graded against the fixture's known truth."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .fixture import StormSeason
from .grid import HailGrid
from .history import parse_day
from .pipeline import RunResult

PEAK_TOLERANCE_IN = 0.001
SCORE_ROUNDING_TOLERANCE = 0.2


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one self-check.

    Attributes:
        name: What was checked.
        passed: True if the check held.
        detail: One line of evidence.
    """

    name: str
    passed: bool
    detail: str


def _expected_event_days(season: StormSeason, result: RunResult) -> list[tuple[str, ...]]:
    cfg = result.config.history
    peaks: dict[str, float] = {}
    for storm in season.storms:
        peaks[storm.date] = max(peaks.get(storm.date, 0.0), storm.peak_in)
    days = sorted(d for d, p in peaks.items() if p >= cfg.storm_day_min_in)
    groups: list[list[str]] = []
    for day in days:
        if groups and (parse_day(day) - parse_day(groups[-1][-1])).days <= cfg.event_gap_days:
            groups[-1].append(day)
        else:
            groups.append([day])
    return [tuple(g) for g in groups]


def _check_peaks(season: StormSeason, day_grids: dict[str, HailGrid]) -> CheckResult:
    worst = 0.0
    for storm in season.storms:
        same_day = max(s.peak_in for s in season.storms if s.date == storm.date)
        worst = max(worst, abs(day_grids[storm.date].max_in - same_day))
    return CheckResult(
        "storm-day peaks match the fixture",
        worst <= PEAK_TOLERANCE_IN,
        f"largest difference {worst:.4f} in over {len(season.storms)} storms",
    )


def _check_events(season: StormSeason, result: RunResult) -> CheckResult:
    expected = _expected_event_days(season, result)
    actual = [ev.days for ev in result.events]
    return CheckResult(
        "storm days group into the expected events",
        actual == expected,
        f"{len(actual)} events, expected {len(expected)}",
    )


def _check_leads(result: RunResult) -> list[CheckResult]:
    w = result.config.weights
    ineligible = [
        ld.property.property_id
        for ld in result.leads
        if ld.property.property_type not in w.eligible_types
        or ld.profile.max_in < w.min_lead_hail_in
        or any(parse_day(h.date).year <= ld.profile.roof_install_year for h in ld.profile.counted)
    ]
    scores = [ld.score.score for ld in result.leads]
    ranked = scores == sorted(scores, reverse=True) and [ld.rank for ld in result.leads] == list(
        range(1, len(result.leads) + 1)
    )
    unbalanced = [
        ld.property.property_id
        for ld in result.leads
        if abs(
            ld.score.score
            - (ld.score.hail_pts + ld.score.roof_pts + ld.score.value_pts + ld.score.occupancy_pts)
        )
        > SCORE_ROUNDING_TOLERANCE
    ]
    return [
        CheckResult(
            "every lead is eligible on its current roof",
            not ineligible,
            f"{len(result.leads)} leads, {len(ineligible)} ineligible",
        ),
        CheckResult("leads are ranked by score", ranked, f"{len(scores)} scores checked"),
        CheckResult(
            "score components add up to the score",
            not unbalanced,
            f"{len(unbalanced)} leads off by more than {SCORE_ROUNDING_TOLERANCE}",
        ),
    ]


def _check_footprints(result: RunResult) -> CheckResult:
    groups = [result.max_ever_footprints, *result.event_footprints.values()]
    nested = all(all(a.n_cells >= b.n_cells for a, b in itertools.pairwise(fps)) for fps in groups)
    return CheckResult(
        "footprints shrink as the threshold rises",
        nested,
        f"{sum(len(g) for g in groups)} footprints checked",
    )


def _check_shrinkage(result: RunResult) -> CheckResult:
    def rank_of(rating: float, reviews: int) -> int | None:
        return next(
            (
                sp.rank
                for sp in result.providers
                if sp.provider.rating == rating and sp.provider.review_count == reviews
            ),
            None,
        )

    thin, deep = rank_of(5.0, 3), rank_of(4.8, 400)
    if thin is None or deep is None:
        return CheckResult("4.8 from 400 reviews outranks 5.0 from 3", True, "pair not present")
    return CheckResult(
        "4.8 from 400 reviews outranks 5.0 from 3", deep < thin, f"ranks {deep} and {thin}"
    )


def demo_checks(
    result: RunResult, season: StormSeason, day_grids: dict[str, HailGrid]
) -> list[CheckResult]:
    """Grade a demo run against the synthetic truth it was built from.

    Args:
        result: Pipeline result from the demo.
        season: The fixture the storm grids came from.
        day_grids: The storm-day grids that were scored.

    Returns:
        One result per check, in a fixed order.
    """
    return [
        _check_peaks(season, day_grids),
        _check_events(season, result),
        *_check_leads(result),
        _check_footprints(result),
        _check_shrinkage(result),
        CheckResult(
            "every property has a status",
            len(result.scores) == len(result.properties),
            f"{len(result.scores)} of {len(result.properties)}",
        ),
    ]
