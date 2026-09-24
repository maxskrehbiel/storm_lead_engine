"""Event grouping, stacking and the per-property ledger rules."""

import pytest

from storm_lead_engine.config import HailTiers, HistoryConfig
from storm_lead_engine.history import (
    build_profiles,
    build_stack,
    group_events,
    profile_property,
)
from storm_lead_engine.properties import Property

TIERS = HailTiers()


def test_consecutive_days_merge_and_quiet_days_drop(day_grids) -> None:
    events = group_events(day_grids, HistoryConfig())
    assert [e.event_id for e in events] == [
        "EV-20210518", "EV-20210721", "EV-20220708", "EV-20230526",
        "EV-20240703", "EV-20240827", "EV-20250604", "EV-20250717",
    ]  # fmt: skip
    merged = events[2]
    assert merged.days == ("2022-07-08", "2022-07-09")
    assert merged.peak_in == pytest.approx(2.75, abs=1e-4)
    assert len(group_events(day_grids, HistoryConfig(event_gap_days=0))) == 9
    assert len(group_events(day_grids, HistoryConfig(storm_day_min_in=1.0))) == 7


def test_event_grid_is_cellwise_max_of_its_days(day_grids) -> None:
    ev = group_events(day_grids, HistoryConfig())[2]
    a, b = day_grids["2022-07-08"].inches, day_grids["2022-07-09"].inches
    assert (ev.grid.inches >= a).all() and (ev.grid.inches >= b).all()
    assert ((ev.grid.inches == a) | (ev.grid.inches == b)).all()


def _events(make_grid, grids_by_day):
    return group_events(
        {d: make_grid(v) for d, v in grids_by_day.items()},
        HistoryConfig(storm_day_min_in=0.5, event_gap_days=1),
    )


def test_stack_counts_and_max(make_grid) -> None:
    events = _events(
        make_grid,
        {"2021-06-01": [[1.2, 0.0]], "2023-06-01": [[1.6, 2.1]], "2025-06-01": [[0.8, 1.0]]},
    )
    stack = build_stack(events, TIERS)
    assert stack.cube.shape == (3, 1, 2)
    assert stack.grid_max.inches[0].tolist() == pytest.approx([1.6, 2.1])
    assert stack.counts[1.0].tolist() == [[2, 2]]
    assert stack.counts[1.5].tolist() == [[1, 1]]
    assert stack.counts[2.0].tolist() == [[0, 1]]
    with pytest.raises(ValueError, match="no storm events"):
        build_stack([], TIERS)


def test_stack_rejects_mixed_lattices(make_grid) -> None:
    a = group_events({"2021-06-01": make_grid([[1.0]])}, HistoryConfig())
    b = group_events({"2023-06-01": make_grid([[1.0]], west=5.0)}, HistoryConfig())
    with pytest.raises(ValueError, match="different lattice"):
        build_stack(a + b, TIERS)


def test_only_hail_after_the_roof_year_counts(make_grid) -> None:
    events = _events(
        make_grid,
        {"2021-06-01": [[2.2]], "2023-06-01": [[1.6]], "2025-06-01": [[1.1]]},
    )
    stack = build_stack(events, TIERS)
    lat, lon = 0.995, 0.005
    original = Property("A", lat, lon, year_built=1990)
    reroofed = Property("B", lat, lon, year_built=1990, roof_year=2023)

    a = profile_property(original, stack, TIERS, neighborhood=1)
    assert (a.n_severe, a.n_damaging, a.n_extreme) == (3, 2, 1)
    assert a.max_in == pytest.approx(2.2) and a.worst_event == "EV-20210601"
    assert a.last_severe_date == "2025-06-01"
    assert a.roof_source == "year_built" and a.excluded_pre_roof == 0

    b = profile_property(reroofed, stack, TIERS, neighborhood=1)
    assert [h.event_id for h in b.counted] == ["EV-20250601"]  # 2023 storm: same year, excluded
    assert (b.n_severe, b.n_damaging, b.max_in) == (1, 0, pytest.approx(1.1))
    assert b.excluded_pre_roof == 2 and b.roof_source == "roof_record"


def test_all_events_read_from_one_anchor_cell(make_grid) -> None:
    # Two neighbours: west cell has the all-time max, east cell has more storms.
    events = _events(
        make_grid,
        {"2021-06-01": [[2.5, 0.0]], "2023-06-01": [[0.0, 1.2]], "2025-06-01": [[0.0, 1.2]]},
    )
    stack = build_stack(events, TIERS)
    prof = profile_property(Property("P", 0.995, 0.015, year_built=1990), stack, TIERS, 1)
    # anchored to the worst (west) cell; the east cell's two storms are not mixed in
    assert prof.max_in == pytest.approx(2.5)
    assert prof.n_severe == 1


def test_outside_region_profile(make_grid, day_grids) -> None:
    stack = build_stack(group_events(day_grids, HistoryConfig()), TIERS)
    far = Property("FAR", 10.0, 10.0, year_built=2000)
    prof = build_profiles([far], stack, TIERS, HistoryConfig())["FAR"]
    assert not prof.in_region
    assert prof.hits == () and prof.max_in == 0.0
