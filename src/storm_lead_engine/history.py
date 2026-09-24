"""Hail history: storm days grouped into events, stacked, and read into a per-property ledger.

Three rules keep the history honest:

1. One storm, one count. Storm days no more than ``event_gap_days`` apart merge
   into one event, and the event grid is the cell-wise max over its days.
2. One cell per property. A property is pinned to a single anchor cell (the
   worst cell of the all-time max grid within its geocode-tolerance window), and
   every event is read from that cell, so counts and maxima never mix neighbours.
3. Only the current roof counts. Hail that fell before the current roof went on
   cannot have damaged it. Events count only when their year is after the roof
   install year. Same-year storms are excluded because a same-year storm is
   most often why the roof was replaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from ._types import CountGrid, FloatGrid
from .config import HailTiers, HistoryConfig
from .grid import HailGrid
from .properties import Property


def parse_day(text: str) -> date:
    """Parse a ``YYYY-MM-DD`` date (shared by the fixture loader, history and the CLI).

    Raises:
        ValueError: If the text is not a valid ``YYYY-MM-DD`` date.
    """
    return datetime.strptime(text.strip(), "%Y-%m-%d").date()


@dataclass(frozen=True)
class StormEvent:
    """One storm, possibly spanning consecutive days.

    Attributes:
        event_id: ``EV-YYYYMMDD`` of the first day.
        days: Storm days in the event, ascending.
        grid: Cell-wise maximum over the event's days.
    """

    event_id: str
    days: tuple[str, ...]
    grid: HailGrid = field(repr=False)

    @property
    def peak_in(self) -> float:
        """Largest stone anywhere in the event, inches."""
        return self.grid.max_in


def group_events(day_grids: dict[str, HailGrid], cfg: HistoryConfig) -> list[StormEvent]:
    """Drop quiet days, then chain storm days within ``event_gap_days`` into events.

    Args:
        day_grids: Storm-day grids keyed by ``YYYY-MM-DD``.
        cfg: Storm-day threshold and gap.

    Returns:
        Events in chronological order.
    """
    storm_days = sorted(d for d, g in day_grids.items() if g.max_in >= cfg.storm_day_min_in)
    groups: list[list[str]] = []
    for day in storm_days:
        if groups and (parse_day(day) - parse_day(groups[-1][-1])).days <= cfg.event_gap_days:
            groups[-1].append(day)
        else:
            groups.append([day])
    return [
        StormEvent(
            event_id=f"EV-{g[0].replace('-', '')}",
            days=tuple(g),
            grid=HailGrid.pixel_max(day_grids[d] for d in g),
        )
        for g in groups
    ]


@dataclass(frozen=True)
class HailStack:
    """Cell-wise cumulative layers across all events.

    Attributes:
        grid_max: Largest hail ever recorded in each cell.
        counts: Tier threshold to the number of events reaching it, per cell.
        events: The events that were stacked.
        cube: Every event grid as one ``(n_events, rows, cols)`` array, built once.
    """

    grid_max: HailGrid
    counts: dict[float, CountGrid]
    events: tuple[StormEvent, ...]
    cube: FloatGrid = field(repr=False)


def build_stack(events: list[StormEvent], tiers: HailTiers) -> HailStack:
    """Stack event grids into cumulative layers.

    Args:
        events: Events on a shared lattice.
        tiers: Thresholds to count.

    Returns:
        The stack.

    Raises:
        ValueError: If there are no events or their lattices differ.
    """
    if not events:
        raise ValueError("no storm events to stack")
    base = events[0].grid
    for ev in events[1:]:
        if not base.same_lattice(ev.grid):
            raise ValueError(f"{ev.event_id} is on a different lattice")
    cube = np.stack([ev.grid.inches for ev in events])
    counts = {t: (cube >= t).sum(axis=0).astype(np.int16) for t in tiers.as_tuple()}
    return HailStack(HailGrid(cube.max(axis=0), base.bounds), counts, tuple(events), cube)


@dataclass(frozen=True)
class EventHit:
    """Hail at one property's anchor cell during one event.

    Attributes:
        event_id: Event identifier.
        date: First day of the event.
        size_in: Hail size at the anchor cell, inches.
    """

    event_id: str
    date: str
    size_in: float


@dataclass(frozen=True)
class HailProfile:
    """A property's hail ledger.

    Attributes:
        property_id: Property identifier.
        in_region: False if the property is outside the storm grids.
        roof_install_year: Year the current roof went on.
        roof_source: ``roof_record`` or ``year_built``.
        hits: Every event with any hail at the anchor cell.
        counted: Hits on the current roof only.
        n_severe: Counted events at or above the severe tier.
        n_damaging: Counted events at or above the damaging tier.
        n_extreme: Counted events at or above the extreme tier.
        max_in: Largest counted stone, inches.
        worst_event: Event id of ``max_in``.
        last_severe_date: Most recent counted severe event.
        excluded_pre_roof: Severe hits that predate the current roof.
    """

    property_id: str
    in_region: bool
    roof_install_year: int
    roof_source: str
    hits: tuple[EventHit, ...] = ()
    counted: tuple[EventHit, ...] = ()
    n_severe: int = 0
    n_damaging: int = 0
    n_extreme: int = 0
    max_in: float = 0.0
    worst_event: str = ""
    last_severe_date: str = ""
    excluded_pre_roof: int = 0


def profile_property(
    p: Property, stack: HailStack, tiers: HailTiers, neighborhood: int
) -> HailProfile:
    """Build one property's ledger from the stack.

    Args:
        p: The property.
        stack: Cumulative layers (anchor-cell choice and per-event values).
        tiers: Tier thresholds.
        neighborhood: Geocode tolerance in cells.

    Returns:
        The property's hail profile.
    """
    roof_year, roof_src = p.roof_install_year()
    anchor = stack.grid_max.anchor_cell(p.lat, p.lon, neighborhood)
    if anchor is None:
        return HailProfile(p.property_id, False, roof_year, roof_src)
    sizes = stack.cube[:, anchor[0], anchor[1]]
    hits = tuple(
        EventHit(ev.event_id, ev.days[0], round(float(s), 2))
        for ev, s in zip(stack.events, sizes, strict=True)
        if s > 0
    )
    counted = tuple(h for h in hits if parse_day(h.date).year > roof_year)
    severe = [h for h in counted if h.size_in >= tiers.severe]
    worst = max(counted, key=lambda h: h.size_in, default=None)
    return HailProfile(
        property_id=p.property_id,
        in_region=True,
        roof_install_year=roof_year,
        roof_source=roof_src,
        hits=hits,
        counted=counted,
        n_severe=len(severe),
        n_damaging=sum(h.size_in >= tiers.damaging for h in counted),
        n_extreme=sum(h.size_in >= tiers.extreme for h in counted),
        max_in=worst.size_in if worst else 0.0,
        worst_event=worst.event_id if worst else "",
        last_severe_date=max((h.date for h in severe), default=""),
        excluded_pre_roof=sum(
            h.size_in >= tiers.severe and parse_day(h.date).year <= roof_year for h in hits
        ),
    )


def build_profiles(
    properties: list[Property], stack: HailStack, tiers: HailTiers, cfg: HistoryConfig
) -> dict[str, HailProfile]:
    """Build every property's ledger.

    Args:
        properties: Property layer.
        stack: Cumulative layers.
        tiers: Tier thresholds.
        cfg: Sampling rules.

    Returns:
        Profiles keyed by property id.
    """
    return {p.property_id: profile_property(p, stack, tiers, cfg.neighborhood) for p in properties}
