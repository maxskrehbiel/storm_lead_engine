"""Orchestration: storm-day grids, properties and providers in, a ranked and explained result out.

The pipeline is a pure function of its inputs (no network, no clock), so equal
inputs always produce equal rankings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from .config import EngineConfig
from .errors import NoStormDataError
from .grid import HailGrid
from .history import HailProfile, HailStack, StormEvent, build_profiles, build_stack, group_events
from .properties import Property
from .providers import Provider
from .scoring import LeadScore, ScoredProvider, recommend_provider, score_property, score_providers
from .swaths import Footprint, footprints

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Lead:
    """A ranked lead with everything needed to explain it.

    Attributes:
        rank: 1 = highest score.
        property: The property.
        profile: Its hail ledger.
        score: Its score breakdown.
        outreach_channel: ``door_knock`` when owner-occupied, otherwise ``mail``.
        provider: Recommended provider, or None if nobody qualified.
    """

    rank: int
    property: Property
    profile: HailProfile
    score: LeadScore
    outreach_channel: str
    provider: ScoredProvider | None


@dataclass(frozen=True)
class RunResult:
    """Everything one pipeline run produced.

    Attributes:
        config: Configuration used.
        as_of: Date roof ages are measured at.
        source: Human-readable description of the storm data.
        synthetic: True when any input is synthetic.
        missing_days: Requested storm days that could not be read.
        events: Storm events.
        stack: Cumulative layers.
        event_footprints: Footprints per event id.
        max_ever_footprints: All-time footprints.
        properties: Property layer.
        profiles: Hail ledgers by property id.
        scores: Scores by property id.
        leads: Eligible properties, ranked.
        providers: Providers, ranked.
    """

    config: EngineConfig
    as_of: date
    source: str
    synthetic: bool
    missing_days: tuple[str, ...]
    events: tuple[StormEvent, ...]
    stack: HailStack
    event_footprints: dict[str, list[Footprint]]
    max_ever_footprints: list[Footprint]
    properties: tuple[Property, ...]
    profiles: dict[str, HailProfile]
    scores: dict[str, LeadScore]
    leads: tuple[Lead, ...]
    providers: tuple[ScoredProvider, ...]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable run summary (counts, areas, parameters)."""
        statuses: dict[str, int] = {}
        for s in self.scores.values():
            statuses[s.status] = statuses.get(s.status, 0) + 1
        cfg = self.config
        return {
            "synthetic_data": self.synthetic,
            "source": self.source,
            "as_of": self.as_of.isoformat(),
            "missing_storm_days": list(self.missing_days),
            "events": [
                {
                    "event_id": ev.event_id,
                    "days": list(ev.days),
                    "peak_in": round(ev.peak_in, 2),
                    "footprint_km2": {
                        f"{fp.threshold_in:.2f}in": round(fp.area_km2, 1)
                        for fp in self.event_footprints[ev.event_id]
                    },
                }
                for ev in self.events
            ],
            "max_ever_footprint_km2": {
                f"{fp.threshold_in:.2f}in": round(fp.area_km2, 1) for fp in self.max_ever_footprints
            },
            "properties": len(self.properties),
            "properties_ever_hit": {
                f"{t:.2f}in": sum(
                    any(h.size_in >= t for h in pr.hits) for pr in self.profiles.values()
                )
                for t in cfg.tiers.as_tuple()
            },
            "leads": len(self.leads),
            "status_counts": dict(sorted(statuses.items())),
            "providers": len(self.providers),
            "parameters": {
                "tiers_in": list(cfg.tiers.as_tuple()),
                "storm_day_min_in": cfg.history.storm_day_min_in,
                "event_gap_days": cfg.history.event_gap_days,
                "neighborhood_cells": cfg.history.neighborhood,
                "min_lead_hail_in": cfg.weights.min_lead_hail_in,
                "lead_weights": {
                    "hail": cfg.weights.hail,
                    "roof_age": cfg.weights.roof_age,
                    "value": cfg.weights.value,
                    "occupancy": cfg.weights.occupancy,
                },
                "provider_prior_weight": cfg.providers.prior_weight,
            },
        }


def score_all(
    properties: list[Property], profiles: dict[str, HailProfile], cfg: EngineConfig, as_of: date
) -> dict[str, LeadScore]:
    """Score every property, keyed by property id (see :func:`scoring.score_property`)."""
    return {
        p.property_id: score_property(
            p, profiles[p.property_id], cfg.weights, cfg.confidence, as_of.year
        )
        for p in properties
    }


def rank_leads(
    properties: list[Property],
    profiles: dict[str, HailProfile],
    scores: dict[str, LeadScore],
    providers: list[ScoredProvider],
    cfg: EngineConfig,
) -> tuple[Lead, ...]:
    """Order eligible properties by score, then pick a channel and a provider for each.

    Ties break on the larger stone, then on property id, so the order is total.

    Args:
        properties: Property layer.
        profiles: Hail ledgers by property id.
        scores: Scores by property id.
        providers: Providers, ranked.
        cfg: Engine configuration.

    Returns:
        Leads, best first.
    """
    eligible = sorted(
        (p for p in properties if scores[p.property_id].eligible),
        key=lambda p: (
            -scores[p.property_id].score,
            -profiles[p.property_id].max_in,
            p.property_id,
        ),
    )
    return tuple(
        Lead(
            rank=i,
            property=p,
            profile=profiles[p.property_id],
            score=scores[p.property_id],
            outreach_channel="door_knock" if p.owner_occupied else "mail",
            provider=recommend_provider(p.lat, p.lon, providers, cfg.providers),
        )
        for i, p in enumerate(eligible, start=1)
    )


def run_pipeline(
    properties: list[Property],
    providers: list[Provider],
    day_grids: dict[str, HailGrid],
    as_of: date,
    source: str,
    synthetic: bool,
    config: EngineConfig | None = None,
    missing_days: list[str] | None = None,
) -> RunResult:
    """Run every stage: events, stack, footprints, profiles, scores, ranking, routing.

    Args:
        properties: Property layer.
        providers: Service providers (may be empty).
        day_grids: Storm-day grids on one lattice.
        as_of: Date roof ages are measured at.
        source: Description of the storm data, recorded in outputs.
        synthetic: Whether any input is synthetic; recorded in outputs.
        config: Engine configuration; defaults to :class:`EngineConfig`.
        missing_days: Storm days that could not be read; recorded in outputs.

    Returns:
        The run result.

    Raises:
        NoStormDataError: If no day reaches the storm-day threshold.
    """
    cfg = config or EngineConfig()
    tiers = cfg.tiers.as_tuple()
    events = group_events(day_grids, cfg.history)
    if not events:
        raise NoStormDataError(
            f"no storm day reached {cfg.history.storm_day_min_in} in; nothing to score"
        )
    logger.info("%d storm days -> %d events", len(day_grids), len(events))
    for ev in events:
        logger.debug("%s: days %s, peak %.2f in", ev.event_id, ", ".join(ev.days), ev.peak_in)
    stack = build_stack(events, cfg.tiers)
    profiles = build_profiles(properties, stack, cfg.tiers, cfg.history)
    scores = score_all(properties, profiles, cfg, as_of)
    ranked_providers = score_providers(providers, cfg.providers)
    leads = rank_leads(properties, profiles, scores, ranked_providers, cfg)
    logger.info("%d of %d properties are leads", len(leads), len(properties))
    return RunResult(
        config=cfg,
        as_of=as_of,
        source=source,
        synthetic=synthetic,
        missing_days=tuple(missing_days or ()),
        events=tuple(events),
        stack=stack,
        event_footprints={ev.event_id: footprints(ev.grid, tiers, ev.event_id) for ev in events},
        max_ever_footprints=footprints(stack.grid_max, tiers, "max-ever"),
        properties=tuple(properties),
        profiles=profiles,
        scores=scores,
        leads=leads,
        providers=tuple(ranked_providers),
    )
