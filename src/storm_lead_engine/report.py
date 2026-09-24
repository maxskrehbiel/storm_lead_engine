"""Outputs: ranked CSVs, GeoJSON footprints, a JSON run summary and an interactive HTML map.

Every text artifact is byte-stable across runs and platforms: LF line endings, a
trailing newline, sequential HTML element IDs and no timestamps.
"""

from __future__ import annotations

import base64
import html
import itertools
import json
import math
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folium
import numpy as np

from ._types import RgbaImage
from .csv_io import write_records
from .grid import HailGrid
from .pipeline import Lead, RunResult
from .swaths import Footprint, feature_collection

# Sequential single-hue blue for hail magnitude (light = small). Stops are
# (inches, hex, alpha 0-255); alpha rises with size so small hail stays faint.
HAIL_RAMP: tuple[tuple[float, str, int], ...] = (
    (0.25, "#cde2fb", 60),
    (0.75, "#9ec5f4", 110),
    (1.00, "#6da7ec", 140),
    (1.50, "#2a78d6", 170),
    (2.00, "#184f95", 195),
    (2.75, "#0d366b", 215),
)
# Ordinal blues for the three footprint tiers, and ordinal oranges for lead-score
# bins. Both ramps pass a colour-vision-deficiency and contrast check on a light surface.
TIER_COLORS = ("#6da7ec", "#2a78d6", "#184f95")
SCORE_BINS: tuple[tuple[float, str], ...] = (
    (0.0, "#f0956a"),
    (45.0, "#eb6834"),
    (60.0, "#b9481b"),
    (75.0, "#7a2c0c"),
)
NEUTRAL_POINT = "#9a9994"
PROVIDER_COLOR = "#2b2a27"
SURFACE_RING = "#ffffff"
METERS_PER_KM = 1000
# A leading =, +, -, @, tab or CR makes spreadsheet apps evaluate a cell as a formula.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
# folium names every element "<kind>_<uuid4 hex>", which changes on every run.
_FOLIUM_ID = re.compile(r"(?<=_)[0-9a-f]{32}(?![0-9a-f])")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# A stored (uncompressed) deflate block holds at most 65535 bytes.
STORED_BLOCK_MAX = 65535
# zlib header for deflate with a 32 KiB window and no preset dictionary.
ZLIB_HEADER = b"\x78\x01"


@dataclass(frozen=True)
class MapStyle:
    """Layout constants for the HTML map.

    Attributes:
        lead_radius_px: Marker radius for leads.
        other_radius_px: Marker radius for non-lead properties.
        provider_radius_px: Marker radius for providers.
        ring_px: White ring drawn around point markers so overlaps stay legible.
        footprint_stroke_px: Outline width of footprint polygons.
        marker_fill_opacity: Fill opacity of point markers.
        zoom_start: Initial zoom level.
        panel_top_leads: Rows in the dashboard's top-leads table.
        panel_top_providers: Rows in the dashboard's provider table.
    """

    lead_radius_px: float = 4.5
    other_radius_px: float = 2.5
    provider_radius_px: float = 7.0
    ring_px: float = 1.0
    footprint_stroke_px: float = 2.0
    marker_fill_opacity: float = 0.95
    zoom_start: int = 12
    panel_top_leads: int = 10
    panel_top_providers: int = 6


LEAD_COLUMNS = (
    "rank", "property_id", "address", "lat", "lon", "score",
    "hail_pts", "roof_pts", "value_pts", "occupancy_pts", "hail_confidence",
    "max_hail_in", "worst_event", "last_severe_event",
    "severe_events", "damaging_events", "extreme_events", "pre_roof_severe_events",
    "roof_age", "roof_age_source", "year_built", "assessed_value", "owner_occupied",
    "property_type", "outreach_channel", "provider_id", "provider_quality",
)  # fmt: skip

PROVIDER_COLUMNS = (
    "rank", "provider_id", "name", "rating", "review_count", "prior_mean",
    "shrunk_rating", "quality", "evidence", "service_radius_km", "lat", "lon",
)  # fmt: skip


def csv_safe(text: str) -> str:
    """Neutralise user text that a spreadsheet would run as a formula (CSV injection)."""
    return "'" + text if text.startswith(_FORMULA_PREFIXES) else text


def lead_row(lead: Lead) -> dict[str, Any]:
    """Flatten a lead into one CSV row (see ``LEAD_COLUMNS``)."""
    p, prof, s = lead.property, lead.profile, lead.score
    return {
        "rank": lead.rank,
        "property_id": csv_safe(p.property_id),
        "address": csv_safe(p.address),
        "lat": p.lat,
        "lon": p.lon,
        "score": s.score,
        "hail_pts": s.hail_pts,
        "roof_pts": s.roof_pts,
        "value_pts": s.value_pts,
        "occupancy_pts": s.occupancy_pts,
        "hail_confidence": s.hail_confidence,
        "max_hail_in": prof.max_in,
        "worst_event": prof.worst_event,
        "last_severe_event": prof.last_severe_date,
        "severe_events": prof.n_severe,
        "damaging_events": prof.n_damaging,
        "extreme_events": prof.n_extreme,
        "pre_roof_severe_events": prof.excluded_pre_roof,
        "roof_age": s.roof_age,
        "roof_age_source": prof.roof_source,
        "year_built": p.year_built,
        "assessed_value": None if p.assessed_value is None else round(p.assessed_value),
        "owner_occupied": p.owner_occupied,
        "property_type": p.property_type,
        "outreach_channel": lead.outreach_channel,
        "provider_id": csv_safe(lead.provider.provider.provider_id) if lead.provider else None,
        "provider_quality": lead.provider.quality if lead.provider else None,
    }


def write_leads_csv(result: RunResult, path: Path, top: int | None = None) -> int:
    """Write ranked leads to CSV.

    Args:
        result: Pipeline result.
        path: Output file.
        top: Keep only the first ``top`` leads; None keeps all.

    Returns:
        Number of rows written.
    """
    leads = result.leads[:top] if top else result.leads
    return write_records((lead_row(ld) for ld in leads), path, LEAD_COLUMNS)


def write_ranked_providers_csv(result: RunResult, path: Path) -> int:
    """Write the provider ranking (raw and shrunk ratings side by side) to CSV.

    Args:
        result: Pipeline result.
        path: Output file.

    Returns:
        Number of rows written.
    """
    rows = (
        {
            "rank": sp.rank,
            "provider_id": csv_safe(sp.provider.provider_id),
            "name": csv_safe(sp.provider.name),
            "rating": sp.provider.rating,
            "review_count": sp.provider.review_count,
            "prior_mean": sp.prior_mean,
            "shrunk_rating": sp.shrunk_rating,
            "quality": sp.quality,
            "evidence": sp.evidence,
            "service_radius_km": sp.provider.service_radius_km,
            "lat": sp.provider.lat,
            "lon": sp.provider.lon,
        }
        for sp in result.providers
    )
    return write_records(rows, path, PROVIDER_COLUMNS)


def all_footprints(result: RunResult) -> list[Footprint]:
    """All-time footprints followed by every event's footprints."""
    out = list(result.max_ever_footprints)
    for ev in result.events:
        out.extend(result.event_footprints[ev.event_id])
    return out


def _write_text(path: Path, text: str) -> None:
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")


def write_footprints_geojson(result: RunResult, path: Path) -> int:
    """Write every footprint as one GeoJSON FeatureCollection.

    Args:
        result: Pipeline result.
        path: Output file.

    Returns:
        Number of features written.
    """
    fps = all_footprints(result)
    fc = feature_collection(fps, {"synthetic_data": result.synthetic, "source": result.source})
    _write_text(path, json.dumps(fc, separators=(",", ":")))
    return len(fps)


def write_summary_json(result: RunResult, path: Path) -> None:
    """Write the run summary as indented JSON."""
    _write_text(path, json.dumps(result.summary(), indent=2))


def _hex_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def colorize_hail(grid: HailGrid) -> RgbaImage:
    """Render a hail grid as RGBA using ``HAIL_RAMP``; cells below the first stop are clear.

    Args:
        grid: Hail grid in inches.

    Returns:
        ``(rows, cols, 4)`` uint8 image, north up.
    """
    x = grid.inches
    stops = np.array([s[0] for s in HAIL_RAMP])
    rgb = np.array([_hex_rgb(s[1]) for s in HAIL_RAMP], dtype=float)
    alpha = np.array([s[2] for s in HAIL_RAMP], dtype=float)
    channels = [np.interp(x, stops, rgb[:, i]) for i in range(3)]
    channels.append(np.where(x < stops[0], 0.0, np.interp(x, stops, alpha)))
    image: RgbaImage = np.rint(np.dstack(channels)).clip(0, 255).astype(np.uint8)
    return image


def _mercator_y(lat: float) -> float:
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def to_web_mercator(img: RgbaImage, south: float, north: float) -> RgbaImage:
    """Resample a north-up latitude/longitude image so it lines up on a Web Mercator map.

    Each output row is a copy of the nearest input row, so the result never
    depends on floating-point interpolation noise and stays byte-stable.

    Args:
        img: ``(rows, cols, 4)`` image whose rows are equal steps of latitude.
        south: Latitude of the bottom edge.
        north: Latitude of the top edge.

    Returns:
        An image of the same shape whose rows are equal steps of Mercator y.
    """
    h = img.shape[0]
    y_south, y_north = _mercator_y(south), _mercator_y(north)
    rows = []
    for r in range(h):
        y = y_north - (r + 0.5) * (y_north - y_south) / h
        lat = math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)
        rows.append(min(h - 1, max(0, int((north - lat) / (north - south) * h))))
    return img[rows]


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png_bytes(img: RgbaImage) -> bytes:
    """Encode an RGBA image as PNG using stored (uncompressed) deflate blocks.

    zlib and zlib-ng compress the same pixels to different bytes. Stored blocks
    are fixed byte-for-byte by the format, so the file is identical on every
    platform. The images here are small, so the size cost is a few tens of KB.

    Args:
        img: ``(rows, cols, 4)`` uint8 image.

    Returns:
        PNG file bytes.
    """
    height, width = img.shape[0], img.shape[1]
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(height))  # filter type 0 per row
    blocks = []
    for start in range(0, len(raw), STORED_BLOCK_MAX):
        part = raw[start : start + STORED_BLOCK_MAX]
        final = int(start + STORED_BLOCK_MAX >= len(raw))
        blocks.append(bytes([final]) + struct.pack("<HH", len(part), 0xFFFF ^ len(part)) + part)
    stream = ZLIB_HEADER + b"".join(blocks) + struct.pack(">I", zlib.adler32(raw))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", stream)
        + _png_chunk(b"IEND", b"")
    )


def score_color(score: float) -> str:
    """Colour of the ordinal score bin that contains ``score``."""
    color = SCORE_BINS[0][1]
    for lo, c in SCORE_BINS:
        if score >= lo:
            color = c
    return color


def stable_ids(page: str) -> str:
    """Replace folium's random element IDs with sequential ones, in order of appearance.

    Args:
        page: Rendered HTML.

    Returns:
        The same page with IDs ``0001``, ``0002``, ... so equal inputs give equal bytes.
    """
    mapping: dict[str, str] = {}

    def renumber(m: re.Match[str]) -> str:
        return mapping.setdefault(m.group(0), f"{len(mapping) + 1:04d}")

    return _FOLIUM_ID.sub(renumber, page)


def _point_feature(lat: float, lon: float, props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }


def _circle_style(fill: str, radius: float, st: MapStyle) -> dict[str, Any]:
    return {
        "radius": radius,
        "fill": True,
        "fillColor": fill,
        "color": SURFACE_RING,
        "weight": st.ring_px,
        "fillOpacity": st.marker_fill_opacity,
        "opacity": 1.0,
    }


def _stars(rating: float | None) -> str:
    return "-" if rating is None else f"{rating:.2f}"


PANEL_CSS = """
.folium-map { background: #f4f3ef; }
.sle-panel, .sle-banner {
  --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --rule: #e4e3df;
  font: 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink);
}
@media (prefers-color-scheme: dark) {
  .sle-panel, .sle-banner { --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --rule: #383835; }
}
.sle-banner { position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000;
  background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 6px 12px; font-weight: 600; max-width: calc(100% - 32px); text-align: center; }
.sle-panel { position: fixed; bottom: 16px; left: 16px; z-index: 1000; width: 340px;
  max-width: calc(100% - 32px); max-height: 70vh; overflow: auto; background: var(--surface);
  border: 1px solid var(--rule); border-radius: 8px; padding: 12px 14px; }
.sle-panel h2 { font-size: 14px; margin: 0 0 4px; }
.sle-panel h3 { font-size: 12px; margin: 10px 0 4px; color: var(--ink-2); font-weight: 600; }
.sle-panel .meta { color: var(--ink-2); }
.sle-panel table { border-collapse: collapse; width: 100%; }
.sle-panel td, .sle-panel th { padding: 2px 4px; border-bottom: 1px solid var(--rule);
  text-align: left; }
.sle-panel th { color: var(--ink-2); font-weight: 600; }
.sle-panel .n { text-align: right; font-variant-numeric: tabular-nums; }
.sle-panel .sw { display: inline-block; width: 12px; height: 9px; vertical-align: middle;
  margin-right: 3px; }
.sle-panel .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  vertical-align: middle; margin-right: 3px; }
@media (max-width: 600px) {
  .sle-panel { max-height: 38vh; bottom: 8px; left: 8px; }
  .sle-banner { font-size: 11px; top: 56px; }
}
"""


def _legend_html(result: RunResult) -> str:
    tier_rows = "".join(
        f'<span class="sw" style="border:2px solid {c}"></span>&ge; {t:.2f} in &nbsp;'
        for t, c in zip(result.config.tiers.as_tuple(), TIER_COLORS, strict=True)
    )
    bounds = [lo for lo, _ in SCORE_BINS[1:]]
    labels = [f"&lt; {bounds[0]:.0f}"]
    labels += [f"{a:.0f}-{b:.0f}" for a, b in itertools.pairwise(bounds)]
    labels += [f"&ge; {bounds[-1]:.0f}"]
    score_rows = "".join(
        f'<span class="dot" style="background:{c}"></span>{lab} &nbsp;'
        for (_, c), lab in zip(SCORE_BINS, labels, strict=True)
    )
    return (
        f"<h3>Hail footprints (outline)</h3><div>{tier_rows}</div>"
        f"<h3>Lead score</h3><div>{score_rows}"
        f'<span class="dot" style="background:{NEUTRAL_POINT}"></span>not a lead</div>'
    )


def _panel_html(result: RunResult, style: MapStyle) -> str:
    s = result.summary()
    esc = html.escape
    lead_rows = "".join(
        f"<tr><td>{ld.rank}</td><td>{esc(ld.property.address or ld.property.property_id)}</td>"
        f"<td class='n'>{ld.score.score:.1f}</td><td class='n'>{ld.profile.n_severe}</td>"
        f"<td class='n'>{ld.profile.max_in:.2f}</td></tr>"
        for ld in result.leads[: style.panel_top_leads]
    )
    prov_rows = "".join(
        f"<tr><td>{esc(sp.provider.name)}</td>"
        f"<td class='n'>{_stars(sp.provider.rating)}</td>"
        f"<td class='n'>{sp.provider.review_count}</td>"
        f"<td class='n'>{sp.shrunk_rating:.2f}</td></tr>"
        for sp in result.providers[: style.panel_top_providers]
    )
    status = " &middot; ".join(f"{esc(k)}: {v}" for k, v in s["status_counts"].items())
    banner = (
        '<div class="sle-banner">SYNTHETIC DEMO DATA: fictional storms, town, properties and '
        "providers on an abstract plane with no basemap.</div>"
        if result.synthetic
        else ""
    )
    return f"""<style>{PANEL_CSS}</style>
{banner}
<div class="sle-panel">
  <h2>storm_lead_engine</h2>
  <div class="meta">{esc(result.source)} &middot; as of {esc(s["as_of"])}</div>
  <div class="meta">{len(result.events)} storm events &middot; {s["properties"]} properties
    &middot; <b>{s["leads"]} leads</b></div>
  <div class="meta">{status}</div>
  {_legend_html(result)}
  <h3>Top leads</h3>
  <table><tr><th>#</th><th>Address</th><th class="n">Score</th><th class="n">Severe</th>
    <th class="n">Max in</th></tr>{lead_rows}</table>
  <h3>Providers: raw vs Bayesian-shrunk rating</h3>
  <table><tr><th>Provider</th><th class="n">Raw</th><th class="n">Reviews</th>
    <th class="n">Shrunk</th></tr>{prov_rows}</table>
</div>
"""


def _lead_props(ld: Lead) -> dict[str, Any]:
    s, prof = ld.score, ld.profile
    return {
        "fill": score_color(s.score),
        "rank": ld.rank,
        "address": html.escape(ld.property.address or ld.property.property_id),
        "score": s.score,
        "breakdown": (
            f"hail {s.hail_pts} + roof {s.roof_pts} + value {s.value_pts}"
            f" + occupancy {s.occupancy_pts}"
        ),
        "hail_history": (
            f"{prof.n_severe} severe / {prof.n_damaging} damaging / {prof.n_extreme} extreme;"
            f" max {prof.max_in:.2f} in ({prof.worst_event})"
        ),
        "roof": f"{s.roof_age} yrs ({prof.roof_source})",
        "channel": ld.outreach_channel,
        "provider": html.escape(ld.provider.provider.name) if ld.provider else "none in range",
    }


def _add_footprints(
    m: folium.Map, result: RunResult, fps: list[Footprint], name: str, show: bool, st: MapStyle
) -> None:
    group = folium.FeatureGroup(name=name, show=show)
    tiers = result.config.tiers.as_tuple()
    for fp in fps:
        color = TIER_COLORS[tiers.index(fp.threshold_in)]
        folium.GeoJson(
            fp.to_feature(),
            style_function=lambda _f, c=color: {
                "color": c,
                "weight": st.footprint_stroke_px,
                "fillOpacity": 0.0,
            },
            tooltip=f"{fp.label}: hail &ge; {fp.threshold_in:.2f} in, {fp.area_km2:.0f} km&sup2;",
        ).add_to(group)
    group.add_to(m)


def _png_data_url(img: RgbaImage) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(img)).decode("ascii")


def _add_hail_layers(m: folium.Map, result: RunResult, st: MapStyle) -> None:
    b = result.stack.grid_max.bounds
    folium.raster_layers.ImageOverlay(
        image=_png_data_url(
            to_web_mercator(colorize_hail(result.stack.grid_max), b.south, b.north)
        ),
        bounds=[[b.south, b.west], [b.north, b.east]],
        mercator_project=False,
        name="Max-ever hail size (MESH)",
    ).add_to(m)
    _add_footprints(m, result, result.max_ever_footprints, "Footprints: max ever", True, st)
    for ev in result.events:
        days = " to ".join(sorted({ev.days[0], ev.days[-1]}))
        fps = result.event_footprints[ev.event_id]
        _add_footprints(m, result, fps, f"Footprints: {days}", False, st)


def _add_property_layers(m: folium.Map, result: RunResult, st: MapStyle) -> None:
    lead_ids = {ld.property.property_id for ld in result.leads}
    others = [
        _point_feature(
            p.lat,
            p.lon,
            {
                "address": html.escape(p.address or p.property_id),
                "status": html.escape(result.scores[p.property_id].status),
            },
        )
        for p in result.properties
        if p.property_id not in lead_ids
    ]
    # folium's tooltip rejects a layer with no features, so empty layers are skipped.
    if others:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": others},
            name="Other properties",
            marker=folium.CircleMarker(),
            style_function=lambda _f: _circle_style(NEUTRAL_POINT, st.other_radius_px, st),
            tooltip=folium.GeoJsonTooltip(fields=["address", "status"], aliases=["", "Status"]),
        ).add_to(m)
    if not result.leads:
        return
    # Best leads are drawn last so they sit on top.
    leads = [
        _point_feature(ld.property.lat, ld.property.lon, _lead_props(ld)) for ld in result.leads
    ]
    folium.GeoJson(
        {"type": "FeatureCollection", "features": leads[::-1]},
        name="Leads (colour = score)",
        marker=folium.CircleMarker(),
        style_function=lambda f: _circle_style(f["properties"]["fill"], st.lead_radius_px, st),
        tooltip=folium.GeoJsonTooltip(
            fields=["rank", "address", "score"], aliases=["Rank", "", "Score"]
        ),
        popup=folium.GeoJsonPopup(
            fields=["address", "score", "breakdown", "hail_history", "roof", "channel", "provider"],
            aliases=["", "Score", "Points", "Hail on current roof", "Roof", "Channel", "Provider"],
        ),
    ).add_to(m)


def _add_provider_layer(m: folium.Map, result: RunResult, st: MapStyle) -> None:
    group = folium.FeatureGroup(name="Providers and service areas", show=False)
    for sp in result.providers:
        p = sp.provider
        rating = f"shrunk {sp.shrunk_rating:.2f} ({p.review_count} reviews)"
        folium.Circle(
            location=(p.lat, p.lon),
            radius=p.service_radius_km * METERS_PER_KM,
            color=PROVIDER_COLOR,
            weight=1,
            dash_array="4 4",
            fill=False,
        ).add_to(group)
        folium.CircleMarker(
            location=(p.lat, p.lon),
            radius=st.provider_radius_px,
            color=SURFACE_RING,
            weight=2,
            fill=True,
            fill_color=PROVIDER_COLOR,
            fill_opacity=1.0,
            tooltip=f"#{sp.rank} {html.escape(p.name)}: {rating}",
        ).add_to(group)
    group.add_to(m)


def render_map(result: RunResult, path: Path, style: MapStyle | None = None) -> None:
    """Write a self-contained Leaflet map with toggleable layers and a summary panel.

    Layers: all-time MESH raster, all-time and per-event footprint outlines, leads
    (coloured by score bin), other properties, and providers with service radii.
    There is no basemap, so no synthetic point is drawn over real land, and the
    page makes no tile requests.

    Args:
        result: Pipeline result.
        path: Output ``.html`` file.
        style: Layout constants; defaults to :class:`MapStyle`.

    Raises:
        TypeError: If folium does not root the map in a Figure.
    """
    st = style or MapStyle()
    lats = [p.lat for p in result.properties]
    lons = [p.lon for p in result.properties]
    b = result.stack.grid_max.bounds
    center = (float(np.mean(lats)), float(np.mean(lons))) if lats else b.center
    m = folium.Map(location=center, zoom_start=st.zoom_start, tiles=None, control_scale=True)
    _add_hail_layers(m, result, st)
    _add_property_layers(m, result, st)
    _add_provider_layer(m, result, st)
    folium.LayerControl(collapsed=True).add_to(m)

    root = m.get_root()
    if not isinstance(root, folium.Figure):
        raise TypeError(f"unexpected folium root {type(root).__name__}")
    title = "Storm lead map (synthetic demo)" if result.synthetic else "Storm lead map"
    root.header.add_child(folium.Element(f"<title>{html.escape(title)}</title>"))
    root.html.add_child(folium.Element(_panel_html(result, st)))
    _write_text(path, stable_ids(root.render()))


def write_outputs(
    result: RunResult, out_dir: Path, prefix: str = "", top: int | None = None
) -> dict[str, Path]:
    """Write every output file into ``out_dir``.

    Args:
        result: Pipeline result.
        out_dir: Output directory (created if needed).
        prefix: File-name prefix, e.g. ``"synthetic_"``.
        top: Leads to keep in the CSV; None keeps all.

    Returns:
        Output kind to file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "leads": out_dir / f"{prefix}leads{f'_top{top}' if top else ''}.csv",
        "providers": out_dir / f"{prefix}providers_ranked.csv",
        "footprints": out_dir / f"{prefix}swath_footprints.geojson",
        "summary": out_dir / f"{prefix}run_summary.json",
        "map": out_dir / f"{prefix}storm_map.html",
    }
    write_leads_csv(result, paths["leads"], top)
    write_ranked_providers_csv(result, paths["providers"])
    write_footprints_geojson(result, paths["footprints"])
    write_summary_json(result, paths["summary"])
    render_map(result, paths["map"])
    return paths
