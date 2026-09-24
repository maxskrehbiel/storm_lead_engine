"""Command-line interface: the demo, synth and run subcommands and their exit codes."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import numpy as np

from . import __version__
from .checks import demo_checks
from .config import BBox
from .errors import InputDataError, MissingDependencyError, NoStormDataError
from .fixture import load_season, season_day_grids
from .grid import HailGrid
from .history import parse_day
from .ingest import fetch_mrms_day_grids
from .pipeline import RunResult, run_pipeline
from .properties import load_properties_csv, write_properties_csv
from .providers import Provider, load_providers_csv, write_providers_csv
from .report import write_outputs
from .synthetic import TownSpec, generate_properties, generate_providers

DEFAULT_SEED = 7
DEFAULT_N_PROPERTIES = 1500
DEFAULT_TOP = 50
DEFAULT_DEMO_DIR = Path("demo_output")
EXIT_OK = 0
EXIT_DOMAIN_CHECK_FAILED = 1
EXIT_USAGE_ERROR = 2
EXIT_MISSING_DEPENDENCY = 3


def positive_int(text: str) -> int:
    """argparse type: an integer >= 1."""
    value = _int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text!r}")
    return value


def non_negative_int(text: str) -> int:
    """argparse type: an integer >= 0."""
    value = _int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {text!r}")
    return value


def _int(text: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from exc


def _arg(parse: Callable[[str], object], what: str) -> Callable[[str], object]:
    def convert(text: str) -> object:
        try:
            return parse(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid {what} {text!r}: {exc}") from exc

    return convert


def _date_list(text: str) -> list[date]:
    days = [parse_day(part) for part in text.split(",") if part.strip()]
    if not days:
        raise ValueError("no dates given")
    return days


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="storm_lead_engine",
        description="Hail swaths (NOAA MRMS MESH) to ranked, explainable roof-inspection leads.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v for progress, -vv for debug detail"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run end to end on synthetic data (offline)")
    demo.add_argument("--out", type=Path, default=DEFAULT_DEMO_DIR, help="output directory")
    demo.add_argument("--seed", type=non_negative_int, default=DEFAULT_SEED)
    demo.add_argument("--n-properties", type=positive_int, default=DEFAULT_N_PROPERTIES)
    demo.add_argument("--top", type=positive_int, default=DEFAULT_TOP, help="leads in the CSV")
    demo.add_argument("--fixture", type=Path, default=None, help="storm-season fixture JSON")

    synth = sub.add_parser("synth", help="write synthetic property and provider CSVs")
    synth.add_argument("--out", type=Path, required=True, help="output directory")
    synth.add_argument("--seed", type=non_negative_int, default=DEFAULT_SEED)
    synth.add_argument("--n-properties", type=positive_int, default=DEFAULT_N_PROPERTIES)
    synth.add_argument("--fixture", type=Path, default=None, help="fixture that sets the town")

    run = sub.add_parser("run", help="score your own property file")
    run.add_argument("--properties", type=Path, required=True, help="property CSV")
    run.add_argument("--providers", type=Path, default=None, help="provider CSV (optional)")
    run.add_argument("--source", choices=("fixture", "mrms"), default="fixture")
    run.add_argument("--fixture", type=Path, default=None, help="storm fixture (source=fixture)")
    run.add_argument("--dates", type=_arg(_date_list, "dates"), help="YYYY-MM-DD,... (mrms)")
    run.add_argument("--bbox", type=_arg(BBox.parse, "bbox"), help="--bbox=W,S,E,N (mrms)")
    run.add_argument("--cache-dir", type=Path, default=Path(".cache/mrms"))
    run.add_argument("--as-of", type=_arg(parse_day, "date"), help="date roof age is measured")
    run.add_argument("--synthetic", action="store_true", help="label outputs as synthetic")
    run.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    run.add_argument("--top", type=positive_int, default=None, help="leads in the CSV (all)")
    return parser


def _report(result: RunResult, paths: dict[str, Path]) -> None:
    s = result.summary()
    counts = f"{len(result.events)} events, {s['properties']} properties, {s['leads']} leads"
    print(f"{s['source']}: {counts}")
    if result.missing_days:
        print(f"missing storm days (not read as 'no hail'): {', '.join(result.missing_days)}")
    for ld in result.leads[:5]:
        print(
            f"  #{ld.rank:<3} {ld.score.score:5.1f}  {ld.profile.n_severe} severe, "
            f"max {ld.profile.max_in:.2f} in  {ld.property.address or ld.property.property_id}"
        )
    for kind, path in paths.items():
        print(f"wrote {kind:<10} {path.as_posix()}")


def _check_capacity(n: int) -> None:
    limit = TownSpec().max_properties
    if n > limit:
        raise InputDataError(f"--n-properties must be at most {limit}, got {n}")


def _demo(args: argparse.Namespace) -> int:
    _check_capacity(args.n_properties)
    season = load_season(args.fixture)
    rng = np.random.default_rng(args.seed)
    props = generate_properties(args.n_properties, season.origin, season.as_of.year, rng)
    provs = generate_providers(season.origin, rng)
    grids = season_day_grids(season)
    result = run_pipeline(props, provs, grids, season.as_of, season.name, synthetic=True)
    _report(result, write_outputs(result, args.out, prefix="synthetic_", top=args.top))
    checks = demo_checks(result, season, grids)
    for check in checks:
        print(f"check {'PASS' if check.passed else 'FAIL'}  {check.name}: {check.detail}")
    return EXIT_OK if all(c.passed for c in checks) else EXIT_DOMAIN_CHECK_FAILED


def _synth(args: argparse.Namespace) -> int:
    _check_capacity(args.n_properties)
    season = load_season(args.fixture)
    rng = np.random.default_rng(args.seed)
    props = generate_properties(args.n_properties, season.origin, season.as_of.year, rng)
    provs = generate_providers(season.origin, rng)
    write_properties_csv(props, args.out / "synthetic_properties.csv")
    write_providers_csv(provs, args.out / "synthetic_providers.csv")
    print(f"wrote {len(props)} properties and {len(provs)} providers to {args.out.as_posix()}")
    return EXIT_OK


def _run(args: argparse.Namespace) -> int:
    props = load_properties_csv(args.properties)
    provs: list[Provider] = load_providers_csv(args.providers) if args.providers else []
    missing: list[str] = []
    grids: dict[str, HailGrid]
    if args.source == "fixture":
        season = load_season(args.fixture)
        grids, as_of, source = season_day_grids(season), season.as_of, season.name
    else:
        if not args.dates or not args.bbox:
            raise InputDataError("--source mrms needs --dates and --bbox")
        grids, missing = fetch_mrms_day_grids(args.dates, args.bbox, args.cache_dir)
        as_of, source = date.today(), "NOAA MRMS MESH_Max_1440min (public)"
    if not grids:
        raise NoStormDataError(f"no storm day could be read (missing: {', '.join(missing)})")
    synthetic = args.synthetic or args.source == "fixture"
    result = run_pipeline(
        props, provs, grids, args.as_of or as_of, source, synthetic, missing_days=missing
    )
    prefix = "synthetic_" if synthetic else ""
    _report(result, write_outputs(result, args.out, prefix=prefix, top=args.top))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run a subcommand.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 success, 1 a domain check failed (no usable storm data, or a demo
        self-check), 2 usage or input error, 3 missing optional dependency. argparse
        itself exits with 2 on bad arguments. Unexpected errors keep their traceback.
    """
    args = _build_parser().parse_args(argv)
    levels = {0: logging.WARNING, 1: logging.INFO}
    logging.basicConfig(
        level=levels.get(args.verbose, logging.DEBUG), format="%(levelname)s %(message)s"
    )
    handlers = {"demo": _demo, "synth": _synth, "run": _run}
    try:
        return handlers[args.command](args)
    except NoStormDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DOMAIN_CHECK_FAILED
    except MissingDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MISSING_DEPENDENCY
    except InputDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
