"""CLI subcommands, argument validation and exit codes."""

import runpy
import sys
from pathlib import Path

import pytest

from storm_lead_engine import cli
from storm_lead_engine.checks import CheckResult
from storm_lead_engine.errors import MissingDependencyError
from storm_lead_engine.fixture import load_season, season_day_grids

PROPS = "property_id,lat,lon,year_built\nA,0.0,0.0,1980\n"


def _props(tmp_path: Path) -> Path:
    path = tmp_path / "p.csv"
    path.write_text(PROPS, encoding="utf-8")
    return path


def test_demo_defaults_to_demo_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["demo", "--n-properties", "200", "--top", "10"]) == cli.EXIT_OK
    names = sorted(p.name for p in (tmp_path / "demo_output").iterdir())
    assert all(n.startswith("synthetic_") for n in names)
    assert "synthetic_leads_top10.csv" in names
    out = capsys.readouterr().out
    assert "8 events, 200 properties" in out
    assert out.count("check PASS") == 8 and "check FAIL" not in out


def test_synth_then_run(tmp_path: Path) -> None:
    assert cli.main(["synth", "--out", str(tmp_path), "--n-properties", "120"]) == 0
    props = tmp_path / "synthetic_properties.csv"
    provs = tmp_path / "synthetic_providers.csv"
    assert props.exists() and provs.exists()
    out = tmp_path / "run"
    args = ["run", "--properties", str(props), "--providers", str(provs), "--out", str(out)]
    assert cli.main([*args, "--as-of", "2025-06-24"]) == 0
    assert (out / "synthetic_leads.csv").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["demo", "--top", "-3"],
        ["demo", "--top", "0"],
        ["demo", "--n-properties", "abc"],
        ["demo", "--seed", "-1"],
        ["run", "--properties", "x.csv", "--bbox=1,2,3"],
        ["run", "--properties", "x.csv", "--dates", "2024-13-40"],
        ["run", "--properties", "x.csv", "--as-of", "tomorrow"],
        ["run", "--properties", "x.csv", "--dates", ","],
    ],
)
def test_bad_arguments_are_usage_errors(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == cli.EXIT_USAGE_ERROR


def test_argument_types() -> None:
    assert cli.positive_int("3") == 3
    assert cli.non_negative_int("0") == 0


def test_input_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("property_id,lat\nA,1\n", encoding="utf-8")
    assert cli.main(["run", "--properties", str(bad)]) == cli.EXIT_USAGE_ERROR
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "missing required column" in err
    assert "Traceback" not in err
    good = _props(tmp_path)
    assert cli.main(["run", "--properties", str(good), "--source", "mrms"]) == 2
    assert cli.main(["run", "--properties", str(tmp_path / "absent.csv")]) == 2
    assert cli.main(["demo", "--out", str(tmp_path), "--n-properties", "999999"]) == 2


def test_mrms_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    grids = season_day_grids(load_season())
    calls: list[tuple[object, object]] = []

    def fake_fetch(days: object, bbox: object, cache_dir: object) -> object:
        calls.append((days, bbox))
        return grids, ["2021-07-27"]

    monkeypatch.setattr(cli, "fetch_mrms_day_grids", fake_fetch)
    args = ["run", "--properties", str(_props(tmp_path)), "--source", "mrms"]
    args += ["--out", str(tmp_path / "o"), "--dates", "2021-05-18,2022-07-08", "--bbox=-1,-1,1,1"]
    assert cli.main(args) == 0
    assert (tmp_path / "o" / "leads.csv").exists()  # real-data runs are not prefixed
    days, _bbox = calls[0]
    assert isinstance(days, list) and len(days) == 2


def test_domain_and_dependency_exit_codes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = ["run", "--properties", str(_props(tmp_path)), "--source", "mrms"]
    base += ["--dates", "2024-08-25", "--bbox=-1,-1,1,1"]

    def missing_dep(*_a: object) -> None:
        raise MissingDependencyError("needs rasterio")

    monkeypatch.setattr(cli, "fetch_mrms_day_grids", missing_dep)
    assert cli.main(base) == cli.EXIT_MISSING_DEPENDENCY
    monkeypatch.setattr(cli, "fetch_mrms_day_grids", lambda *_a: ({}, ["2024-08-25"]))
    assert cli.main(base) == cli.EXIT_DOMAIN_CHECK_FAILED


def test_module_entry_point(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "m"
    monkeypatch.setattr(
        sys, "argv", ["storm_lead_engine", "demo", "--out", str(out), "--n-properties", "60"]
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("storm_lead_engine", run_name="__main__")
    assert exit_info.value.code == 0
    assert (out / "synthetic_storm_map.html").exists()


def test_failed_self_check_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli, "demo_checks", lambda *_a: [CheckResult("forced failure", False, "for the test")]
    )
    args = ["demo", "--out", str(tmp_path), "--n-properties", "60"]
    assert cli.main(args) == cli.EXIT_DOMAIN_CHECK_FAILED
    assert "check FAIL  forced failure: for the test" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flags", "level"), [([], "WARNING"), (["-v"], "INFO"), (["-vv"], "DEBUG")]
)
def test_verbosity_levels(
    flags: list[str], level: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, int] = {}
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kw: seen.update(level=kw["level"]))
    argv = [*flags, "synth", "--out", str(tmp_path), "--n-properties", "20"]
    assert cli.main(argv) == 0
    assert seen["level"] == getattr(cli.logging, level)
