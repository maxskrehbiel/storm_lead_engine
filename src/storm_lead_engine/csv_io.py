"""Shared CSV handling: header checks, per-row validation that reports every bad row, LF output."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from .errors import InputDataError

T = TypeVar("T")

MAX_ERRORS_SHOWN = 10
_TRUE = frozenset({"true", "t", "yes", "y", "1"})
_FALSE = frozenset({"false", "f", "no", "n", "0"})


def opt_int(value: str) -> int | None:
    """Parse an optional integer; blank means None."""
    value = value.strip()
    return int(float(value)) if value else None


def opt_float(value: str) -> float | None:
    """Parse an optional float; blank means None."""
    value = value.strip()
    return float(value) if value else None


def opt_bool(value: str) -> bool | None:
    """Parse an optional boolean (true/false, yes/no, y/n, 1/0); blank means None.

    Raises:
        ValueError: If the value is not a recognised boolean.
    """
    value = value.strip().lower()
    if not value:
        return None
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def lat_lon(row: Mapping[str, str]) -> tuple[float, float]:
    """Read and range-check the ``lat`` and ``lon`` columns.

    Raises:
        ValueError: If either value is missing, non-numeric or out of range.
    """
    lat, lon = float(row["lat"]), float(row["lon"])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"lat/lon out of range: {lat}, {lon}")
    return lat, lon


def _validate_rows(
    reader: csv.DictReader[str],
    required: tuple[str, ...],
    parse: Callable[[dict[str, str]], T],
    record_id: Callable[[T], str],
    path: str | Path,
) -> tuple[list[T], list[str]]:
    header = [h.strip() for h in (reader.fieldnames or [])]
    missing = [c for c in required if c not in header]
    if missing:
        raise InputDataError(f"{path}: missing required column(s): {', '.join(missing)}")
    records: list[T] = []
    errors: list[str] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(reader, start=2):
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        try:
            record = parse(row)
        except (ValueError, KeyError) as exc:
            errors.append(f"line {lineno}: {exc}")
            continue
        rid = record_id(record)
        if not rid:
            errors.append(f"line {lineno}: empty id")
        elif rid in seen:
            errors.append(f"line {lineno}: duplicate id {rid!r}")
        else:
            seen.add(rid)
            records.append(record)
    return records, errors


def read_records(
    path: str | Path,
    required: tuple[str, ...],
    parse: Callable[[dict[str, str]], T],
    record_id: Callable[[T], str],
) -> list[T]:
    """Read a CSV, validate every row, and fail once with every problem listed.

    Args:
        path: CSV file.
        required: Columns that must be present in the header.
        parse: Builds a record from one row; raises ValueError or KeyError on bad input.
        record_id: Returns a record's identifier, which must be non-empty and unique.

    Returns:
        Records in file order.

    Raises:
        InputDataError: If the file cannot be read, a required column is missing or any
            row is invalid.
    """
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as f:
            records, errors = _validate_rows(csv.DictReader(f), required, parse, record_id, path)
    except OSError as exc:
        raise InputDataError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise InputDataError(f"{path} is not UTF-8 text: {exc.reason}") from exc
    if errors:
        shown = "\n  ".join(errors[:MAX_ERRORS_SHOWN])
        extra = len(errors) - MAX_ERRORS_SHOWN
        more = f"\n  ... and {extra} more" if extra > 0 else ""
        raise InputDataError(f"{path}: {len(errors)} invalid row(s):\n  {shown}{more}")
    return records


def write_records(
    rows: Iterable[Mapping[str, Any]], path: str | Path, columns: tuple[str, ...]
) -> int:
    """Write rows with LF line endings; None becomes an empty cell.

    Args:
        rows: One mapping per row.
        path: Output file (parent directories are created).
        columns: Column order.

    Returns:
        Number of rows written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row[c]) for c in columns})
            n += 1
    return n
