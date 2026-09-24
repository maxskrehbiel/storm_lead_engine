"""Exceptions raised by the package; ``cli.py`` maps each one to an exit code."""

from __future__ import annotations


class StormLeadError(Exception):
    """Base class for every error this package raises on purpose."""


class InputDataError(StormLeadError, ValueError):
    """An input file or value is missing, malformed or implausible (exit code 2)."""


class NoStormDataError(StormLeadError):
    """No usable storm day: nothing could be read or nothing reached the threshold (exit code 1)."""


class MissingDependencyError(StormLeadError, RuntimeError):
    """An optional dependency needed for the requested source is not installed (exit code 3)."""
