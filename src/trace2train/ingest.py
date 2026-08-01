"""Backward-compatible ingest shim.

The real importer logic now lives in `trace2train.importers`. This module is
kept so `load_langsmith_jsonl(...)` keeps working, and exposes the generic
`load(...)` entry point with auto-detection.
"""

from __future__ import annotations

from pathlib import Path

from .importers import detect_format, load
from .importers.langsmith import LangSmithImporter
from .models import Trace

__all__ = ["load", "detect_format", "load_langsmith_jsonl"]


def load_langsmith_jsonl(path: str | Path, max_traces: int | None = None) -> list[Trace]:
    """Load a LangSmith export JSONL (kept for backward compatibility)."""
    return LangSmithImporter().load(Path(path), max_traces=max_traces)
