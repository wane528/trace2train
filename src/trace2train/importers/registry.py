"""Importer registry + auto-detection.

Public entry point: `load(path, fmt="auto")`. Auto-detects the format by
sniffing the first JSONL record, then dispatches to the right importer.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import Trace
from .langfuse import LangfuseImporter
from .langsmith import LangSmithImporter
from .messages import MessagesImporter

# Registration order matters for sniffing: more specific formats first.
_IMPORTERS = [LangSmithImporter(), LangfuseImporter(), MessagesImporter()]
_BY_ID = {imp.format_id: imp for imp in _IMPORTERS}


def _first_record(path: Path) -> dict:
    """Read the first non-empty JSON line for format sniffing."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def detect_format(path: str | Path) -> str:
    """Sniff the input format. Returns a format_id or 'unknown'."""
    first = _first_record(Path(path))
    if not first:
        return "unknown"
    for imp in _IMPORTERS:
        if imp.sniff(first):
            return imp.format_id
    return "unknown"


def load(
    path: str | Path,
    fmt: str = "auto",
    max_traces: int | None = None,
) -> list[Trace]:
    """Load traces from a JSONL file, auto-detecting the format by default.

    Raises FileNotFoundError if the file is missing, and ValueError if the
    format cannot be recognized (or an explicit unknown fmt is passed).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    if fmt == "auto":
        fmt = detect_format(path)

    importer = _BY_ID.get(fmt)
    if importer is None:
        raise ValueError(
            f"Unknown/unsupported trace format: {fmt!r}. "
            f"Supported: {', '.join(_BY_ID)} (or omit --format for auto-detect)."
        )
    return importer.load(path, max_traces=max_traces)
