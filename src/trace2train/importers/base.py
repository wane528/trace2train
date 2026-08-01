"""Importer protocol.

Every input format (LangSmith, Langfuse, generic messages, later OTel) is an
Importer: it knows how to recognize its own format and load a file into the
unified `Trace` model. New formats = one new file + one registry entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..models import Trace


@runtime_checkable
class Importer(Protocol):
    """A pluggable trace importer."""

    #: short format id, e.g. "langsmith" | "langfuse" | "messages"
    format_id: str

    def sniff(self, first_line: dict) -> bool:
        """Return True if the first JSONL record looks like this format."""
        ...

    def load(self, path: Path, max_traces: int | None = None) -> list[Trace]:
        """Parse the whole file into Trace objects."""
        ...
