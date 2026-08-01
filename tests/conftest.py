"""Shared pytest fixtures: sample trace files written without BOM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# A LangSmith-style export: one success, one behavioral failure, one env error.
LANGSMITH_ROWS = [
    {
        "id": "r1", "trace_id": "t1", "run_type": "chain", "name": "agent",
        "parent_run_id": None,
        "inputs": {"messages": [{"role": "user", "content": "What is 2+2?"}]},
        "outputs": {"output": "2+2 is 4."}, "error": None,
    },
    {
        "id": "r2", "trace_id": "t2", "run_type": "tool", "name": "agent",
        "parent_run_id": None,
        "inputs": {"messages": [{"role": "user", "content": "Weather in Tokyo?"}]},
        "outputs": {"output": "calculator says 42"},
        "error": "ToolError: wrong tool for weather",
    },
    {
        "id": "r3", "trace_id": "t3", "run_type": "chain", "name": "agent",
        "parent_run_id": None,
        "inputs": {"messages": [{"role": "user", "content": "Fetch news"}]},
        "outputs": {}, "error": "TimeoutError: timed out",
    },
]

# A generic messages-JSONL: one failure (marked), one success.
MESSAGES_ROWS = [
    {
        "messages": [
            {"role": "user", "content": "Email me at a@b.com"},
            {"role": "assistant", "content": "done"},
        ],
        "error": "delivery failed", "id": "m1",
    },
    {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
        "id": "m2",
    },
]


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    # Explicit UTF-8 without BOM (Windows-safe).
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def langsmith_file(tmp_path: Path) -> Path:
    return _write_jsonl(tmp_path / "langsmith.jsonl", LANGSMITH_ROWS)


@pytest.fixture
def messages_file(tmp_path: Path) -> Path:
    return _write_jsonl(tmp_path / "messages.jsonl", MESSAGES_ROWS)


@pytest.fixture
def langfuse_observations_file() -> Path:
    return Path(__file__).parent / "fixtures" / "langfuse_observations.jsonl"
