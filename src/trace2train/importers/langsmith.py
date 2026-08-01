"""LangSmith trace-export importer.

Reads LangSmith JSONL exports (each line = a run) and flattens nested run
trees into `Trace` objects, grouping by trace_id.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..models import Run, RunType, Trace


def _parse_run_type(raw: str | None) -> RunType:
    try:
        return RunType(raw or "")
    except ValueError:
        return RunType.UNKNOWN


def _parse_dt(val) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val)
    except (TypeError, ValueError):
        return None


def _walk_runs(node: dict, parent_id: str | None = None) -> list[Run]:
    """Recursively flatten a run + its child_runs into a Run list."""
    run = Run(
        id=str(node.get("id", "")),
        name=node.get("name", ""),
        run_type=_parse_run_type(node.get("run_type")),
        inputs=node.get("inputs") or {},
        outputs=node.get("outputs") or {},
        error=node.get("error"),
        parent_run_id=parent_id,
        start_time=_parse_dt(node.get("start_time")),
        end_time=_parse_dt(node.get("end_time")),
        extra=node.get("extra") or {},
        tags=node.get("tags") or [],
        raw=node,
    )
    runs = [run]
    for child in node.get("child_runs") or []:
        runs.extend(_walk_runs(child, parent_id=run.id))
    return runs


class LangSmithImporter:
    format_id = "langsmith"

    def sniff(self, first_line: dict) -> bool:
        return any(
            k in first_line for k in ("run_type", "child_runs", "dotted_order")
        )

    def load(self, path: Path, max_traces: int | None = None) -> list[Trace]:
        by_trace: dict[str, list[dict]] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    node = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines; keep going
                tid = node.get("trace_id") or str(
                    node.get("session_id") or node.get("id")
                )
                by_trace.setdefault(tid, []).append(node)

        traces: list[Trace] = []
        for tid, nodes in by_trace.items():
            if max_traces is not None and len(traces) >= max_traces:
                break
            root_node = next((n for n in nodes if not n.get("parent_run_id")), nodes[0])
            root = _walk_runs(root_node)[0]

            all_runs: list[Run] = []
            for node in nodes:
                if node.get("id") == root.id:
                    continue
                all_runs.extend(_walk_runs(node, parent_id=root.id))

            seen_ids: set[str] = set()
            deduped: list[Run] = []
            for r in [root, *all_runs]:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    deduped.append(r)

            traces.append(
                Trace(
                    trace_id=tid,
                    root=root,
                    runs=deduped,
                    session_id=root_node.get("session_id"),
                    source_file=str(path),
                    source_format=self.format_id,
                )
            )

        traces.sort(key=lambda t: t.start_time or datetime.min)
        return traces
