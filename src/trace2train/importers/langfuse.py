"""Langfuse observations JSONL importer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from ..models import Run, RunType, Trace

_TYPE_MAP: Final[dict[str, RunType]] = {
    "GENERATION": RunType.LLM,
    "AGENT": RunType.AGENT,
    "TOOL": RunType.TOOL,
    "CHAIN": RunType.CHAIN,
    "SPAN": RunType.CHAIN,
    "RETRIEVER": RunType.RETRIEVER,
    "EMBEDDING": RunType.EMBEDDING,
}
_SNIFF_DISCRIMINATORS: Final[frozenset[str]] = frozenset(
    {"type", "parentObservationId", "isRootObservation"}
)
_MIN_DT: Final[datetime] = datetime.min.replace(tzinfo=UTC)


def _parse_dt(value: str | None) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _normalize_payload(value: str | dict | list | None, key: str) -> dict:
    match value:
        case None:
            return {}
        case dict():
            return value
        case list():
            return {key: value}
        case str():
            if value == "":
                return {key: ""}
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {key: value}
            match parsed:
                case None:
                    return {}
                case dict():
                    return parsed
                case list() | str() | int() | float() | bool():
                    return {key: parsed}
                case _:
                    return {key: value}
        case _:
            return {}


def _map_run_type(raw_type: str | None) -> RunType:
    if raw_type is None:
        return RunType.UNKNOWN
    return _TYPE_MAP.get(raw_type, RunType.UNKNOWN)


def _build_error(level: str | None, status_message: str | None, run_id: str) -> str | None:
    if level != "ERROR":
        return None
    if status_message:
        return status_message
    return f"Langfuse observation {run_id} failed"


def _is_sniffable_row(row: dict[str, object]) -> bool:
    required = ("id", "traceId", "startTime")
    if not all(isinstance(row.get(key), str) and row.get(key) for key in required):
        return False
    return any(key in row for key in _SNIFF_DISCRIMINATORS)


def _has_usable_start_time(row: dict[str, object]) -> bool:
    start_time_raw = row.get("startTime")
    if not isinstance(start_time_raw, str) or start_time_raw == "":
        return False
    return _parse_dt(start_time_raw) is not None


def _row_sort_key(row: dict[str, object]) -> tuple[datetime, str]:
    start_time_raw = row.get("startTime")
    start_time = _parse_dt(start_time_raw if isinstance(start_time_raw, str) else None)
    return (start_time or _MIN_DT, str(row.get("id") or ""))


def _is_root_candidate(row: dict[str, object], row_ids: set[str]) -> bool:
    if row.get("isRootObservation") is True:
        return True
    if "parentObservationId" not in row:
        return True
    parent_id = row.get("parentObservationId")
    if parent_id is None:
        return True
    return isinstance(parent_id, str) and parent_id != "" and parent_id not in row_ids


def _root_row(rows: list[dict[str, object]], row_ids: set[str]) -> dict[str, object]:
    flagged_root = next(
        (row for row in rows if row.get("isRootObservation") is True),
        None,
    )
    if flagged_root is not None:
        return flagged_root
    return next((row for row in rows if _is_root_candidate(row, row_ids)), rows[0])


def _max_end_time(runs: list[Run]) -> datetime | None:
    end_times = [run.end_time for run in runs if run.end_time is not None]
    if not end_times:
        return None
    return max(end_times)


def _build_run(row: dict[str, object]) -> Run:
    run_id = row.get("id")
    name = row.get("name")
    parent_observation_id = row.get("parentObservationId")
    status_message = row.get("statusMessage")
    level = row.get("level")
    tags = row.get("tags")
    raw_type = row.get("type")
    input_value = row.get("input")
    output_value = row.get("output")
    start_time_raw = row.get("startTime")
    end_time_raw = row.get("endTime")

    return Run(
        id=run_id if isinstance(run_id, str) else "",
        name=name if isinstance(name, str) else "",
        run_type=_map_run_type(raw_type if isinstance(raw_type, str) else None),
        inputs=_normalize_payload(
            input_value
            if isinstance(input_value, (str, dict, list)) or input_value is None
            else None,
            "input",
        ),
        outputs=_normalize_payload(
            output_value
            if isinstance(output_value, (str, dict, list)) or output_value is None
            else None,
            "output",
        ),
        error=_build_error(
            level if isinstance(level, str) else None,
            status_message if isinstance(status_message, str) else None,
            run_id if isinstance(run_id, str) else "",
        ),
        start_time=_parse_dt(start_time_raw if isinstance(start_time_raw, str) else None),
        end_time=_parse_dt(end_time_raw if isinstance(end_time_raw, str) else None),
        parent_run_id=(
            parent_observation_id
            if isinstance(parent_observation_id, str)
            else None
        ),
        tags=[tag for tag in tags if isinstance(tag, str)] if isinstance(tags, list) else [],
        raw=row,
    )


class LangfuseImporter:
    format_id = "langfuse"

    def sniff(self, first_line: dict) -> bool:
        """Return True when the row looks like a Langfuse v2 observation export."""
        return _is_sniffable_row(first_line)

    def load(self, path: Path, max_traces: int | None = None) -> list[Trace]:
        """Parse Langfuse observation rows into grouped traces."""
        grouped_rows: dict[str, list[dict[str, object]]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = line.strip()
                if record == "":
                    continue
                try:
                    parsed = json.loads(record)
                except json.JSONDecodeError:
                    continue
                if not isinstance(parsed, dict) or not _is_sniffable_row(parsed):
                    continue
                if not _has_usable_start_time(parsed):
                    continue
                trace_id = parsed.get("traceId")
                if isinstance(trace_id, str):
                    grouped_rows.setdefault(trace_id, []).append(parsed)

        traces: list[Trace] = []
        for trace_id, rows in grouped_rows.items():
            non_empty_row_ids = [
                row_id
                for row in rows
                if isinstance((row_id := row.get("id")), str) and row_id != ""
            ]
            if len(non_empty_row_ids) != len(set(non_empty_row_ids)):
                continue
            rows.sort(key=_row_sort_key)
            row_ids = {str(row["id"]) for row in rows if isinstance(row.get("id"), str)}
            root_row = _root_row(rows, row_ids)
            runs = [_build_run(row) for row in rows]
            root_run = next(run for run in runs if run.id == root_row.get("id"))
            trace_start = next((run.start_time for run in runs if run.start_time is not None), None)
            root_session_id = root_row.get("sessionId")
            traces.append(
                Trace(
                    trace_id=trace_id,
                    root=root_run,
                    runs=runs,
                    session_id=root_session_id if isinstance(root_session_id, str) else None,
                    start_time=trace_start,
                    end_time=_max_end_time(runs),
                    source_file=str(path),
                    source_format=self.format_id,
                )
            )

        traces.sort(key=lambda trace: (trace.start_time or _MIN_DT, trace.trace_id))
        if max_traces is None:
            return traces
        return traces[:max_traces]
