"""Inspect: the instant, rules-only quality report (the distribution hook).

`inspect` answers "how dirty are my traces, and how many are trainable?"
WITHOUT calling an LLM — so it is free, offline, and fast. The resulting
`InspectReport` (headline + breakdown) is the screenshot-worthy artifact that
drives adoption, before the user ever spends a cent on `convert`.

It reuses the same detection rules as `convert` (via detect.py) plus the same
dirtiness checks as clean.py, so the numbers it reports match what convert
will actually produce.
"""

from __future__ import annotations

from .clean import has_pii
from .detect import DetectionResult, detect_failures
from .models import DirtyBreakdown, InspectReport, Trace, fingerprint_text


def _trace_text(trace: Trace) -> str:
    """Concatenate all message content in a trace for dirtiness checks."""
    parts: list[str] = []
    for run in trace.runs:
        for msg in run.messages:
            content = msg.get("content") or msg.get("text") or ""
            if isinstance(content, str):
                parts.append(content)
    return " ".join(parts)


def build_report(
    traces: list[Trace],
    detections: list[DetectionResult] | None = None,
) -> InspectReport:
    """Compute the instant quality report. Pure rules, no LLM."""
    if detections is None:
        detections = detect_failures(traces)

    total = len(traces)
    failed = sum(1 for d in detections if d.failed)
    env_only = sum(1 for d in detections if d.failed and d.env_only)
    trainable = sum(1 for d in detections if d.trainable)

    dirty = DirtyBreakdown()
    seen: set[str] = set()
    failure_types: dict[str, int] = {}

    for det in detections:
        if not det.failed:
            continue
        if det.env_only:
            dirty.env_noise += 1
            # env-noise traces are not further inspected for pii/dup below
            continue

        # tally the tool-call / behavioral failure sub-type (trainable only)
        failure_types[det.failure_type] = failure_types.get(det.failure_type, 0) + 1

        text = _trace_text(det.trace)
        stripped = text.strip()

        if not stripped:
            dirty.empty += 1
            continue

        fp = fingerprint_text(stripped)
        if fp in seen:
            dirty.duplicate += 1
        else:
            seen.add(fp)
        if has_pii(stripped):
            dirty.pii += 1

    source_format = traces[0].source_format if traces else "unknown"

    return InspectReport(
        total_traces=total,
        failed=failed,
        env_only=env_only,
        trainable=trainable,
        dirty=dirty,
        # conservative estimates: one SFT per trainable trace; DPO when a
        # distinct failed answer is recoverable (~half, refined at convert time)
        est_sft=trainable,
        est_dpo=trainable // 2,
        source_format=source_format,
        failure_types=dict(sorted(failure_types.items(), key=lambda kv: -kv[1])),
    )
