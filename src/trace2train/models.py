"""Trace2Train data models.

These are the contract every pipeline stage (ingest -> detect -> clean ->
generate -> export) operates on. Kept minimal and pragmatic: we model enough
of a LangSmith trace to do the job, not the entire schema.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Ingestion layer
# ---------------------------------------------------------------------------


class RunType(str, Enum):
    """The type of a single run inside a trace."""

    LLM = "llm"
    CHAIN = "chain"
    TOOL = "tool"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"
    AGENT = "agent"
    UNKNOWN = "unknown"


class Run(BaseModel):
    """A single node in the execution tree of an agent run."""

    id: str
    name: str = ""
    run_type: RunType = RunType.UNKNOWN
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    parent_run_id: str | None = None
    child_runs: list[Run] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    # raw keeps the untouched source line for provenance / debugging
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Best-effort extraction of the message list from inputs/outputs.

        Handles LangSmith's typical shapes: {'messages': [...]},
        {'input': 'str'}, {'output': 'str'}, and tool/retriever documents.
        """
        msgs: list[dict[str, Any]] = []

        def _collect(container: dict[str, Any]) -> None:
            for key in ("messages", "input", "output", "prompt", "completion"):
                val = container.get(key)
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and ("role" in item or "type" in item):
                            msgs.append(item)
                elif isinstance(val, str) and val.strip():
                    role = "assistant" if key in ("output", "completion") else "user"
                    msgs.append({"role": role, "content": val})

        _collect(self.inputs)
        _collect(self.outputs)
        return msgs


class Trace(BaseModel):
    """A full agent session = the root run plus its nested execution tree."""

    trace_id: str
    root: Run
    runs: list[Run] = Field(default_factory=list)
    session_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_file: str | None = None
    source_format: str = "unknown"  # "langsmith" | "langfuse" | "messages" | "unknown"

    @property
    def failed(self) -> bool:
        """A trace is failed if any run in it errored."""
        return any(not r.succeeded for r in self.runs)

    def find_errors(self) -> list[Run]:
        return [r for r in self.runs if not r.succeeded]


# ---------------------------------------------------------------------------
# Clean / quality layer
# ---------------------------------------------------------------------------


class CleanIssue(str, Enum):
    PII = "pii"
    DUPLICATE = "duplicate"
    LEAK = "leak"
    EMPTY = "empty"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"


class CleanedText(BaseModel):
    text: str
    issues: list[CleanIssue] = Field(default_factory=list)
    fingerprint: str = ""


# ---------------------------------------------------------------------------
# Generation / export layer
# ---------------------------------------------------------------------------


class Role(str, Enum):
    SYSTEM = "system"
    HUMAN = "human"
    ASSISTANT = "assistant"


class ConversationTurn(BaseModel):
    """One turn in the exported conversation."""

    model_config = ConfigDict(populate_by_name=True)

    from_: Role = Field(alias="from")
    value: str


class Provenance(BaseModel):
    """Where this training sample came from, for auditing and dedup."""

    source_file: str | None = None
    trace_id: str = ""
    run_id: str = ""
    original_error: str | None = None
    attribution: str | None = None
    generated_by: str = "trace2train"


class SFTRecord(BaseModel):
    """A supervised fine-tuning example, LLaMA-Factory ChatML compatible."""

    conversations: list[ConversationTurn]
    provenance: Provenance = Field(default_factory=Provenance)

    def to_llama_factory(self) -> dict[str, Any]:
        """Serialize to the LLaMA-Factory / ShareGPT JSON shape.

        Returns {'conversations': [{'from': 'human'|'gpt'|'system', 'value': ...}]}
        with an extra _provenance key stripped from the core record.
        """
        role_map = {Role.SYSTEM: "system", Role.HUMAN: "human", Role.ASSISTANT: "gpt"}
        convos = [
            {"from": role_map[t.from_], "value": t.value} for t in self.conversations
        ]
        return {"conversations": convos}


class DPORecord(BaseModel):
    """A preference (DPO) example: chosen vs rejected assistant turn."""

    conversations: list[ConversationTurn]
    chosen: str
    rejected: str
    provenance: Provenance = Field(default_factory=Provenance)

    def to_llama_factory(self) -> dict[str, Any]:
        role_map = {Role.SYSTEM: "system", Role.HUMAN: "human", Role.ASSISTANT: "gpt"}
        convos = [
            {"from": role_map[t.from_], "value": t.value} for t in self.conversations
        ]
        return {
            "conversations": convos,
            "chosen": {"from": "gpt", "value": self.chosen},
            "rejected": {"from": "gpt", "value": self.rejected},
        }


# ---------------------------------------------------------------------------
# Inspect layer (the distribution hook: instant, rules-only, no LLM)
# ---------------------------------------------------------------------------


class DirtyBreakdown(BaseModel):
    """Why traces are 'dirty' — the numbers that make a shareable headline."""

    pii: int = 0
    duplicate: int = 0
    empty: int = 0
    env_noise: int = 0

    @property
    def total(self) -> int:
        return self.pii + self.duplicate + self.empty + self.env_noise


class InspectReport(BaseModel):
    """Instant quality report produced by `trace2train inspect`.

    Pure rules, no LLM — so it is free, offline, and fast. This report (not the
    JSONL) is the product's viral hook: a screenshot-worthy "your traces are
    X% garbage, Y are trainable" summary.
    """

    total_traces: int = 0
    failed: int = 0
    env_only: int = 0
    trainable: int = 0
    dirty: DirtyBreakdown = Field(default_factory=DirtyBreakdown)
    est_sft: int = 0
    est_dpo: int = 0
    source_format: str = "unknown"
    # tool-call / behavioral failure sub-type → count (trainable failures only)
    failure_types: dict[str, int] = Field(default_factory=dict)

    @property
    def dirty_pct(self) -> int:
        """Percent of failed traces that are dirty (rounded, for the headline)."""
        if self.failed == 0:
            return 0
        return round(100 * self.dirty.total / self.failed)

    @property
    def headline(self) -> str:
        """One screenshot-worthy line summarizing the report."""
        return self.headline_for(unicode_ok=True)

    def headline_for(self, *, unicode_ok: bool) -> str:
        """Headline with a unicode or ASCII arrow depending on console support."""
        arrow = "\u2192" if unicode_ok else "->"
        return (
            f"{self.total_traces:,} traces {arrow} {self.failed:,} failed "
            f"{arrow} {self.dirty_pct}% dirty (PII/dupes/noise) "
            f"{arrow} {self.trainable:,} trainable"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def fingerprint_text(text: str) -> str:
    """Normalized fingerprint for dedup: lowercase, strip whitespace/punct."""
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()
