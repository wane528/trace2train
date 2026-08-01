"""Training-sample generation.

Turns a *failed* trace into:
  - an SFT candidate  (conversation ending with a corrected assistant turn)
  - a DPO pair        (chosen = corrected answer, rejected = the failed answer)

Correction is done by the LLM given the trace context, so we capture the
"right way to do it" that the failed run didn't produce. The generated
records carry provenance (trace id, run id, original error, attribution)
for auditability and dedup downstream.
"""

from __future__ import annotations

from .attribute import Attribution
from .llm import LLMClient
from .models import (
    ConversationTurn,
    DPORecord,
    Provenance,
    Role,
    SFTRecord,
    Trace,
)

GENERATE_SYSTEM = (
    "You convert a failed AI agent trace into corrected training data — but ONLY "
    "when the correct answer can be reliably derived from the trace itself.\n"
    "Reply ONLY with a JSON object of this exact shape:\n"
    '{"category": "<one of>", '
    '"summary": "<one sentence, <140 chars, why it failed>", '
    '"correctable": true|false, '
    '"instruction": "<the user request being attempted, verbatim>", '
    '"correct_answer": "<the response the model SHOULD have given>", '
    '"failed_answer": "<what the model actually did wrong, or empty string>"}\n'
    "\n"
    "category must be exactly one of: wrong_tool, bad_args, lost_context, "
    "hallucination, refusal, env_error, unknown. Pick the one that best names "
    "the failure. summary is a short human-readable reason.\n"
    "\n"
    "CRITICAL — set correctable=false (and leave the other fields empty) when the "
    "correct answer CANNOT be determined from the trace alone, e.g.:\n"
    "- the task needs external verification (did code pass tests? is a fact "
    "accurate? did a file get written correctly?) and the trace does not contain "
    "that ground truth;\n"
    "- the agent CLAIMS success but you cannot confirm it from the trace (never "
    "treat an unverified success claim as the correct answer);\n"
    "- the failure is a transient/environment issue the agent recovered from;\n"
    "- the trace is too ambiguous to know what 'correct' means.\n"
    "\n"
    "Set correctable=true ONLY for behavioral mistakes whose fix is obvious from "
    "context: wrong tool chosen, malformed arguments, lost conversation context, "
    "wrong output format, a plainly wrong factual/common-sense answer, or an "
    "over-refusal of a benign request.\n"
    "\n"
    "Rules when correctable=true:\n"
    "- instruction: the user's request, verbatim if present.\n"
    "- correct_answer: the response that would have SUCCEEDED — grounded in the "
    "trace, never invented, never an unverified success claim.\n"
    "- failed_answer: the actual bad output (<=2000 chars), or \"\".\n"
    "When in doubt, prefer correctable=false. Producing NO data is better than "
    "producing misleading data."
)


_VALID_CATEGORIES = frozenset(
    {"wrong_tool", "bad_args", "lost_context", "hallucination", "refusal", "env_error", "unknown"}
)


def _valid_category(category: str) -> str:
    """Clamp an LLM-returned category to the known set, defaulting to 'unknown'."""
    return category if category in _VALID_CATEGORIES else "unknown"


def _dedupe_consecutive(turns: list[ConversationTurn]) -> list[ConversationTurn]:
    """Collapse identical back-to-back turns (same role + value).

    Raw traces often repeat the final assistant turn (e.g. the model's answer
    is echoed as both the step output and the run output). Handing a human two
    identical lines is noise, so we drop the immediate repeat.
    """
    deduped: list[ConversationTurn] = []
    for turn in turns:
        if deduped and deduped[-1].from_ == turn.from_ and deduped[-1].value == turn.value:
            continue
        deduped.append(turn)
    return deduped


def _last_user_message(trace: Trace) -> str:
    """Find the most recent user-side message across runs (best effort)."""
    for run in reversed(trace.runs):
        for msg in reversed(run.messages):
            role = msg.get("role")
            if role in ("user", "human", "tool"):  # tool=tool result, still context
                content = str(msg.get("content") or msg.get("text") or "").strip()
                if content:
                    return content[:4000]
    return ""


def _failed_answer(trace: Trace) -> str:
    """Recover the failed output: the erroring run's output or error text."""
    for run in trace.runs:
        if not run.succeeded:
            out = run.outputs or {}
            for key in ("output", "completion", "text"):
                val = out.get(key)
                if isinstance(val, str) and val.strip():
                    return val[:2000]
            if run.error:
                return f"[error] {run.error[:2000]}"
    return ""


def generate_records(
    trace: Trace,
    attribution: Attribution | None,
    client: LLMClient | None,
    *,
    force_sft_only: bool = False,
) -> tuple[list[SFTRecord], list[DPORecord]]:
    """Generate SFT/DPO records for one failed trace.

    With an LLM this makes a SINGLE call that both attributes the failure
    (category + summary) AND produces the corrected answer — halving the LLM
    cost vs. a separate attribution pass. `attribution`, when provided, is only
    used as a fallback label; the LLM's own category wins.

    If no LLM client is provided (or it fails), we emit an SFT record that
    preserves the trace verbatim so the user can hand-fix it later — better
    than silently dropping data.
    """
    fallback_attr = str(attribution) if attribution is not None else "unknown"
    prov = Provenance(
        source_file=trace.source_file,
        trace_id=trace.trace_id,
        run_id=trace.root.id,
        original_error="; ".join(r.error for r in trace.find_errors() if r.error) or None,
        attribution=fallback_attr,
    )

    sft_records: list[SFTRecord] = []
    dpo_records: list[DPORecord] = []

    if client is None or not client.configured:
        # No LLM: we CANNOT derive the corrected answer, so this is NOT
        # trainable data. We emit the raw conversation for human review only
        # (the caller routes it to needs_review/, never to train_sft.jsonl),
        # dropping the failed/erroring turns and collapsing repeats so the
        # human isn't handed the wrong answer as if it were correct.
        turns: list[ConversationTurn] = []
        for run in trace.runs:
            # Skip the output of a run that errored: that IS the failed answer.
            skip_assistant = not run.succeeded
            for msg in run.messages:
                role = msg.get("role")
                content = str(msg.get("content") or "").strip()
                if not content:
                    continue
                if role in ("user", "human", "system"):
                    mapped = Role.HUMAN if role != "system" else Role.SYSTEM
                    turns.append(ConversationTurn(from_=mapped, value=content))
                elif role in ("assistant", "ai"):
                    if skip_assistant:
                        continue
                    turns.append(ConversationTurn(from_=Role.ASSISTANT, value=content))
        turns = _dedupe_consecutive(turns)
        if turns:
            prov.attribution = (
                f"{prov.attribution} | offline: raw trace for human review, "
                "NOT a verified correction"
            )
            sft_records.append(SFTRecord(conversations=turns, provenance=prov))
        return sft_records, dpo_records

    # LLM path — single call attributes the failure AND derives the correction.
    try:
        data = client.complete_json(
            GENERATE_SYSTEM,
            _serialize_trace(trace),
            max_tokens=2000,
        )
        correctable = bool(data.get("correctable", False))
        instruction = str(data.get("instruction") or "").strip()
        correct = str(data.get("correct_answer") or "").strip()
        failed = str(data.get("failed_answer") or "").strip()
        category = _valid_category(str(data.get("category") or "unknown"))
        summary = str(data.get("summary") or "").strip()[:140]
        # The single call is also the attribution: record it in provenance.
        prov.attribution = str(Attribution(category, summary, correctable))
    except Exception as exc:  # noqa: BLE001
        correctable, instruction, correct, failed = False, "", "", ""
        prov.attribution = f"{prov.attribution} | generate llm error: {exc}"

    # Guardrail: only emit data when the correct answer is derivable from the
    # trace. correctable=false means the failure needs external ground truth
    # (code tests, fact-checks, unverified success claims) — skip it rather than
    # produce misleading training data.
    if not correctable or not instruction or not correct:
        return sft_records, dpo_records  # skipped_uncertain (logged by caller)

    # SFT: instruction + corrected answer
    sft_records.append(
        SFTRecord(
            conversations=[
                ConversationTurn(from_=Role.HUMAN, value=instruction),
                ConversationTurn(from_=Role.ASSISTANT, value=correct),
            ],
            provenance=prov,
        )
    )

    # DPO: only when we actually recovered a distinct failed answer
    if failed and failed != correct and not force_sft_only:
        dpo_records.append(
            DPORecord(
                conversations=[
                    ConversationTurn(from_=Role.HUMAN, value=instruction),
                ],
                chosen=correct,
                rejected=failed,
                provenance=prov,
            )
        )

    return sft_records, dpo_records


_MAX_MSG_CHARS = 800
_MAX_TOTAL_CHARS = 10_000


def _serialize_trace(trace: Trace) -> str:
    """Render the trace compactly and *well-formed* for the generator LLM.

    Long real trajectories can be 150k+ chars — dumping raw JSON and truncating
    breaks the structure and the LLM returns nothing. Instead we extract a
    readable transcript: each turn's role + truncated content, with the failed
    step marked, and a global size budget. This keeps the prompt coherent even
    for very long multi-turn agent sessions.
    """
    lines: list[str] = [f"# Failed agent trace ({trace.trace_id})"]

    # Collect the conversation turns across runs, truncating each message.
    for run in trace.runs:
        if not run.succeeded and run.error:
            lines.append(f"[STEP ERROR in {run.name}] {run.error[:300]}")
        for msg in run.messages:
            role = msg.get("role") or msg.get("from") or "?"
            content = msg.get("content") or msg.get("text") or ""
            if not isinstance(content, str):
                content = str(content)
            content = content.strip()
            if not content:
                continue
            if len(content) > _MAX_MSG_CHARS:
                half = _MAX_MSG_CHARS // 2
                content = content[:half] + " …[truncated]… " + content[-half:]
            lines.append(f"{role}: {content}")

    text = "\n".join(lines)
    if len(text) <= _MAX_TOTAL_CHARS:
        return text
    # Keep the head (task setup) and tail (where the failure happened).
    head = text[: _MAX_TOTAL_CHARS // 2]
    tail = text[-_MAX_TOTAL_CHARS // 2 :]
    return head + "\n…[middle of trajectory omitted]…\n" + tail
