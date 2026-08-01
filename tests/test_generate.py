"""Generate tests — offline path (no LLM) must preserve data, not drop it."""

from __future__ import annotations

from trace2train.attribute import Attribution
from trace2train.generate import generate_records
from trace2train.models import Run, RunType, Trace


def _failed_trace():
    run = Run(
        id="r1", name="agent", run_type=RunType.CHAIN,
        inputs={"messages": [{"role": "user", "content": "remember my name is Alex"}]},
        outputs={"output": "I don't know your name."},
        error="ContextError: lost context",
    )
    return Trace(trace_id="t1", root=run, runs=[run], source_file="x.jsonl")


def test_offline_passthrough_emits_sft():
    """Without an LLM, we still emit an SFT record (never silently drop)."""
    trace = _failed_trace()
    attr = Attribution("unknown", "offline", False)
    sfts, dpos = generate_records(trace, attr, client=None)
    assert len(sfts) == 1
    assert dpos == []
    # provenance carries the original error
    assert "ContextError" in (sfts[0].provenance.original_error or "")


def test_offline_record_has_turns():
    trace = _failed_trace()
    sfts, _ = generate_records(trace, Attribution("unknown", "", False), client=None)
    values = [t.value for t in sfts[0].conversations]
    assert any("Alex" in v for v in values)


class _FakeClient:
    """Stand-in LLM client returning a fixed JSON payload."""

    configured = True

    def __init__(self, payload: dict):
        self.payload = payload

    def complete_json(self, system, user, **kw):
        return self.payload


def test_guardrail_skips_when_not_correctable():
    """correctable=false must produce NO records (no misleading data)."""
    trace = _failed_trace()
    client = _FakeClient({
        "correctable": False,
        "instruction": "",
        "correct_answer": "",
        "failed_answer": "",
    })
    sfts, dpos = generate_records(trace, Attribution("hallucination", "", True), client)
    assert sfts == [] and dpos == []


def test_guardrail_skips_even_if_answer_present_but_not_correctable():
    """Even if the LLM fills an answer, correctable=false wins (safety-first)."""
    trace = _failed_trace()
    client = _FakeClient({
        "correctable": False,
        "instruction": "do X",
        "correct_answer": "all tests pass",  # unverified success claim
        "failed_answer": "err",
    })
    sfts, dpos = generate_records(trace, Attribution("unknown", "", True), client)
    assert sfts == [] and dpos == []


def test_correctable_emits_sft_and_dpo():
    """correctable=true with a distinct failed answer emits SFT + DPO."""
    trace = _failed_trace()
    client = _FakeClient({
        "correctable": True,
        "instruction": "remember my name is Alex; what is it?",
        "correct_answer": "Your name is Alex.",
        "failed_answer": "I don't know your name.",
    })
    sfts, dpos = generate_records(trace, Attribution("lost_context", "", True), client)
    assert len(sfts) == 1
    assert len(dpos) == 1
    assert dpos[0].chosen == "Your name is Alex."
    assert dpos[0].rejected == "I don't know your name."
