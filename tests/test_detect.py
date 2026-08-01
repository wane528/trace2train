"""Failure-detection tests — including the empty-output regression."""

from __future__ import annotations

from trace2train.detect import detect_failures
from trace2train.importers import load
from trace2train.models import Run, RunType, Trace


def _trace(run_type, outputs, error=None):
    run = Run(id="x", name="n", run_type=run_type, outputs=outputs, error=error)
    return Trace(trace_id="x", root=run, runs=[run])


def test_success_not_flagged():
    """Regression: a run with real output must NOT be 'empty output'."""
    t = _trace(RunType.CHAIN, {"output": "The capital of France is Paris."})
    (d,) = detect_failures([t])
    assert not d.failed


def test_error_is_behavioral_failure():
    t = _trace(RunType.TOOL, {}, error="ToolError: wrong tool")
    (d,) = detect_failures([t])
    assert d.failed and d.trainable and not d.env_only


def test_env_error_not_trainable():
    t = _trace(RunType.CHAIN, {}, error="TimeoutError: timed out after 30s")
    (d,) = detect_failures([t])
    assert d.failed and d.env_only and not d.trainable


def test_empty_output_is_silent_failure():
    t = _trace(RunType.LLM, {"output": ""})
    (d,) = detect_failures([t])
    assert d.failed and d.trainable


def test_on_fixture(langsmith_file):
    dets = detect_failures(load(langsmith_file))
    failed = [d for d in dets if d.failed]
    trainable = [d for d in dets if d.trainable]
    assert len(failed) == 2       # t2 (tool err) + t3 (timeout)
    assert len(trainable) == 1    # only t2 is behavioral
