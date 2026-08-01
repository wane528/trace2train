"""Importer + auto-detection tests."""

from __future__ import annotations

from trace2train.importers import detect_format, load


def test_detect_langsmith(langsmith_file):
    assert detect_format(langsmith_file) == "langsmith"


def test_detect_messages(messages_file):
    assert detect_format(messages_file) == "messages"


def test_detect_langfuse(langfuse_observations_file):
    assert detect_format(langfuse_observations_file) == "langfuse"


def test_load_langsmith_groups_by_trace(langsmith_file):
    traces = load(langsmith_file)
    assert len(traces) == 3
    ids = {t.trace_id for t in traces}
    assert ids == {"t1", "t2", "t3"}
    for t in traces:
        assert t.source_format == "langsmith"


def test_load_messages(messages_file):
    traces = load(messages_file)
    assert len(traces) == 2
    assert all(t.source_format == "messages" for t in traces)
    # first convo is marked failed via error
    failed = [t for t in traces if t.failed]
    assert len(failed) == 1


def test_load_auto_dispatches(langsmith_file, messages_file):
    assert load(langsmith_file)[0].source_format == "langsmith"
    assert load(messages_file)[0].source_format == "messages"


def test_load_auto_dispatches_langfuse(langfuse_observations_file):
    traces = load(langfuse_observations_file)
    assert traces[0].source_format == "langfuse"
    assert traces[0].trace_id == "trace-a"


def test_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.jsonl")


def test_malformed_lines_skipped(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(
        '{"messages":[{"role":"user","content":"ok"}]}\n'
        "NOT JSON AT ALL\n"
        '{"messages":[{"role":"user","content":"ok2"}]}\n',
        encoding="utf-8",
    )
    traces = load(p, fmt="messages")
    assert len(traces) == 2  # bad line skipped, not crashed
