from __future__ import annotations

from pathlib import Path

from trace2train.importers.langfuse import LangfuseImporter
from trace2train.models import RunType

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "langfuse_observations.jsonl"


def test_sniff_matches_official_v2_observation_rows() -> None:
    importer = LangfuseImporter()

    assert importer.sniff(
        {
            "id": "obs-1",
            "traceId": "trace-1",
            "startTime": "2024-05-01T10:00:00+00:00",
            "type": "GENERATION",
        }
    )
    assert importer.sniff(
        {
            "id": "obs-2",
            "traceId": "trace-2",
            "startTime": "2024-05-01T10:00:00+00:00",
            "parentObservationId": "parent-1",
        }
    )
    assert importer.sniff(
        {
            "id": "obs-3",
            "traceId": "trace-3",
            "startTime": "2024-05-01T10:00:00+00:00",
            "isRootObservation": True,
        }
    )


def test_sniff_requires_core_fields_and_exact_discriminators() -> None:
    importer = LangfuseImporter()

    assert not importer.sniff(
        {
            "id": "obs-1",
            "traceId": "trace-1",
            "type": "GENERATION",
        }
    )
    assert not importer.sniff(
        {
            "id": "obs-1",
            "traceId": "trace-1",
            "startTime": "2024-05-01T10:00:00+00:00",
        }
    )
    assert not importer.sniff(
        {
            "id": "obs-4",
            "traceId": "trace-4",
            "startTime": "2024-05-01T10:00:00+00:00",
            "level": "ERROR",
        }
    )
    assert not importer.sniff(
        {
            "id": "obs-5",
            "traceId": "trace-5",
            "startTime": "2024-05-01T10:00:00+00:00",
            "statusMessage": "oops",
        }
    )
    assert not importer.sniff(
        {
            "id": "obs-6",
            "traceId": "trace-6",
            "startTime": "2024-05-01T10:00:00+00:00",
            "sessionId": "session-only",
        }
    )
    assert not importer.sniff(
        {
            "id": "obs-7",
            "traceId": "trace-7",
            "startTime": "2024-05-01T10:00:00+00:00",
            "tags": ["only-tags"],
        }
    )


def test_load_groups_rows_into_sorted_traces_and_maps_runs() -> None:
    importer = LangfuseImporter()

    traces = importer.load(FIXTURE_PATH)

    assert [trace.trace_id for trace in traces] == [
        "trace-a",
        "trace-b",
        "trace-c",
        "trace-d",
        "trace-e",
        "trace-f",
        "trace-h",
    ]

    trace_a = traces[0]
    assert trace_a.source_format == "langfuse"
    assert trace_a.source_file == str(FIXTURE_PATH)
    assert trace_a.session_id == "session-a"
    assert trace_a.root.id == "obs-root-a"
    assert [run.id for run in trace_a.runs] == ["obs-root-a", "obs-child-a"]
    assert trace_a.root.run_type is RunType.CHAIN
    assert trace_a.root.inputs == {"messages": [{"role": "user", "content": "hello"}]}
    assert trace_a.root.outputs == {"output": "done"}
    assert trace_a.root.error is None
    assert trace_a.root.start_time is not None and trace_a.root.start_time.tzinfo is not None
    assert trace_a.root.tags == ["prod"]

    child_a = trace_a.runs[1]
    assert child_a.run_type is RunType.LLM
    assert child_a.inputs == {"input": "hello"}
    assert child_a.outputs == {"output": [{"role": "assistant", "content": "hi"}]}
    assert child_a.error == "model overloaded"
    assert child_a.parent_run_id == "obs-root-a"
    assert child_a.tags == ["prod", "llm"]
    assert child_a.raw == {
        "id": "obs-child-a",
        "traceId": "trace-a",
        "parentObservationId": "obs-root-a",
        "startTime": "2024-05-01T10:01:00+00:00",
        "endTime": "2024-05-01T10:03:30+00:00",
        "type": "GENERATION",
        "name": "draft answer",
        "level": "ERROR",
        "statusMessage": "model overloaded",
        "input": "hello",
        "output": [{"role": "assistant", "content": "hi"}],
        "sessionId": "session-a",
        "tags": ["prod", "llm", 9],
    }
    assert trace_a.end_time == child_a.end_time

    trace_b = traces[1]
    assert trace_b.root.id == "obs-orphan-b"
    assert trace_b.root.parent_run_id == "missing-remote-parent"
    assert [run.id for run in trace_b.runs] == ["obs-orphan-b", "obs-child-b"]
    assert trace_b.root.run_type is RunType.TOOL
    assert trace_b.root.inputs == {"input": ["query", "langfuse"]}
    assert trace_b.root.outputs == {"output": "found docs"}
    assert trace_b.root.error is None
    assert trace_b.root.tags == ["search"]
    assert trace_b.runs[1].run_type is RunType.CHAIN

    trace_c = traces[2]
    assert trace_c.root.id == "obs-root-c"
    assert trace_c.root.run_type is RunType.EMBEDDING
    assert trace_c.root.inputs == {}
    assert trace_c.root.outputs == {}
    assert trace_c.runs[1].run_type is RunType.UNKNOWN
    assert trace_c.runs[1].error == "Langfuse observation obs-child-c failed"
    assert trace_c.runs[1].inputs == {"input": "raw text"}
    assert trace_c.runs[1].outputs == {"output": "still text"}

    trace_d = traces[3]
    assert trace_d.root.id == "obs-root-d"
    assert trace_d.root.run_type is RunType.AGENT
    assert trace_d.runs[1].run_type is RunType.RETRIEVER
    assert trace_d.runs[1].error is None

    trace_e = traces[4]
    assert trace_e.root.id == "obs-root-e"
    assert trace_e.root.parent_run_id is None
    assert trace_e.runs[1].name == ""
    assert trace_e.runs[1].run_type is RunType.CHAIN
    assert trace_e.runs[1].inputs == {"input": 42}
    assert trace_e.runs[1].outputs == {"output": True}
    assert trace_e.runs[2].inputs == {"input": "quoted string"}
    assert trace_e.runs[2].outputs == {"output": "plain text"}
    assert trace_e.runs[3].inputs == {"tool_calls": [{"name": "search", "arguments": {"q": "x"}}]}
    assert trace_e.runs[3].outputs == {"items": [1, 2]}
    assert trace_e.runs[4].inputs == {"input": "not-json"}
    assert trace_e.runs[4].outputs == {"output": "{bad json"}

    trace_f = traces[5]
    assert trace_f.root.id == "obs-root-f-early"
    assert trace_f.root.parent_run_id == "missing-f-parent"
    assert trace_f.runs[0].error is None
    assert trace_f.runs[1].error is None
    assert trace_f.runs[2].error == "Langfuse observation obs-f-error failed"

    trace_h = traces[6]
    assert trace_h.root.id == "obs-root-h"
    assert [run.id for run in trace_h.runs] == ["obs-root-h", "obs-child-h"]
    assert trace_h.start_time == trace_h.root.start_time
    assert trace_h.end_time == trace_h.root.end_time
    assert trace_h.runs[1].end_time is None


def test_load_selects_roots_by_required_precedence() -> None:
    importer = LangfuseImporter()

    traces = {trace.trace_id: trace for trace in importer.load(FIXTURE_PATH)}

    assert traces["trace-a"].root.id == "obs-root-a"
    assert traces["trace-b"].root.id == "obs-orphan-b"
    assert traces["trace-c"].root.id == "obs-root-c"
    assert traces["trace-d"].root.id == "obs-root-d"
    assert traces["trace-e"].root.id == "obs-root-e"
    assert traces["trace-f"].root.id == "obs-root-f-early"
    assert traces["trace-h"].root.id == "obs-root-h"


def test_load_skips_malformed_and_structurally_unusable_rows() -> None:
    importer = LangfuseImporter()

    traces = importer.load(FIXTURE_PATH)

    loaded_run_ids = {run.id for trace in traces for run in trace.runs}
    assert "bad-missing-start" not in loaded_run_ids
    assert "bad-not-langfuse" not in loaded_run_ids
    assert "obs-invalid-start-h" not in loaded_run_ids
    assert all(not trace.trace_id.startswith("trace-skip") for trace in traces)


def test_load_skips_entire_trace_when_observation_ids_are_duplicated() -> None:
    importer = LangfuseImporter()

    traces = importer.load(FIXTURE_PATH)

    assert "trace-g" not in {trace.trace_id for trace in traces}


def test_load_skips_rows_with_invalid_non_empty_start_time_before_grouping() -> None:
    importer = LangfuseImporter()

    traces = {trace.trace_id: trace for trace in importer.load(FIXTURE_PATH)}

    trace_h = traces["trace-h"]

    assert trace_h.root.id == "obs-root-h"
    assert [run.id for run in trace_h.runs] == ["obs-root-h", "obs-child-h"]
    assert all(run.id != "obs-invalid-start-h" for run in trace_h.runs)
    assert trace_h.start_time == trace_h.root.start_time
    assert trace_h.runs[1].end_time is None


def test_load_applies_max_traces_to_complete_traces() -> None:
    importer = LangfuseImporter()

    traces = importer.load(FIXTURE_PATH, max_traces=2)

    assert [trace.trace_id for trace in traces] == ["trace-a", "trace-b"]
    assert [run.id for run in traces[1].runs] == ["obs-orphan-b", "obs-child-b"]
