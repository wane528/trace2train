"""CLI tests for user-facing integration flows."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import httpx
from typer.testing import CliRunner

from trace2train.cli import _configure_console_output, app
from trace2train.langfuse import LangfuseConfig, PullOptions, pull_observations

runner = CliRunner()


def _langfuse_page(rows: list[dict[str, object]], cursor: str | None) -> httpx.Response:
    return httpx.Response(200, json={"data": rows, "meta": {"cursor": cursor}})


def _langfuse_row(
    observation_id: str,
    trace_id: str,
    start_time: str,
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": observation_id,
        "traceId": trace_id,
        "startTime": start_time,
        "type": "GENERATION",
    }
    row.update(extra)
    return row


def test_inspect_help_lists_langfuse_format() -> None:
    result = runner.invoke(app, ["inspect", "--help"])

    assert result.exit_code == 0
    assert "--format" in result.stdout
    assert "langfuse" in result.stdout


def test_convert_help_lists_langfuse_format() -> None:
    result = runner.invoke(app, ["convert", "--help"])

    assert result.exit_code == 0
    assert "--format" in result.stdout


def test_configure_console_output_reconfigures_non_utf8_streams_to_replace() -> None:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp936")

    _configure_console_output(stream)
    stream.write("↳ ✓ ✗ →")
    stream.flush()

    assert stream.errors == "replace"
    assert buffer.getvalue()


def test_inspect_unknown_format_exits_2_without_traceback(messages_file: Path) -> None:
    result = runner.invoke(app, ["inspect", str(messages_file), "--format", "nope"])

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_inspect_directory_path_exits_2_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["inspect", str(tmp_path)])

    assert result.exit_code == 2
    assert "Could not read traces" in result.stdout
    assert "Traceback" not in result.stdout


def test_convert_unknown_format_exits_2_without_traceback(messages_file: Path) -> None:
    result = runner.invoke(app, ["convert", str(messages_file), "--format", "nope", "--no-llm"])

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_convert_empty_file_exits_2_without_traceback(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")

    result = runner.invoke(app, ["convert", str(empty_file), "--no-llm"])

    assert result.exit_code == 2
    assert "Unknown/unsupported trace format" in result.stdout
    assert "Traceback" not in result.stdout


def test_convert_directory_path_exits_2_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["convert", str(tmp_path), "--no-llm"])

    assert result.exit_code == 2
    assert "Could not read traces" in result.stdout
    assert "Traceback" not in result.stdout


def test_convert_validates_max_traces_at_cli_boundary(messages_file: Path) -> None:
    result = runner.invoke(app, ["convert", str(messages_file), "--no-llm", "--max-traces", "0"])

    assert result.exit_code == 2


def test_inspect_next_command_quotes_path_with_spaces() -> None:
    with tempfile.TemporaryDirectory(prefix="trace2train space ") as temp_dir:
        path = Path(temp_dir) / "messages fixture.jsonl"
        path.write_text(
            '{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]}\n',
            encoding="utf-8",
        )

        result = runner.invoke(app, ["inspect", str(path)])

    assert result.exit_code == 0
    assert f'trace2train convert "{path}"' in result.stdout


def test_convert_offline_routes_raw_traces_to_needs_review(messages_file: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="trace2train out ") as temp_dir:
        out_dir = Path(temp_dir) / "model output"

        result = runner.invoke(
            app,
            ["convert", str(messages_file), "--no-llm", "--out-dir", str(out_dir)],
        )

        assert result.exit_code == 0
        # Offline mode must NOT present raw traces as trainable SFT data.
        assert not (out_dir / "train_sft.jsonl").exists()
        assert (out_dir / "needs_review" / "raw_traces.jsonl").exists()
    # It must not tell the user to feed unverified data straight to a trainer.
    assert "llamafactory-cli train" not in result.stdout
    assert "needs_review" in result.stdout


class _FakeLLMClient:
    """Configured fake client: a single call returns attribution + correction."""

    configured = True

    def complete_json(self, system, user, **kw):
        return {
            "category": "wrong_tool",
            "summary": "used calculator instead of get_weather",
            "correctable": True,
            "instruction": "What is the weather in Tokyo?",
            "correct_answer": "Use the get_weather tool with city=Tokyo.",
            "failed_answer": "calculator says 42",
        }


def _patch_llm(monkeypatch) -> None:
    import trace2train.cli as cli_module

    monkeypatch.setattr(cli_module, "LLMClient", lambda *a, **k: _FakeLLMClient())


def test_convert_review_keep_writes_sft(monkeypatch, langsmith_file: Path, tmp_path: Path) -> None:
    _patch_llm(monkeypatch)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        ["convert", str(langsmith_file), "--review", "--out-dir", str(out_dir)],
        input="k\n",
    )

    assert result.exit_code == 0
    sft = out_dir / "train_sft.jsonl"
    assert sft.exists()
    assert sft.read_text(encoding="utf-8").strip()  # kept the sample


def test_convert_review_drop_all_yields_empty_sft(
    monkeypatch, langsmith_file: Path, tmp_path: Path
) -> None:
    _patch_llm(monkeypatch)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        ["convert", str(langsmith_file), "--review", "--out-dir", str(out_dir)],
        input="D\n",  # drop all rest
    )

    assert result.exit_code == 0
    assert "rejected in review" in result.stdout
    sft_lines = (out_dir / "train_sft.jsonl").read_text(encoding="utf-8").splitlines()
    assert sft_lines == []


def test_convert_writes_distribution_stats_to_meta(
    monkeypatch, langsmith_file: Path, tmp_path: Path
) -> None:
    _patch_llm(monkeypatch)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        ["convert", str(langsmith_file), "--out-dir", str(out_dir)],
    )

    assert result.exit_code == 0
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert "distribution" in meta
    dist = meta["distribution"]
    assert "failure_type_dist" in dist
    assert "sft_length" in dist
    assert "warnings" in dist
    # health table rendered to the terminal
    assert "dataset health" in result.stdout


def test_version_flag_prints_version_and_exits() -> None:
    from trace2train import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_inspect_json_emits_machine_readable(messages_file: Path) -> None:
    result = runner.invoke(app, ["inspect", str(messages_file), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "traces" in payload and "trainable" in payload
    assert "failure_types" in payload
    # no rich tables in JSON mode
    assert "trace2train inspect" not in result.stdout


def test_convert_json_emits_machine_readable(
    monkeypatch, langsmith_file: Path, tmp_path: Path
) -> None:
    _patch_llm(monkeypatch)
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app, ["convert", str(langsmith_file), "--out-dir", str(out_dir), "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "llm"
    assert "sft_records" in payload
    assert "distribution" in payload
    assert "outputs" in payload
    # tables suppressed
    assert "dataset health" not in result.stdout


def test_convert_single_call_attribution_in_provenance(
    monkeypatch, langsmith_file: Path, tmp_path: Path
) -> None:
    """The merged single LLM call must still record category/summary provenance."""
    _patch_llm(monkeypatch)
    out_dir = tmp_path / "out"
    runner.invoke(app, ["convert", str(langsmith_file), "--out-dir", str(out_dir)])
    rows = [
        json.loads(ln)
        for ln in (out_dir / "train_sft.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert rows
    assert "wrong_tool" in rows[0]["_provenance"]["attribution"]


def test_convert_resume_skips_already_done(
    monkeypatch, langsmith_file: Path, tmp_path: Path
) -> None:
    _patch_llm(monkeypatch)
    out_dir = tmp_path / "out"

    first = runner.invoke(app, ["convert", str(langsmith_file), "--out-dir", str(out_dir)])
    assert first.exit_code == 0
    count_after_first = len(
        (out_dir / "train_sft.jsonl").read_text(encoding="utf-8").splitlines()
    )

    second = runner.invoke(
        app, ["convert", str(langsmith_file), "--out-dir", str(out_dir), "--resume"]
    )
    assert second.exit_code == 0
    assert "resume: skipping" in second.stdout
    # resume must not duplicate the already-written records
    count_after_resume = len(
        (out_dir / "train_sft.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert count_after_resume == count_after_first


def test_review_kind_rejects_invalid_value(tmp_path: Path) -> None:
    result = runner.invoke(app, ["review", "--kind", "stf", "-o", str(tmp_path)])
    assert result.exit_code != 0  # enum validation kicks in


def test_langfuse_pull_help_lists_approved_options() -> None:
    result = runner.invoke(app, ["langfuse", "pull", "--help"])

    assert result.exit_code == 0
    assert "--from-time" in result.stdout
    assert "--to-time" in result.stdout
    assert "--max-observations" in result.stdout
    assert "--page-size" in result.stdout
    assert "--base-url" in result.stdout
    assert "LANGFUSE_PUBLIC_KEY" not in result.stdout
    assert "LANGFUSE_SECRET_KEY" not in result.stdout


def test_langfuse_pull_validates_page_size_before_http(monkeypatch, tmp_path: Path) -> None:
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("pull should not be called")

    monkeypatch.setattr("trace2train.cli.pull_observations", fail_if_called)

    output_path = tmp_path / "langfuse.jsonl"
    result = runner.invoke(app, ["langfuse", "pull", str(output_path), "--page-size", "0"])

    assert result.exit_code == 2
    assert called is False
    assert not output_path.exists()


def test_langfuse_pull_prints_sanitized_error_and_exit_1(monkeypatch, tmp_path: Path) -> None:
    def raise_pull_error(*args, **kwargs):
        from trace2train.langfuse import LangfusePullError

        raise LangfusePullError("Langfuse authentication failed")

    monkeypatch.setattr("trace2train.cli.pull_observations", raise_pull_error)

    result = runner.invoke(app, ["langfuse", "pull", str(tmp_path / "snapshot.jsonl")])

    assert result.exit_code == 1
    assert "Langfuse authentication failed" in result.stdout
    assert "Traceback" not in result.stdout


def test_langfuse_pull_to_inspect_uses_real_pull_client_and_auto_detects_langfuse(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    rows = [
        _langfuse_row(
            "obs-root",
            "trace-cli",
            "2026-08-01T10:00:00Z",
            isRootObservation=True,
            type="CHAIN",
            name="agent",
            input={"messages": [{"role": "user", "content": "Need weather"}]},
            output="tool result pending",
            sessionId="session-cli",
        ),
        _langfuse_row(
            "obs-child",
            "trace-cli",
            "2026-08-01T10:00:01Z",
            parentObservationId="obs-root",
            type="TOOL",
            name="weather tool",
            level="ERROR",
            statusMessage="ToolError: wrong tool for weather",
            input={"city": "Tokyo"},
            output="42",
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _langfuse_page(rows, None)

    def pull_with_mock_transport(
        output: Path,
        *,
        config: LangfuseConfig,
        options: PullOptions,
    ) -> int:
        return pull_observations(
            output,
            config=config,
            options=options,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr("trace2train.cli.pull_observations", pull_with_mock_transport)

    snapshot_path = tmp_path / "langfuse_snapshot.jsonl"
    pull_result = runner.invoke(app, ["langfuse", "pull", str(snapshot_path)])
    inspect_result = runner.invoke(app, ["inspect", str(snapshot_path)])

    assert pull_result.exit_code == 0
    assert "2 observations" in pull_result.stdout
    assert f"trace2train inspect {snapshot_path}" in pull_result.stdout
    assert inspect_result.exit_code == 0
    assert "langfuse" in inspect_result.stdout.lower()
    assert "wrong_tool" in inspect_result.stdout
    assert "1 failed" in inspect_result.stdout


def test_langfuse_pull_to_convert_no_llm_writes_sft_dpo_and_langfuse_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    rows = [
        _langfuse_row(
            "obs-root",
            "trace-convert",
            "2026-08-01T11:00:00Z",
            isRootObservation=True,
            type="CHAIN",
            name="agent",
            input={
                "messages": [
                    {"role": "user", "content": "remember my name is Alex"},
                    {"role": "assistant", "content": "I forgot"},
                ]
            },
            output="I forgot",
            sessionId="session-convert",
        ),
        _langfuse_row(
            "obs-child",
            "trace-convert",
            "2026-08-01T11:00:01Z",
            parentObservationId="obs-root",
            type="GENERATION",
            name="draft answer",
            level="ERROR",
            statusMessage="ContextError: lost context",
            input="remember my name is Alex",
            output="I don't know your name.",
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _langfuse_page(rows, None)

    def pull_with_mock_transport(
        output: Path,
        *,
        config: LangfuseConfig,
        options: PullOptions,
    ) -> int:
        return pull_observations(
            output,
            config=config,
            options=options,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr("trace2train.cli.pull_observations", pull_with_mock_transport)

    snapshot_path = tmp_path / "langfuse_snapshot.jsonl"
    out_dir = tmp_path / "out"

    pull_result = runner.invoke(app, ["langfuse", "pull", str(snapshot_path)])
    convert_result = runner.invoke(
        app,
        ["convert", str(snapshot_path), "--no-llm", "--out-dir", str(out_dir)],
    )

    assert pull_result.exit_code == 0
    assert convert_result.exit_code == 0
    # Offline: raw traces go to needs_review/, not to trainable train_sft.jsonl.
    raw_path = out_dir / "needs_review" / "raw_traces.jsonl"
    assert not (out_dir / "train_sft.jsonl").exists()
    assert raw_path.exists()
    assert (out_dir / "train_dpo.jsonl").exists()
    assert (out_dir / "meta.json").exists()

    sft_lines = raw_path.read_text(encoding="utf-8").splitlines()
    dpo_lines = (out_dir / "train_dpo.jsonl").read_text(encoding="utf-8").splitlines()
    sft_rows = [json.loads(line) for line in sft_lines]
    dpo_rows = [json.loads(line) for line in dpo_lines]
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

    assert sft_rows
    assert dpo_rows == []
    assert meta["mode"] == "offline"
    assert sft_rows[0]["_provenance"]["source_file"] == str(snapshot_path)
    assert sft_rows[0]["_provenance"]["trace_id"] == "trace-convert"
    assert sft_rows[0]["_provenance"]["run_id"] == "obs-root"
    assert "ContextError: lost context" in (sft_rows[0]["_provenance"]["original_error"] or "")
    assert meta["outputs"]["raw_review"].endswith("raw_traces.jsonl")
    assert meta["outputs"]["dpo"].endswith("train_dpo.jsonl")
