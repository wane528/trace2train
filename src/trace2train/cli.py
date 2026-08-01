"""trace2train CLI — two commands, UX-first.

    trace2train inspect [FILE]     # instant quality report (the hook), no LLM
    trace2train convert [FILE]     # produce SFT/DPO training data

Design goals: 30-second-to-value, zero-config defaults, screenshot-worthy
output, next-step guidance, never crash on bad input.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .attribute import Attribution
from .clean import CleanIssue, clean_text, filter_leaked, redact_dpo, redact_sft
from .demo import demo_path
from .detect import detect_failures
from .export import export_dpo, export_sft, write_meta
from .generate import generate_records
from .importers import detect_format, load
from .inspect import build_report
from .langfuse import LangfuseConfig, LangfusePullError, PullOptions, pull_observations
from .llm import LLMClient
from .stats import build_stats

app = typer.Typer(
    help="Turn failed agent traces (wrong tool / bad args / wrong format / "
    "over-refusal) into clean tool-call & behavior-fix SFT/DPO training data.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
langfuse_app = typer.Typer(help="Pull Langfuse observation snapshots for local inspect/convert.")
app.add_typer(langfuse_app, name="langfuse")
_CLI_GLYPHS = "↳✓✗→"

# Rough per-trace LLM cost for the pre-run estimate. One call per trace on a
# DeepSeek-class model (~1-2k tokens in/out) lands around this; it is a
# ballpark to set expectations, not a billing figure.
_EST_COST_PER_TRACE = 0.0004

# Set once at startup by main(): can the active console render our box-drawing
# and arrow glyphs? On a legacy Windows code page (cp1252/gbk) it can't, so we
# fall back to ASCII borders and plain-text symbols instead of emitting the
# mojibake "�?" the previous errors="replace" path produced.
_UNICODE_OK = True


class TraceFormat(StrEnum):
    AUTO = "auto"
    LANGSMITH = "langsmith"
    LANGFUSE = "langfuse"
    MESSAGES = "messages"


class ReviewKind(StrEnum):
    SFT = "sft"
    DPO = "dpo"
    BOTH = "both"


def _console_supports_glyphs(stream: object) -> bool:
    """True if the stream's encoding can render trace2train's box/arrow glyphs."""

    encoding = getattr(stream, "encoding", None)
    if not isinstance(encoding, str):
        return False
    try:
        _CLI_GLYPHS.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _configure_console_output(stream: object) -> None:
    """Detect glyph capability; degrade gracefully instead of forcing UTF-8.

    When the console can't encode our glyphs we (a) flip a global ASCII flag so
    tables/panels use ASCII borders, and (b) still set errors='replace' as a
    last-resort guard for any stray unicode in user data.
    """

    global _UNICODE_OK
    if not _console_supports_glyphs(stream):
        _UNICODE_OK = False
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="replace")


def _table_box():
    """Box style for tables/panels — ASCII when the console can't do unicode."""

    return box.ROUNDED if _UNICODE_OK else box.ASCII


def _arrow() -> str:
    return "→" if _UNICODE_OK else "->"


def _branch() -> str:
    return "↳" if _UNICODE_OK else "-"


_ROLE_MAP = {"system": "system", "human": "human", "assistant": "gpt"}


def _sft_panel(convo: list[dict], attribution: str, *, title: str = "SFT sample") -> Panel:
    """Render one SFT sample as a Panel. Shared by `review` and `convert --review`."""

    body = "\n".join(f"[bold]{t['from']}[/bold]: {t['value']}" for t in convo)
    return Panel(
        body + f"\n\n[dim]why: {attribution}[/dim]",
        title=title,
        border_style="green",
        box=_table_box(),
    )


def _dpo_panel(
    prompt: str,
    chosen: str,
    rejected: str,
    attribution: str,
    *,
    title: str = "DPO preference pair",
) -> Panel:
    """Render one DPO pair as a Panel. Shared by `review` and `convert --review`."""

    ok = "✓" if _UNICODE_OK else "[+]"
    bad = "✗" if _UNICODE_OK else "[-]"
    return Panel(
        f"[bold]prompt[/bold]: {prompt}\n\n"
        f"[green]{ok} CHOSEN[/green]:  {chosen}\n"
        f"[red]{bad} REJECTED[/red]: {rejected}\n\n"
        f"[dim]why: {attribution}[/dim]",
        title=title,
        border_style="cyan",
        box=_table_box(),
    )


def _sft_record_to_convo(rec) -> list[dict]:
    """Convert an in-memory SFTRecord's turns to the {from,value} dicts used above."""

    return [{"from": _ROLE_MAP[t.from_.value], "value": t.value} for t in rec.conversations]


def _accept_sft(
    rec,
    *,
    redact: bool,
    seen: set[str],
    leak_fps: set[str],
    dropped: dict,
) -> bool:
    """Redact + decontaminate one SFT record. Returns True if it should be kept.

    Records the rejection reason into `dropped` so the summary stays honest.
    Extracted from the convert loop so `--review` can reuse identical logic.
    """

    if redact:
        redact_sft(rec)
    text = " ".join(t.value for t in rec.conversations)
    cleaned = clean_text(text, seen_fingerprints=seen, redact=False)
    if CleanIssue.EMPTY in cleaned.issues or not cleaned.text:
        dropped["unusable"] += 1
        return False
    if filter_leaked(cleaned.text, leak_fingerprints=leak_fps):
        dropped["leak"] += 1
        return False
    if CleanIssue.DUPLICATE in cleaned.issues:
        dropped["duplicate"] += 1
        return False
    return True


def _accept_dpo(rec, *, redact: bool, leak_fps: set[str], dropped: dict) -> bool:
    """Redact + leak-filter one DPO record. Returns True if it should be kept."""

    if redact:
        redact_dpo(rec)
    if filter_leaked(rec.chosen + "\n" + rec.rejected, leak_fingerprints=leak_fps):
        dropped["leak"] += 1
        return False
    return True


def _already_done_trace_ids(out_dir: Path, offline: bool) -> set[str]:
    """Collect trace_ids already written by a previous run (for --resume).

    Reads the provenance of whichever output file this mode writes to, so a
    re-run after an interruption / rate-limit skips work already paid for.
    """
    import json

    name = "needs_review/raw_traces.jsonl" if offline else "train_sft.jsonl"
    path = out_dir / name
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = rec.get("_provenance", {}).get("trace_id")
        if tid:
            done.add(tid)
    return done


def _render_stats(stats) -> None:
    """Render the dataset health check: failure-type mix, length spread, warnings."""

    total = sum(stats.failure_type_dist.values())
    health = Table(
        title="dataset health",
        show_header=True,
        header_style="bold",
        box=_table_box(),
    )
    health.add_column("failure type")
    health.add_column("count", justify="right")
    health.add_column("share", justify="right")
    for name, count in stats.failure_type_dist.items():
        share = (count / total) if total else 0.0
        # Flag the dominant type in yellow so skew is visible at a glance.
        share_txt = f"{share:.0%}"
        if share > 0.60:
            share_txt = f"[yellow]{share_txt}[/yellow]"
        health.add_row(name, str(count), share_txt)
    console.print(health)

    sl, dl = stats.sft_length, stats.dpo_length
    console.print(
        f"[dim]length (chars)  SFT: min {sl.min} / med {sl.median} / max {sl.max}"
        + (f"   DPO: min {dl.min} / med {dl.median} / max {dl.max}" if dl.count else "")
        + "[/dim]"
    )

    for warning in stats.warnings:
        console.print(f"[yellow]![/yellow] {warning}")


def _run_review(generated: list, dropped: dict) -> list:
    """Interactively approve/reject each generated sample before it is written.

    The human-in-the-loop trust step: shows the corrected sample (same panels as
    `review`) and asks to keep or drop it. Rejected samples are counted in
    `dropped['rejected']` — nothing is silently discarded. Returns the kept
    subset of `generated`. 'a'/'r' apply to the remaining items so large batches
    stay tractable.
    """

    from rich.prompt import Prompt

    console.print(
        f"\n[bold]Review[/bold] {len(generated)} generated sample(s). "
        "[green]k[/green]=keep  [red]d[/red]=drop  "
        "[green]A[/green]=keep all rest  [red]D[/red]=drop all rest\n"
    )

    kept: list = []
    auto: str | None = None  # None | "keep" | "drop"
    for idx, (det, sfts, dpos) in enumerate(generated, start=1):
        for rec in sfts:
            console.print(
                _sft_panel(
                    _sft_record_to_convo(rec),
                    rec.provenance.attribution or "",
                    title=f"SFT sample {idx}/{len(generated)}",
                )
            )
        for rec in dpos:
            console.print(
                _dpo_panel(
                    rec.conversations[0].value if rec.conversations else "",
                    rec.chosen,
                    rec.rejected,
                    rec.provenance.attribution or "",
                    title=f"DPO pair {idx}/{len(generated)}",
                )
            )

        if auto is None:
            choice = Prompt.ask(
                "keep this sample?",
                choices=["k", "d", "A", "D"],
                default="k",
                show_choices=True,
            )
            if choice == "A":
                auto = "keep"
                choice = "k"
            elif choice == "D":
                auto = "drop"
                choice = "d"
        else:
            choice = "k" if auto == "keep" else "d"

        if choice == "k":
            kept.append((det, sfts, dpos))
        else:
            dropped["rejected"] += len(sfts) + len(dpos)

    console.print(
        f"[dim]review complete: kept {len(kept)}, "
        f"dropped {dropped['rejected']}.[/dim]\n"
    )
    return kept


def _quote_display_arg(arg: str) -> str:
    """Quote display-only command arguments when they contain shell-breaking whitespace."""

    escaped = arg.replace('"', '\\"')
    if any(char.isspace() for char in escaped) or '"' in arg:
        return f'"{escaped}"'
    return escaped


def _format_display_command(*args: str) -> str:
    """Build a copy-paste-friendly display command."""

    return " ".join(_quote_display_arg(arg) for arg in args)


def _print_cli_error(message: str) -> None:
    console.print(f"[red]{message}[/red]")


def _print_command_line(command: str, *, style: str) -> None:
    del style
    typer.echo(command)


def _load_traces_or_exit(
    path: Path,
    *,
    fmt: str,
    max_traces: int | None = None,
) -> tuple[list, str]:
    """Load traces and convert expected file/importer failures into CLI exits."""

    try:
        detected_format = detect_format(path)
        traces = load(path, fmt=fmt, max_traces=max_traces)
    except ValueError as error:
        _print_cli_error(str(error))
        raise typer.Exit(code=2) from None
    except OSError as error:
        _print_cli_error(f"Could not read traces: {error}")
        raise typer.Exit(code=2) from None
    return traces, detected_format


def _version_callback(value: bool) -> None:
    if value:
        from . import __version__

        typer.echo(f"trace2train {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the trace2train version and exit.",
    ),
) -> None:
    """Configure CLI process output streams before command execution."""

    _configure_console_output(sys.stdout)
    _configure_console_output(sys.stderr)


def _resolve_input(file: Path | None, demo: bool) -> Path:
    """Pick the input file, with a friendly error if neither is given."""
    if demo:
        return demo_path()
    if file is None:
        console.print(
            "[red]No input.[/red] Pass a traces file, or try [bold]--demo[/bold] "
            "to see it work with sample data:\n"
            "  [dim]trace2train inspect --demo[/dim]"
        )
        raise typer.Exit(code=2)
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(code=2)
    return file


@app.command()
def inspect(
    file: Path | None = typer.Argument(None, help="Traces JSONL (or use --demo)"),
    demo: bool = typer.Option(False, "--demo", help="Use bundled sample data"),
    fmt: TraceFormat = typer.Option(TraceFormat.AUTO, "--format"),
    export: Path | None = typer.Option(None, "--export", help="Write a Markdown report"),
    as_json: bool = typer.Option(
        False, "--json", help="Emit a machine-readable JSON report instead of tables"
    ),
):
    """Instant quality report — how dirty are your traces, how many are trainable?

    Pure rules, no LLM: free, offline, and fast.
    """
    path = _resolve_input(file, demo)
    traces, _ = _load_traces_or_exit(path, fmt=fmt.value)
    report = build_report(traces)

    if as_json:
        import json as _json

        payload = {
            "traces": report.total_traces,
            "failed": report.failed,
            "env_only": report.env_only,
            "trainable": report.trainable,
            "dirty": {
                "pii": report.dirty.pii,
                "duplicate": report.dirty.duplicate,
            },
            "est_sft": report.est_sft,
            "est_dpo": report.est_dpo,
            "failure_types": report.failure_types,
            "source_format": report.source_format,
        }
        typer.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # Headline panel — the screenshot-worthy line.
    console.print(
        Panel(
            f"[bold]{report.headline_for(unicode_ok=_UNICODE_OK)}[/bold]",
            title="trace2train inspect",
            border_style="cyan",
            box=_table_box(),
        )
    )

    branch = _branch()
    table = Table(show_header=True, header_style="bold", box=_table_box())
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("traces", str(report.total_traces))
    table.add_row("failed", f"[yellow]{report.failed}[/yellow]")
    table.add_row(f"  {branch} environmental (not trainable)", f"[dim]{report.env_only}[/dim]")
    table.add_row(f"  {branch} with PII", f"[red]{report.dirty.pii}[/red]")
    table.add_row(f"  {branch} duplicates", f"[red]{report.dirty.duplicate}[/red]")
    table.add_row("[green]trainable[/green]", f"[green]{report.trainable}[/green]")
    table.add_row("SFT candidates (upper bound)", str(report.est_sft))
    table.add_row("DPO candidates (upper bound)", str(report.est_dpo))
    console.print(table)

    # failure sub-type breakdown — what KIND of behavioral failures you have
    if report.failure_types:
        ft = Table(
            title="failure types (trainable)",
            show_header=True,
            header_style="bold",
            box=_table_box(),
        )
        ft.add_column("type")
        ft.add_column("count", justify="right")
        for name, count in report.failure_types.items():
            ft.add_row(name, str(count))
        console.print(ft)

    console.print(
        "[dim]candidates = upper bound. convert only emits data where the correct "
        "answer is derivable from the trace (behavioral fixes), skipping failures "
        "that need external verification.[/dim]"
    )

    if export is not None:
        try:
            _write_markdown_report(report, export)
        except OSError as error:
            _print_cli_error(f"Could not write report: {error}")
            raise typer.Exit(code=2) from None
        console.print(f"[dim]report written to {export}[/dim]")

    # Next step.
    next_command = _format_display_command(
        "trace2train",
        "convert",
        "--demo" if demo else str(path),
    )
    console.print(f"\n[bold]Next:[/bold] turn these into training data {_arrow()}")
    _print_command_line(f"  {next_command}", style="cyan")


@app.command()
def convert(
    file: Path | None = typer.Argument(None, help="Traces JSONL (or use --demo)"),
    demo: bool = typer.Option(False, "--demo", help="Use bundled sample data"),
    out_dir: Path = typer.Option(Path("out"), "--out-dir", "-o", help="Output dir"),
    fmt: TraceFormat = typer.Option(TraceFormat.AUTO, "--format"),
    max_traces: int | None = typer.Option(None, "--max-traces", min=1, help="Limit processed"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Run without an LLM"),
    redact: bool = typer.Option(True, "--redact/--no-redact", help="Redact PII"),
    review: bool = typer.Option(
        False,
        "--review",
        help="Approve/reject each corrected sample before it is written (needs an LLM)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Skip traces already in a previous run's output (avoids re-paying for LLM calls)",
    ),
    leak_file: Path | None = typer.Option(
        None, "--leak-file", help="Eval-set fingerprints to exclude"
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit a machine-readable JSON summary instead of tables"
    ),
):
    """Convert failed agent traces into LLaMA-Factory-ready SFT/DPO JSONL.

    Scope: corrects BEHAVIORAL failures whose right answer is derivable from the
    trace (wrong tool, bad args, lost context, wrong format, plainly wrong
    common-sense answer, over-refusal). It deliberately SKIPS failures that need
    external verification (did code pass tests? is a fact accurate?) rather than
    fabricate a correct answer — those are reported as "skipped".
    """
    path = _resolve_input(file, demo)
    traces, detected_format = _load_traces_or_exit(
        path,
        fmt=fmt.value,
        max_traces=max_traces,
    )

    if not as_json:
        console.print(f"[bold]Loading[/bold] {path} [dim]({detected_format})[/dim]")

    detections = detect_failures(traces)
    failed = [d for d in detections if d.failed]
    trainable = [d for d in failed if d.trainable]
    if not as_json:
        console.print(
            f"  {len(traces)} traces, [yellow]{len(failed)} failed[/yellow], "
            f"[green]{len(trainable)} trainable[/green]"
        )

    if as_json:
        review = False  # interactive review is incompatible with machine output

    client = None if no_llm else LLMClient()
    if client is not None and not client.configured:
        if not as_json:
            console.print(
                "[yellow]No T2T_LLM_API_KEY found - running offline.[/yellow]\n"
                "[dim]For LLM-corrected data: copy .env.example to .env and add a key "
                "(DeepSeek is cheap; any OpenAI-compatible provider works via "
                "T2T_LLM_BASE_URL/T2T_LLM_MODEL).[/dim]"
            )
        client = None

    offline = client is None
    if offline:
        if not as_json:
            console.print(
                "[yellow]Offline mode: the corrected answer can't be derived without an "
                "LLM. Raw failed traces are written to [bold]needs_review/[/bold] for "
                "human curation - NOT to train_sft.jsonl.[/yellow]"
            )
        if review:
            if not as_json:
                console.print(
                    "[dim]--review has no effect offline (no corrections to approve).[/dim]"
                )
            review = False

    leak_fps: set[str] = set()
    if leak_file is not None and leak_file.exists():
        leak_fps = {
            ln.strip()
            for ln in leak_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }

    sft_records: list = []
    dpo_records: list = []
    seen: set[str] = set()
    dropped = {
        "unusable": 0,
        "duplicate": 0,
        "leak": 0,
        "skipped_uncertain": 0,
        "rejected": 0,
    }

    # --resume: skip traces already written by a previous run so a re-run after
    # an interruption / rate-limit doesn't re-pay for the same LLM calls.
    to_process = trainable
    if resume:
        done_ids = _already_done_trace_ids(out_dir, offline)
        if done_ids:
            before = len(to_process)
            to_process = [d for d in to_process if d.trace.trace_id not in done_ids]
            skipped = before - len(to_process)
            if not as_json:
                console.print(
                    f"[dim]resume: skipping {skipped} trace(s) already in {out_dir}.[/dim]"
                )

    # Cost/time expectation: with an LLM, convert makes ONE call per trace.
    if client is not None and not as_json:
        n = len(to_process)
        console.print(
            f"[dim]~{n} LLM call(s) (~${n * _EST_COST_PER_TRACE:.4f} est. on a "
            "DeepSeek-class model; 1 call per trace).[/dim]"
        )

    # Generate first (collect everything), then optionally hand it to the human
    # for approval, then decontaminate + keep. Interactive review must NOT run
    # inside a live progress bar (it would fight the prompt).
    generated: list = []  # list of (det, sfts, dpos)

    def _generate_one(det, on_step) -> None:
        # One LLM call inside generate_records now does both attribution and
        # correction (offline passes a synthetic label).
        attribution = None if client is not None else Attribution(
            "unknown", "offline mode", False
        )
        sfts, dpos = generate_records(det.trace, attribution, client)
        if not sfts and not dpos:
            # trainable failure, but the correct answer isn't derivable from the
            # trace (needs external ground truth) — skipped on purpose, not
            # silently dropped. Surfaced in the summary.
            dropped["skipped_uncertain"] += 1
        else:
            generated.append((det, sfts, dpos))
        on_step()

    if as_json:
        for det in to_process:
            _generate_one(det, lambda: None)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]Generating[/bold]"),
            BarColumn(),
            TextColumn("[progress.completed]{task.completed}/{task.total}"),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("gen", total=len(to_process))
            for det in to_process:
                _generate_one(det, lambda: progress.advance(task_id))

    if review:
        generated = _run_review(generated, dropped)

    kept_dets: list = []  # detections that contributed >=1 kept record (for stats)
    for det, sfts, dpos in generated:
        contributed = False
        for rec in sfts:
            if _accept_sft(rec, redact=redact, seen=seen, leak_fps=leak_fps, dropped=dropped):
                sft_records.append(rec)
                contributed = True
        for rec in dpos:
            if _accept_dpo(rec, redact=redact, leak_fps=leak_fps, dropped=dropped):
                dpo_records.append(rec)
                contributed = True
        if contributed:
            kept_dets.append(det)

    stats = build_stats(sft_records, dpo_records, kept_dets, dropped)

    # In offline mode the records are raw, unverified traces — they must NOT be
    # presented as ready-to-train SFT data. Route them to needs_review/ instead.
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # --resume appends to the previous run's files instead of overwriting.
        if offline:
            review_dir = out_dir / "needs_review"
            review_dir.mkdir(parents=True, exist_ok=True)
            sft_path = export_sft(sft_records, review_dir / "raw_traces.jsonl", append=resume)
            dpo_path = export_dpo(dpo_records, out_dir / "train_dpo.jsonl", append=resume)
        else:
            sft_path = export_sft(sft_records, out_dir / "train_sft.jsonl", append=resume)
            dpo_path = export_dpo(dpo_records, out_dir / "train_dpo.jsonl", append=resume)
        meta_path = write_meta(
            sft_path,
            dpo_path,
            total_traces=len(traces),
            failed_traces=len(failed),
            trainable_traces=len(trainable),
            dropped=dropped,
            out_dir=out_dir,
            offline=offline,
            distribution=stats.to_meta(),
        )
    except OSError as error:
        _print_cli_error(f"Could not write output: {error}")
        raise typer.Exit(code=2) from None

    if as_json:
        import json as _json

        summary = {
            "mode": "offline" if offline else "llm",
            "traces": len(traces),
            "trainable": len(trainable),
            ("raw_review_records" if offline else "sft_records"): len(sft_records),
            "dpo_records": len(dpo_records),
            "dropped": dropped,
            "distribution": stats.to_meta(),
            "outputs": {
                ("raw_review" if offline else "sft"): str(sft_path),
                "dpo": str(dpo_path),
                "meta": str(meta_path),
            },
        }
        typer.echo(_json.dumps(summary, ensure_ascii=False, indent=2))
        return

    sft_label = "raw traces (needs review)" if offline else "SFT records"
    table = Table(title="trace2train convert", box=_table_box())
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("traces", str(len(traces)))
    table.add_row("trainable", str(len(trainable)))
    table.add_row(f"[green]{sft_label}[/green]", f"[green]{len(sft_records)}[/green]")
    table.add_row("[green]DPO records[/green]", f"[green]{len(dpo_records)}[/green]")
    table.add_row(
        "skipped (answer not derivable)",
        f"[dim]{dropped['skipped_uncertain']}[/dim]",
    )
    table.add_row("dropped (dup/leak/empty)",
                  str(dropped["duplicate"] + dropped["leak"] + dropped["unusable"]))
    if dropped["rejected"]:
        table.add_row("rejected in review", f"[red]{dropped['rejected']}[/red]")
    console.print(table)

    # Dataset health check — is the produced set balanced, or skewed? (non-offline
    # only: offline raw traces aren't a training set yet.)
    if not offline and (sft_records or dpo_records):
        _render_stats(stats)

    if dropped["skipped_uncertain"] and not sft_records:
        console.print(
            "[yellow]Note:[/yellow] all failures needed external verification "
            "(e.g. code tests, fact-checks) — the correct answer isn't in the "
            "trace. trace2train corrects behavioral failures (wrong tool/format/"
            "common-sense), not correctness that needs ground truth."
        )

    if offline:
        console.print(
            f"\n[green]Done.[/green] Raw traces: {sft_path}  meta: {meta_path}"
        )
        console.print(
            "[bold]Next:[/bold] set T2T_LLM_API_KEY and re-run to get corrected, "
            "trainable data - or hand-fix the raw traces above."
        )
    else:
        next_command = _format_display_command(
            "llamafactory-cli",
            "train",
            "--dataset",
            str(sft_path),
            "--stage",
            "sft",
            "...",
        )
        console.print(
            f"\n[green]Done.[/green] SFT: {sft_path}  DPO: {dpo_path}  meta: {meta_path}"
        )
        console.print("[bold]Next:[/bold] fine-tune with LLaMA-Factory →")
        _print_command_line(f"  {next_command}", style="dim")


@langfuse_app.command("pull")
def langfuse_pull(
    output: Path = typer.Argument(
        Path("langfuse_observations.jsonl"),
        help="Output JSONL snapshot path",
    ),
    from_time: str | None = typer.Option(
        None,
        "--from-time",
        help="Inclusive ISO-8601 observation start time",
    ),
    to_time: str | None = typer.Option(
        None,
        "--to-time",
        help="Exclusive ISO-8601 observation start time",
    ),
    max_observations: int | None = typer.Option(
        None,
        "--max-observations",
        min=1,
        help="Stop after this many observations",
    ),
    page_size: int = typer.Option(
        100,
        "--page-size",
        min=1,
        max=1000,
        help="API page size",
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="Langfuse host override"),
) -> None:
    """Pull Langfuse v4 Public API v2 observations into a local JSONL snapshot."""

    config = LangfuseConfig.from_env(base_url=base_url)
    options = PullOptions(
        from_time=from_time,
        to_time=to_time,
        max_observations=max_observations,
        page_size=page_size,
    )
    try:
        observation_count = pull_observations(output, config=config, options=options)
    except LangfusePullError as error:
        _print_cli_error(str(error))
        raise typer.Exit(code=1) from None

    console.print(f"Pulled {observation_count} observations")
    _print_command_line(f"Snapshot: {str(output)}", style="none")
    _print_command_line(
        f"Next: {_format_display_command('trace2train', 'inspect', str(output))}",
        style="none",
    )


@app.command()
def review(
    out_dir: Path = typer.Option(
        Path("out"), "--out-dir", "-o", help="Where convert wrote its files"
    ),
    limit: int = typer.Option(5, "--limit", "-n", help="How many samples to show"),
    kind: ReviewKind = typer.Option(ReviewKind.BOTH, "--kind", help="Which records to show"),
):
    """Pretty-print generated training samples so you can judge quality by eye.

    For SFT: shows the human prompt and the corrected assistant answer.
    For DPO: shows prompt, the CHOSEN (good) answer, and the REJECTED (failed) one.
    """
    import json

    def _load(name: str) -> list[dict]:
        p = out_dir / name
        if not p.exists():
            return []
        return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    shown = 0
    if kind in (ReviewKind.SFT, ReviewKind.BOTH):
        for rec in _load("train_sft.jsonl")[:limit]:
            prov = rec.get("_provenance", {})
            console.print(
                _sft_panel(rec.get("conversations", []), prov.get("attribution", ""))
            )
            shown += 1

    if kind in (ReviewKind.DPO, ReviewKind.BOTH):
        for rec in _load("train_dpo.jsonl")[:limit]:
            prov = rec.get("_provenance", {})
            console.print(
                _dpo_panel(
                    rec.get("conversations", [{}])[0].get("value", ""),
                    rec.get("chosen", {}).get("value", ""),
                    rec.get("rejected", {}).get("value", ""),
                    prov.get("attribution", ""),
                )
            )
            shown += 1

    if shown == 0:
        console.print(
            f"[yellow]No samples found in {out_dir}.[/yellow] "
            f"Run [cyan]trace2train convert ...[/cyan] first."
        )


def _write_markdown_report(report, path: Path) -> None:
    """Export the inspect report as a shareable Markdown file."""
    d = report.dirty
    md = (
        f"# trace2train inspect report\n\n"
        f"> {report.headline}\n\n"
        f"| Metric | Count |\n|---|---:|\n"
        f"| traces | {report.total_traces} |\n"
        f"| failed | {report.failed} |\n"
        f"| environmental (not trainable) | {report.env_only} |\n"
        f"| with PII | {d.pii} |\n"
        f"| duplicates | {d.duplicate} |\n"
        f"| trainable | {report.trainable} |\n"
        f"| SFT candidates (upper bound) | {report.est_sft} |\n"
        f"| DPO candidates (upper bound) | {report.est_dpo} |\n\n"
        f"_candidates = upper bound; `convert` only emits data where the correct "
        f"answer is derivable from the trace (behavioral fixes)._\n"
    )
    if report.failure_types:
        md += "\n## Failure types (trainable)\n\n| Type | Count |\n|---|---:|\n"
        for name, count in report.failure_types.items():
            md += f"| {name} | {count} |\n"
    path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    app()
