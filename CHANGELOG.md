# Changelog

All notable changes to trace2train are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/) style and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `convert --review` — human-in-the-loop approval: shows each corrected sample
  and lets you keep/drop it before it is written (`k`/`d` per sample, `A`/`D`
  for all remaining). Rejections are counted, never silently dropped.
- `convert --resume` — skips traces already present in a previous run's output
  and appends, so a re-run after a rate-limit or interruption does not re-pay
  for the same LLM calls.
- `convert`/`inspect --json` — machine-readable JSON output for CI/scripts.
- `--version` — global flag to print the installed version.
- Dataset health check after `convert`: failure-type mix, sample-length spread,
  decontamination rate, and warnings when the set is skewed (one failure type
  dominating), too small, or noisy. Also written to `meta.json`'s `distribution`.
- Pre-run cost/count estimate and a live progress bar during generation.

### Changed

- `convert` now makes **one** LLM call per trace (failure attribution and
  correction combined), roughly halving the per-trace LLM cost.
- Offline (`--no-llm`) `convert` writes raw failed traces to
  `out/needs_review/raw_traces.jsonl` for human curation instead of emitting
  them as `train_sft.jsonl`, and drops the erroring turn so a failed answer is
  never presented as correct.
- Non-UTF-8 consoles (e.g. legacy Windows code pages) now automatically fall
  back to ASCII borders/symbols instead of rendering mojibake.
- `review --kind` is validated against `sft|dpo|both` (invalid values now error
  instead of silently showing nothing).
- Default output directory is `out/` (matches the documented examples).

### Fixed

- Version stamp is consistent across the package, `pyproject.toml`, and
  `meta.json`.

## [0.1.0] — 2026-08-01

First release. A local-first CLI that turns failed agent tool-call / behavior
traces into clean SFT/DPO training data.

### Added

- `trace2train inspect` — instant, rules-only quality report (no LLM, no API
  key): failed / dirty / trainable counts plus a tool-call failure-type
  breakdown.
- `trace2train convert` — turns behavioral failures into LLaMA-Factory-ready
  SFT/DPO JSONL with `_provenance` on every record. Corrects only failures whose
  answer is derivable from the trace; skips (does not fabricate) anything needing
  external verification.
- `trace2train review` — pretty-prints generated samples for eyeballing quality.
- Input importers with auto-detection: LangSmith JSONL, generic `messages`
  JSONL, and Langfuse v4 Public API v2 observation snapshots.
- `trace2train langfuse pull` — downloads a local JSONL snapshot from the
  official Langfuse Cloud / self-hosted **v4** Public API v2 observations
  endpoint (Basic Auth, cursor pagination, bounded 429 retry, atomic write,
  sanitized errors). Validated end to end against a live Langfuse Cloud v4.2.0
  project using seeded synthetic, non-sensitive observations.
- Decontamination: PII redaction, dedup, and eval-set leak filtering.
- `--demo` bundled dataset and a reproducible markdown report
  (`examples/sample_report.md`).
- Helper scripts: `scripts/fetch_dataset.py` (public datasets),
  `scripts/seed_langfuse_validation.py` and
  `scripts/cleanup_langfuse_validation.py` (synthetic Langfuse validation).

### Scope / limitations

- Langfuse: Cloud / self-hosted **v4** Public API v2 observations only. No v3
  legacy APIs, blob-storage exports, UI-download JSON shapes, OpenTelemetry
  ingestion/export, sync/daemon behavior, or write-back.
- Correction is limited to behavioral failures derivable from the trace itself.

[Unreleased]: https://github.com/wane528/trace2train/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wane528/trace2train/releases/tag/v0.1.0
