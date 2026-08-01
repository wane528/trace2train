# trace2train

> **Turn your agent's tool-call & behavior failures into clean SFT/DPO training data.**

[![PyPI](https://img.shields.io/pypi/v/trace2train)](https://pypi.org/project/trace2train/)
[![CI](https://github.com/wane528/trace2train/actions/workflows/ci.yml/badge.svg)](https://github.com/wane528/trace2train/actions/workflows/ci.yml)
![python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![license: MIT](https://img.shields.io/badge/license-MIT-green)

**English · [简体中文](README.zh-CN.md)**

Your agent picks the wrong tool, passes bad arguments, returns prose where JSON
was required, over-refuses a benign request, or ignores a system rule.
trace2train turns those failed traces into training data to fix exactly that —
PII scrubbed, dupes removed, full provenance. **No server, runs on your laptop.**

```console
$ trace2train inspect --demo
┌──────────────────────────── trace2train inspect ────────────────────────────┐
│ 19 traces → 16 failed → 38% dirty (PII/dupes/noise) → 14 trainable          │
└─────────────────────────────────────────────────────────────────────────────┘
┌───────────────────────────────────┬───────┐
│ Metric                            │ Count │
├───────────────────────────────────┼───────┤
│ traces                            │    19 │
│ failed                            │    16 │
│   ↳ environmental (not trainable) │     2 │
│   ↳ with PII                      │     3 │
│   ↳ duplicates                    │     1 │
│ trainable                         │    14 │
│ SFT candidates (upper bound)      │    14 │
│ DPO candidates (upper bound)      │     7 │
└───────────────────────────────────┴───────┘
```

`inspect` is instant, rules-only, **no LLM and no API key** — it tells you what
kind of failures you have and how much is usable *before* you spend a cent.

## Contents

- [Features](#features)
- [Install](#install)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Configuration](#configuration)
- [Supported inputs](#supported-inputs)
- [How it works](#how-it-works)
- [Scope](#scope--what-it-fixes-and-what-it-wont)
- [Langfuse](#langfuse)
- [Design principles](#design-principles)
- [FAQ](#faq)
- [Contributing & license](#contributing--license)

## Features

- **Free quality report** — `inspect` scores your traces with pure rules: how
  many failed, how dirty (PII / dupes / noise), how many are trainable, and a
  breakdown by failure type. No LLM, no key.
- **Honest correction** — `convert` only emits data when the fix is derivable
  from the trace; failures needing external ground truth are reported as
  `skipped`, never fabricated.
- **One LLM call per trace** — attribution *and* correction happen in a single
  call, roughly halving cost. A cost estimate prints before it runs.
- **Human-in-the-loop** — `convert --review` lets you approve/reject each
  sample before it's written.
- **Resumable** — `convert --resume` skips traces already processed, so a re-run
  after a rate-limit doesn't re-pay for the same calls.
- **Dataset health check** — after `convert`, see the failure-type mix, length
  spread, and warnings when the set is skewed / too small / noisy.
- **Auditable output** — LLaMA-Factory-ready JSONL with `_provenance` on every
  record, plus a `meta.json` audit.
- **Scriptable** — `--json` on `inspect` and `convert` for CI/pipelines.
- **Local-first** — no server, no account, no telemetry.

## Install

Requires Python 3.11+.

```bash
pip install trace2train       # normal use
pip install -e ".[dev]"       # development (tests + lint)
```

## Quick start

**Try it in 30 seconds** — no data, no API key:

```bash
# 1. instant quality report on bundled sample data
trace2train inspect --demo

# 2. convert it (offline: writes raw traces to out/needs_review/ for curation)
trace2train convert --demo --no-llm -o out

# 3. eyeball the result
trace2train review -o out
```

**With your own data and an LLM** (recommended — this is where the corrected
training data comes from):

```bash
cp .env.example .env          # add T2T_LLM_API_KEY (DeepSeek is cheap)

trace2train inspect traces.jsonl          # free, instant
trace2train convert traces.jsonl -o out   # LLM-corrected SFT/DPO
```

> **Offline vs. LLM.** Without an API key, `convert` can't derive the *corrected*
> answer, so it writes the raw failed traces to `out/needs_review/raw_traces.jsonl`
> for you to hand-fix — it never passes an unverified answer off as training data.
> Set `T2T_LLM_API_KEY` to get corrected `train_sft.jsonl` / `train_dpo.jsonl`.

## Commands

| Command | What it does |
|---|---|
| `trace2train inspect [FILE]` | Instant, rules-only quality report (no LLM). |
| `trace2train convert [FILE]` | Turn failures into LLaMA-Factory SFT/DPO JSONL. |
| `trace2train review` | Pretty-print generated samples to judge quality by eye. |
| `trace2train langfuse pull [OUT]` | Snapshot Langfuse v4 observations to local JSONL. |
| `trace2train --version` | Print the installed version. |

Add `--help` to any command for its full options. Key flags:

**`inspect`**

| Flag | Purpose |
|---|---|
| `--demo` | Use the bundled sample dataset. |
| `--format auto\|langsmith\|langfuse\|messages` | Force the input format (default: auto-detect). |
| `--export PATH` | Also write a shareable Markdown report. |
| `--json` | Emit a machine-readable JSON report instead of tables. |

**`convert`**

| Flag | Purpose |
|---|---|
| `-o, --out-dir PATH` | Output directory (default: `out`). |
| `--no-llm` | Run without an LLM (raw traces → `needs_review/`). |
| `--review` | Approve/reject each sample before writing (needs an LLM). |
| `--resume` | Skip traces already in a previous run's output. |
| `--redact / --no-redact` | PII redaction (default: on). |
| `--leak-file PATH` | Exclude samples matching eval-set fingerprints. |
| `--max-traces N` | Limit how many traces are processed. |
| `--json` | Emit a machine-readable JSON summary. |

In `--review`, use `k`/`d` to keep/drop a sample and `A`/`D` to apply to all
remaining.

**`review`**

| Flag | Purpose |
|---|---|
| `-o, --out-dir PATH` | Where `convert` wrote its files (default: `out`). |
| `-n, --limit N` | How many samples to show (default: 5). |
| `--kind sft\|dpo\|both` | Which records to show (default: both). |

## Configuration

Set these in a `.env` file (copy `.env.example`) or as environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `T2T_LLM_API_KEY` | Enables LLM-corrected `convert`. Without it, `convert` runs offline. | *(none)* |
| `T2T_LLM_BASE_URL` | OpenAI-compatible endpoint. | `https://api.deepseek.com` |
| `T2T_LLM_MODEL` | Model name. | `deepseek-chat` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Auth for `langfuse pull`. | *(none)* |
| `LANGFUSE_BASE_URL` | Langfuse host. | `https://cloud.langfuse.com` |

Any OpenAI-compatible provider (OpenAI, Moonshot, Qwen, …) works by swapping
`T2T_LLM_BASE_URL` / `T2T_LLM_MODEL`.

## Supported inputs

| Input | How to use it | Status |
|---|---|---|
| LangSmith JSONL export | `trace2train inspect traces.jsonl` | Supported |
| Generic messages JSONL | `trace2train inspect messages.jsonl` | Supported |
| Langfuse v4 Public API v2 snapshot | `trace2train langfuse pull` → `inspect` | Supported |

**Output** (in `out/`): `train_sft.jsonl` + `train_dpo.jsonl` (LLaMA-Factory
ShareGPT format) + `meta.json` audit. No traces of your own yet?
`scripts/fetch_dataset.py` pulls public agent-trajectory datasets from
HuggingFace — see [`scripts/README.md`](scripts/README.md).

## How it works

```
traces (LangSmith, Langfuse snapshot, or messages JSONL)
   │
   ├─▶ inspect ──▶ rules-only quality report          (free, instant, no LLM)
   │
   └─▶ convert
         ① detect failures (rules)
         ② attribute + correct in ONE LLM call — why it failed AND the fix,
            only when that fix is derivable from the trace
         ③ decontaminate: PII redact · dedup · leak filter
         ④ generate SFT + DPO
         ⑤ export LLaMA-Factory JSONL + provenance + meta.json
         ⑥ dataset health check — failure-type mix, length spread, skew warnings
```

## Scope — what it fixes, and what it won't

trace2train corrects **behavioral failures** whose right answer is derivable
from the trace itself:

- ✅ wrong tool · bad arguments · lost context · wrong output format · plainly
  wrong common-sense answers · over-refusals

It **skips** (and tells you) failures whose correctness needs external ground
truth, instead of fabricating an answer:

- ❌ *"did the code pass its tests?"* · *"is this fact accurate?"* · *"did the
  task really complete?"* — the trace alone can't say, so these are reported as
  `skipped`.

> This honesty is the point: a training set full of plausible-but-wrong
> corrections is worse than no training set. Ground-truth-assisted correction is
> planned as a future feature.

## Langfuse

Langfuse support is a **two-stage snapshot flow** — pull a local snapshot, then
inspect/convert it like any other input:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

trace2train langfuse pull langfuse_observations.jsonl
trace2train inspect langfuse_observations.jsonl
trace2train convert langfuse_observations.jsonl -o out
```

**Included:** Langfuse Cloud and self-hosted **v4** via the official Public API
v2 observations endpoint. **Excluded:** v3 legacy APIs, blob-storage exports,
UI-download JSON shapes, OpenTelemetry, sync/daemon behavior, and write-back.

> ⚠️ **Privacy:** the pulled snapshot contains raw prompt content, tool
> arguments, and outputs. Redaction happens during `convert`, **not** at pull
> time — store and share the snapshot accordingly.

Validated end to end against a live Langfuse Cloud **v4.2.0** project using
seeded synthetic, non-sensitive observations. Details and sanitized evidence:
[`docs/validation/langfuse-cloud-v4.md`](docs/validation/langfuse-cloud-v4.md).
Additional public validation evidence:
[`docs/validation/agentforge.md`](docs/validation/agentforge.md).

## Design principles

- **Local-first CLI.** No server, no account, no telemetry.
- **Honest over eager.** Producing no data beats producing misleading data.
- **Auditable.** Every record carries `_provenance` (source trace, run id,
  original error, attribution).
- **Cheap by default.** DeepSeek costs fractions of a cent per trace; swap
  `base_url`/`model` for any OpenAI-compatible provider.

<sub>On a non-UTF-8 Windows console, output automatically falls back to ASCII
borders/symbols. For Unicode box-drawing, run under UTF-8:
`python -X utf8 -m trace2train.cli ...`.</sub>

## FAQ

**How do I turn my agent's failures into fine-tuning data?**
Point trace2train at your trace export: `trace2train convert traces.jsonl -o out`.
It detects the behavioral failures (wrong tool, bad args, over-refusal, …) and
writes SFT/DPO JSONL you can feed to a trainer.

**Can I convert LangSmith / Langfuse logs into an SFT or DPO dataset?**
Yes. LangSmith JSONL exports work directly. For Langfuse, `trace2train langfuse
pull` snapshots your v4 observations first, then `convert` them. See
[Supported inputs](#supported-inputs).

**How do I build a tool-calling / function-calling fine-tuning dataset from agent traces?**
That's the core use case — trace2train specializes in tool-call and agent-behavior
failures (wrong tool chosen, malformed arguments, lost context, wrong output
format, over-refusals) and produces training pairs that correct exactly those.

**Does it work offline / without an API key?**
`inspect` is fully offline (rules only). `convert` needs an LLM to derive the
corrected answer; without a key it writes the raw failed traces to
`needs_review/` for you to hand-fix instead of fabricating data.

**What format is the output, and can I use it with LLaMA-Factory?**
Output is LLaMA-Factory-ready ShareGPT JSONL (`train_sft.jsonl` /
`train_dpo.jsonl`) plus a `meta.json` audit.

**How is this different from just dumping my logs into a trainer?**
It decontaminates (PII redaction, dedup, eval-set leak filtering) and, crucially,
only emits data when the fix is derivable from the trace — failures needing
external ground truth are skipped, not fabricated.

## Contributing & license

Contributions that keep trace2train truthful, well-tested, and focused on
tool-call / agent-behavior failures are welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md). Released under the [MIT](LICENSE) license.
