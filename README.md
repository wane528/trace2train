# trace2train

**Turn your agent's tool-call & behavior failures into clean SFT/DPO training data.**

Your agent picks the wrong tool, passes bad arguments, returns prose where JSON
was required, over-refuses a benign request, or ignores a system rule.
trace2train turns those failed traces into training data to fix exactly that —
PII scrubbed, dupes removed, full provenance. No server, runs on your laptop.

```console
$ trace2train inspect --demo
┌──────────────────────────── trace2train inspect ────────────────────────────┐
│ 19 traces → 16 failed → 38% dirty (PII/dupes/noise) → 14 trainable         │
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
 failure types (trainable)
┌──────────────────┬───────┐
│ type             │ count │
├──────────────────┼───────┤
│ wrong_tool       │     8 │
│ bad_args         │     2 │
│ over_refusal     │     2 │
│ policy_violation │     1 │
│ lost_context     │     1 │
└──────────────────┴───────┘
```

That report is `inspect` — instant, rules-only, **no LLM and no API key**. It
tells you what kind of failures you have and how much is usable *before* you
spend a cent. Then `convert` turns the usable failures into training data.

## Quick start

```bash
# install (normal use)
pip install .
# or, for development (tests + lint):  pip install -e ".[dev]"

# 1. 30-second demo — no data, no API key needed
trace2train inspect --demo

# 2. convert the demo — offline (no key) writes raw failed traces to
#    out/needs_review/ for human curation, NOT to train_sft.jsonl
trace2train convert --demo --no-llm -o out

# 3. eyeball the result
trace2train review -o out
```

> **Offline vs. LLM.** Without an API key, `convert` can't derive the *corrected*
> answer, so it writes the raw failed traces to `out/needs_review/raw_traces.jsonl`
> for you to hand-fix — it never passes an unverified answer off as training data.
> Set `T2T_LLM_API_KEY` to get corrected `train_sft.jsonl` / `train_dpo.jsonl`.

With your own data:

```bash
# inspect first (free, instant)
trace2train inspect traces.jsonl

# convert with LLM-corrected answers (cheap: defaults to DeepSeek)
cp .env.example .env          # add T2T_LLM_API_KEY if you want LLM correction
trace2train convert traces.jsonl -o out
```

Useful `convert` flags:

- `--review` — approve/reject each corrected sample before it is written
  (human-in-the-loop; needs an LLM). `k`/`d` per sample, `A`/`D` for all rest.
- `--resume` — skip traces already in a previous run's output, so a re-run
  after a rate-limit/interruption doesn't re-pay for the same LLM calls.
- `--json` — emit a machine-readable summary instead of tables (for CI/scripts).
  `inspect --json` is available too.

`convert` makes **one** LLM call per trace (attribution + correction combined),
and prints a cost/count estimate before it starts. After it finishes it prints a
**dataset health** check — failure-type mix, sample-length spread, and warnings
when the set is skewed (one failure type dominating), too small, or noisy.

## Supported inputs

| Input | How to use it | Status |
|---|---|---|
| LangSmith JSONL export | `trace2train inspect traces.jsonl` | Supported |
| Langfuse v4 Public API v2 observation snapshot JSONL | `trace2train langfuse pull snapshot.jsonl` → `trace2train inspect snapshot.jsonl` | Supported |
| Generic messages JSONL | `trace2train inspect messages.jsonl` | Supported |

**Output**: `out/train_sft.jsonl` + `out/train_dpo.jsonl` (LLaMA-Factory
ShareGPT format) + `out/meta.json` audit.

## Langfuse quick start

Langfuse support is a **two-stage snapshot flow**:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
# optional: export LANGFUSE_BASE_URL=https://cloud.langfuse.com

trace2train langfuse pull langfuse_observations.jsonl
trace2train inspect langfuse_observations.jsonl
trace2train convert langfuse_observations.jsonl --no-llm -o out
```

Compatibility boundary for this release:

- **Included:** Langfuse Cloud and self-hosted **v4** via the official Public
  API v2 observations endpoint.
- **Excluded:** v3 legacy APIs, blob-storage exports, UI-download JSON shapes,
  OpenTelemetry ingestion/export, sync/daemon behavior, and write-back.

### Langfuse status: Supported

The Langfuse importer and `trace2train langfuse pull` CLI are validated end to
end against a live Langfuse Cloud **v4.2.0** project via the official Public API
v2 observations endpoint, using **seeded synthetic, non-sensitive** observations
(not real customer data): authentication, cursor pagination, atomic snapshot,
auto-detection, trace grouping, failure detection, and offline (`--no-llm`)
conversion with correct provenance all worked — 10 observations → 3 traces → 2
trainable failures → **2 SFT records (0 DPO under `--no-llm`)**. Sanitized
aggregate evidence (no content or credentials):
[`docs/validation/langfuse-cloud-v4.md`](docs/validation/langfuse-cloud-v4.md).

The `scripts/seed_langfuse_validation.py` helper can seed a fresh synthetic,
non-sensitive validation project to reproduce this.

### Raw snapshot privacy warning

`trace2train langfuse pull` writes the raw observation payloads to a local JSONL
snapshot. That snapshot may contain prompt content, tool arguments, outputs, and
other sensitive trace data. Redaction happens later during `convert`, **not**
during snapshot creation. Review, store, and share the raw snapshot accordingly.

## How it works

```
traces (LangSmith, Langfuse snapshot, or messages JSONL)
   │
   ├─▶ inspect ──▶ rules-only quality report        (free, instant, no LLM)
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

## Scope — what it fixes, and what it deliberately won't

trace2train corrects **behavioral failures** whose right answer is derivable
from the trace itself:

- ✅ wrong tool chosen · bad arguments · lost context · wrong output format ·
  plainly wrong common-sense answers · over-refusals

It **skips** (and tells you) failures whose correctness needs external ground
truth, instead of fabricating an answer:

- ❌ "did the code pass its tests?" · "is this fact accurate?" · "did the task
  really complete?" — the trace alone can't say, so these are reported as
  `skipped`, not turned into misleading data.

> This honesty is the point: a training set full of plausible-but-wrong
> "corrections" is worse than no training set. Ground-truth-assisted correction
> is planned as a future feature.

## Evidence

Small public validation evidence is documented in
[`docs/validation/agentforge.md`](docs/validation/agentforge.md):

- AgentForge slice date: **2026-08-01**
- `8` trainable traces processed
- `7` SFT records + `7` DPO records emitted
- `1` `skipped_uncertain`
- pair yield: `7/8 = 87.5%`

That note is deliberately small-scope evidence, **not** a general accuracy or
production guarantee.

## Artifacts

- Reproducible demo markdown report:
  [`examples/sample_report.md`](examples/sample_report.md)
- Terminal screenshot status / manual capture instructions:
  [`docs/assets/README.md`](docs/assets/README.md)

The README does **not** embed a PNG because no faithful automated terminal
screenshot was produced in this environment.

On a non-UTF-8 Windows console, trace2train **automatically** falls back to
ASCII borders and symbols, so the tables stay readable out of the box. If you
prefer the Unicode box-drawing output, run under UTF-8 with
`python -X utf8 -m trace2train.cli ...`. The standard installed `trace2train ...`
command is the primary path.

## Design principles

- **Local-first CLI.** No server, no account, no telemetry.
- **Honest over eager.** Producing no data beats producing misleading data.
- **Auditable.** Every record carries `_provenance` (source trace, run id,
  original error, attribution).
- **Cheap by default.** DeepSeek costs fractions of a cent per trace; swap
  `base_url`/`model` for any OpenAI-compatible provider.

## Getting real traces to try

No traces of your own yet? `scripts/fetch_dataset.py` pulls public agent
trajectory datasets from HuggingFace and converts them to trace2train's format.
See [`scripts/README.md`](scripts/README.md).

## License

MIT
