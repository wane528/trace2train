---
title: "Your agent's failed traces are wasted fine-tuning data"
published: false
description: "Wrong tool, bad args, over-refusals — the failures buried in your LangSmith/Langfuse exports are exactly what you'd want to fine-tune on. Here's a small local CLI that turns them into clean SFT/DPO data, and why 'honest over eager' matters."
tags: llm, machinelearning, opensource, python
canonical_url:
cover_image:
---

Every agent you run produces a pile of traces. A meaningful chunk of them are
failures: the model picks the wrong tool, passes malformed arguments, returns
prose where JSON was required, over-refuses a benign request, or quietly drops
context it was given earlier.

Here's the thing most teams miss: **those failures are the single most valuable
fine-tuning signal you own.** They are concrete, in-domain examples of your model
doing the wrong thing — the exact behavior you'd want to train away. And almost
everyone throws them out.

They get thrown out because they're *stuck*. They live inside LangSmith or
Langfuse exports, tangled up with retries, cache hits, environmental timeouts,
PII, and near-duplicate runs. Turning that raw sludge into a clean SFT/DPO
dataset is annoying enough that it rarely happens. So the gold sits in the tailings.

I built a small tool to fix exactly this: **[trace2train](https://github.com/wane528/trace2train)**
— a local-first CLI that turns failed agent traces into clean SFT/DPO training
data. No server, no account, no telemetry. This post is about the problem and the
design decisions, not a feature dump.

## See it before you commit a cent

The first command, `inspect`, is pure rules — **no LLM, no API key**. It just
tells you what you're sitting on:

```console
$ pip install trace2train
$ trace2train inspect --demo

19 traces → 16 failed → 38% dirty (PII/dupes/noise) → 14 trainable

failure types (trainable):
  wrong_tool        8
  bad_args          2
  over_refusal      2
  policy_violation  1
  lost_context      1
```

That's the whole pitch of `inspect`: *before* you spend anything on an LLM pass,
you know how many traces failed, how much is dirty, how many are actually
trainable, and what kind of failures dominate. Run it on the bundled demo in 30
seconds — no data of your own required.

Then `convert` turns the usable failures into LLaMA-Factory-ready SFT/DPO JSONL,
with full provenance on every record.

## Why not just dump the failures into a trainer?

Because raw failed traces are *dirty*, and dirty training data actively hurts.
Two categories of problem:

1. **Noise and leakage.** Retries, cache hits, PII in prompts, near-duplicates,
   and — worst of all — samples that overlap your eval set. Train on those and
   you get a model that's memorized its own benchmark.
2. **Not every failure is trainable.** A timeout or a 429 isn't a behavior the
   model can learn to avoid. Neither is a failure whose "correct answer" you
   can't actually determine from the trace.

trace2train handles the first with built-in PII redaction, dedup, and eval-set
leak filtering. The second is where the interesting design decision lives.

## The design decision that matters: honest over eager

Here's the part I care about most, and the reason I think this tool is worth
using over a hand-rolled script.

**trace2train only emits a corrected training sample when the correct answer is
derivable from the trace itself.**

If the agent picked `calculator` for a weather question, the fix is obvious from
context — that's trainable. But if the failure is *"did the code actually pass
its tests?"* or *"is this claimed fact accurate?"* or *"did the task really
complete?"*, the trace alone can't tell you. So trace2train reports those as
`skipped` — it does **not** invent a plausible-looking answer.

This sounds like a limitation. It's actually the point.

> A training set full of plausible-but-wrong "corrections" is worse than no
> training set at all. Confidently wrong labels are the fastest way to make a
> fine-tune worse than the base model.

Most "turn your logs into training data" pipelines are *eager*: they'll happily
hallucinate a correct answer for every failure so the output row count looks
impressive. trace2train is deliberately the opposite. It would rather hand you 7
trustworthy pairs and 1 honest `skipped` than 8 rows you can't trust.

## What a run looks like

With an API key set (DeepSeek by default — cents, and any OpenAI-compatible
provider works), `convert` makes **one** LLM call per trace that does both the
failure attribution *and* the correction, and prints a cost estimate before it
starts. A few of the touches that came out of actually using it:

- **`--review`** — approve or reject each corrected sample before it's written.
  Human-in-the-loop when you don't fully trust the model's fix yet.
- **`--resume`** — if a run dies halfway (rate limit, Ctrl-C), the re-run skips
  traces you already processed so you don't re-pay for the same calls.
- **A dataset health check** after conversion: it warns when your set is skewed
  toward one failure type (e.g. 90% `wrong_tool`), too small to fine-tune on, or
  noisy. Because a dataset that's all one failure mode will just overfit that one
  behavior.
- **Full provenance** on every record — source trace, run id, original error,
  attribution — so the output is auditable, not a black box.

Inputs today: LangSmith JSONL exports, a generic messages-JSONL shape (anything
that can dump conversations), and Langfuse v4 Public API v2 snapshots.

## Try it

```bash
pip install trace2train

# 30-second demo, no data or key needed
trace2train inspect --demo

# convert the demo (offline mode writes raw traces for you to hand-curate)
trace2train convert --demo --no-llm -o out
trace2train review -o out
```

With your own data and an LLM key, `convert traces.jsonl -o out` gives you the
corrected `train_sft.jsonl` / `train_dpo.jsonl`.

## Honest status

This is an early **v0.1 / MVP**. The public validation so far is deliberately
small-scope: a handful of trainable traces from a public dataset, plus an
end-to-end Langfuse Cloud v4 run against synthetic, non-sensitive data. It is
**not** a production benchmark or an accuracy guarantee, and I'm not going to
pretend otherwise — that'd rather defeat the "honest over eager" premise.

What I'd genuinely value feedback on:

- the failure-detection rules — what failure modes am I missing?
- what trace sources you'd want supported next
- whether the "skip instead of fabricate" boundary matches your intuition

Repo, install, and full docs: **https://github.com/wane528/trace2train** (MIT).
If the idea resonates or you try it on real traces, I'd love to hear how it went.
