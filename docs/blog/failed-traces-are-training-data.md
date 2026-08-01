---
title: "Your agent's failed traces are wasted training data"
published: false
description: "The failures in your LangSmith/Langfuse logs are exactly what you'd want to fine-tune on. I made a small local CLI that turns them into SFT/DPO data — and tries hard not to make up answers it can't verify."
tags: llm, machinelearning, opensource, python
canonical_url:
cover_image:
---

I kept noticing the same thing while building agents: when the agent screws up —
picks the wrong tool, sends bad arguments, over-refuses something harmless — that
failure is genuinely useful. It's a concrete example of my model doing the wrong
thing, which is exactly what I'd want to fine-tune away.

And then it just... sits in the logs. Buried in LangSmith/Langfuse exports, mixed
with retries and PII and duplicates, annoying enough to clean up that I never did.

So I made a small tool for it: [**trace2train**](https://github.com/wane528/trace2train).
It's a local CLI (no server, no account) that turns failed agent traces into SFT/DPO
training data. Sharing it here in case it's useful to anyone else, and because I'd
like feedback.

## The quick version

```console
$ pip install trace2train
$ trace2train inspect --demo

19 traces → 16 failed → 38% dirty (PII/dupes/noise) → 14 trainable

failure types (trainable):
  wrong_tool        8
  bad_args          2
  over_refusal      2
```

`inspect` is just rules — no LLM, no API key. It tells you how much of your log
is actually usable *before* you pay for anything. Then `convert` turns the usable
failures into LLaMA-Factory-ready JSONL.

## A concrete example

Here's a failure from the bundled demo. The agent has a `get_weather` tool but
reaches for `calculator` instead:

```
user:      What's the weather in Berlin?
assistant: [tool_call] calculator(expr="weather Berlin")
tool:      {"error": "invalid expression"}
assistant: I got 42.
```

`wrong_tool` — and the right fix is obvious from the trace (it should have called
`get_weather`). So `convert` can turn it into a training pair:

```json
{
  "conversations": [
    {"from": "human", "value": "What's the weather in Berlin?"},
    {"from": "gpt",   "value": "[tool_call] get_weather(city=\"Berlin\")"}
  ]
}
```

Every record also carries provenance (source trace, original error, why it was
corrected) so the output isn't a black box.

## The one design choice I actually care about

Most "turn your logs into training data" tools are *eager* — they'll happily
generate a confident answer for every failure so the row count looks big.

trace2train doesn't. It only writes a corrected sample when **the right answer is
actually derivable from the trace.** If the failure is "did the code pass its
tests?" or "is this fact true?" — things the trace can't confirm — it marks them
`skipped` instead of inventing something.

The reason is simple: a dataset full of confident-but-wrong "corrections" is worse
than no dataset. I'd rather hand you 7 pairs you can trust and 1 honest skip than
8 rows you can't.

That's the whole personality of the tool, really.

## Other bits that came from actually using it

- PII redaction, dedup, and eval-set leak filtering are built in
- one LLM call per trace (cheap — DeepSeek by default, any OpenAI-compatible API works)
- `--review` to approve/reject each sample by hand before it's written
- `--resume` so a rate-limited re-run doesn't re-pay for calls you already made
- a health check that warns if your set is 90% one failure type (it'll just overfit)

Inputs today: LangSmith JSONL, a generic messages format, and Langfuse v4 snapshots.

## Try it

```bash
pip install trace2train
trace2train inspect --demo               # 30s, no data or key needed
trace2train convert --demo --no-llm -o out
trace2train review -o out
```

## Honest status

It's an early v0.1. I've validated it on a small slice of a public dataset and an
end-to-end Langfuse run on synthetic data — that's it. Not a benchmark, not an
accuracy claim. (Pretending otherwise would kind of defeat the whole "don't make
things up" premise.)

What I'd love feedback on: the failure-detection rules — what am I missing? — and
what trace sources you'd want supported next.

Repo: [github.com/wane528/trace2train](https://github.com/wane528/trace2train) (MIT).
If you try it on real traces, I'd genuinely like to hear how it went.
