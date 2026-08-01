# HuggingFace forum post — Show and Tell

**Suggested category:** Show and Tell (fall back to Datasets / general if unavailable)

**Title:**

> trace2train — turning failed agent traces into SFT/DPO data (without making up answers)

**Body:**

---

Hi all,

While building tool-using agents I kept running into the same waste: a good chunk
of every run is *failures* — the model picks the wrong tool, sends malformed
arguments, over-refuses a harmless request. Those failures are exactly the kind of
behavior I'd want to fine-tune away, but they just sit in LangSmith/Langfuse
exports, mixed with retries, PII, and duplicates.

So I built a small local CLI to turn them into training data:
**trace2train** — https://github.com/wane528/trace2train (`pip install trace2train`, MIT).

**What it does**

Two main commands:
- `inspect` — rules-only, no LLM, no API key. Tells you how many traces failed,
  how dirty they are (PII/dupes/noise), and how many are actually trainable,
  before you spend anything.
- `convert` — turns the usable failures into LLaMA-Factory-ready SFT/DPO JSONL,
  with PII redaction, dedup, eval-set leak filtering, and full provenance on
  every record.

**A concrete example**

A failure from the bundled demo — the agent has a `get_weather` tool but reaches
for `calculator`:

```
user:      What's the weather in Berlin?
assistant: [tool_call] calculator(expr="weather Berlin")
tool:      {"error": "invalid expression"}
assistant: I got 42.
```

The right fix is obvious from the trace, so `convert` turns it into a training pair:

```json
{
  "conversations": [
    {"from": "human", "value": "What's the weather in Berlin?"},
    {"from": "gpt",   "value": "[tool_call] get_weather(city=\"Berlin\")"}
  ]
}
```

**The design choice I'd most like feedback on**

It only writes a corrected sample when the right answer is *derivable from the
trace itself*. If the failure needs external ground truth ("did the code pass its
tests?", "is this fact accurate?"), it marks the sample `skipped` instead of
inventing an answer. My reasoning: a dataset full of confident-but-wrong
"corrections" is worse than no dataset. But I'm genuinely unsure where exactly
that line should sit — curious what people here think.

**Honest status**

Early v0.1. Validated only on a small slice of a public dataset and an end-to-end
Langfuse v4 run on synthetic data — not a benchmark, not an accuracy claim.
Inputs today: LangSmith JSONL, a generic messages JSONL, and Langfuse v4 snapshots.

**Try it (30s, no data or key needed):**

```
pip install trace2train
trace2train inspect --demo
```

Questions I'd love input on:
1. The failure-detection rules — what failure modes am I missing?
2. What trace sources would you want supported next?
3. Does the "skip instead of fabricate" boundary match your intuition?

Thanks for reading — happy to answer anything.
