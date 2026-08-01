# Launch drafts

Copy-paste starting points for launching trace2train. The current reproducible
artifact is the real markdown export in `examples/sample_report.md`; the terminal
PNG remains a manual follow-up.

---

## Show HN

**Title:**
> Show HN: trace2train – turn failed agent traces into fine-tuning data

**Body:**

I kept hitting the same wall: I had piles of agent traces where the model failed,
but the raw data was too dirty to train on directly (PII, retries, environment
noise, duplicate runs, and plenty of failures where the trace itself still isn't
enough to know the right answer).

So I built trace2train, a local CLI with two stages:

1. `inspect` — instant, rules-only, no LLM, no API key. It tells you what kind
   of failures you actually have before you spend anything.
2. `convert` — turns the usable behavioral failures into SFT/DPO JSONL with
   provenance, while skipping anything that would need external verification.

The current demo output is:

```text
19 traces → 16 failed → 38% dirty (PII/dupes/noise) → 14 trainable
wrong_tool 8, bad_args 2, over_refusal 2, policy_violation 1, lost_context 1
```

That first command is the hook. It is basically a quality report for your trace
pile.

The scope is deliberately narrow: trace2train only emits corrected data when the
fix is **derivable from the trace itself**. Wrong tool? Bad args? Lost context?
Wrong format? Over-refusal? Great. But if the trace only says "tests passed" or
asserts a fact without proof, it skips that sample rather than inventing a
plausible answer.

I also added a Langfuse flow: `trace2train langfuse pull` downloads a local
snapshot from the Langfuse Cloud/self-hosted v4 Public API v2, then you run the
same `inspect` / `convert` commands on that local file. It is validated end to
end against a live Langfuse Cloud v4 project using seeded synthetic,
non-sensitive observations (10 observations -> 3 traces -> 2 SFT, 0 DPO under
`--no-llm`, with correct provenance). It is explicitly Cloud/self-hosted v4
Public API v2 only — no v3 legacy, blob exports, UI-download guessing, OTel,
sync, or write-back.

Small public validation note: on a documented AgentForge slice,
`8 trainable traces -> 7 SFT + 7 DPO, 1 skipped_uncertain` (87.5% pair yield).
That is a tiny reproducibility check for the in-scope behavior-fix scenario, not
a general accuracy claim.

30-second demo:

```bash
pip install -e ".[dev]"
trace2train inspect --demo
trace2train convert --demo --no-llm -o out
```

Repo: https://github.com/wane528/trace2train

---

## Reddit — r/LocalLLaMA

**Title:**
> I built a local CLI that turns failed agent traces into clean SFT/DPO data

**Body:**

If you're fine-tuning small models on your own agent's failures, you've probably
hit the "raw traces are too dirty to train on" problem.

trace2train is a local CLI for that:

- `inspect` gives an instant rules-only report: current demo is
  `19 traces -> 16 failed -> 38% dirty -> 14 trainable`
- failure breakdown on the demo is:
  `wrong_tool 8, bad_args 2, over_refusal 2, policy_violation 1, lost_context 1`
- `convert` exports LLaMA-Factory-ready JSONL with provenance on every row

The main design choice is the guardrail: it only emits corrected data when the
right answer is derivable from the trace. If a failure needs external
verification (tests/fact-checking/real task completion), it gets skipped instead
of turned into misleading training data.

There is also a Langfuse path (validated end to end on a real Langfuse Cloud v4
project):

```bash
trace2train langfuse pull langfuse_observations.jsonl
trace2train inspect langfuse_observations.jsonl
trace2train convert langfuse_observations.jsonl --no-llm -o out
```

That flow is currently limited to Langfuse Cloud/self-hosted v4 Public API v2
snapshots only. No legacy v3/blob/UI-download/OTel/write-back claims.

Small public validation note: on a documented AgentForge slice,
`8 trainable -> 7 SFT + 7 DPO, 1 skipped_uncertain` (87.5% pair yield). That's a
small reproducibility check, not a broad benchmark promise.

30-second demo:

```bash
pip install -e ".[dev]"
trace2train inspect --demo
trace2train convert --demo --no-llm -o out
trace2train review -o out
```

Repo: https://github.com/wane528/trace2train

---

## Pre-launch checklist

- [ ] Push to GitHub (public), confirm `pip install` works from a clean clone
- [ ] Manual terminal PNG captured faithfully (100x30 target) or launch without it
- [x] `examples/sample_report.md` exported from a real `inspect --demo` run
- [x] Real Langfuse Cloud v4 validation PASSED with actual credentials, using seeded synthetic non-sensitive observations: non-empty v2 pull->inspect->convert on 10 observations -> 3 traces -> 2 SFT (0 DPO under --no-llm) with correct provenance (see docs/validation/langfuse-cloud-v4.md); Langfuse promoted to Supported
- [x] README top block matches current demo values and compatibility wording
- [x] `pytest` green (110 tests), `ruff check` clean, `python -m build` + clean-venv wheel install smoke checks completed
- [x] LICENSE present (MIT), `pyproject` author/URL filled in (wane528 / github.com/wane528/trace2train)
- [ ] Post to Show HN in the morning (US ET); cross-post r/LocalLLaMA + r/LLMDevs
- [ ] Reply to every comment for the first few hours
