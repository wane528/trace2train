# AgentForge validation note

Date: 2026-08-01

This is a **small reproducibility note**, not an accuracy or generalization
claim.

## Source

The public dataset source and fetch workflow are already documented in
[`scripts/README.md`](../../scripts/README.md).

## Exact evidence used

This repo currently has an ignored local output file at `out_af/meta.json` with
the following counts.

Because the dataset is external and may change over time, and because this repo
does not pin a frozen downloaded copy, these exact counts come from that local
ignored artifact rather than from a guaranteed clean-checkout replay.

- total traces: `8`
- failed traces: `8`
- trainable traces: `8`
- SFT records: `7`
- DPO records: `7`
- skipped_uncertain: `1`

Derived pair yield for this slice:

- `7 / 8 = 87.5%`

Restated succinctly:

- `8` trainable traces -> `7` SFT + `7` DPO
- `1` `skipped_uncertain`

## Reproduction commands

```bash
python scripts/fetch_dataset.py --dataset agentforge --limit 30 --out data/af_traces.jsonl
trace2train inspect data/af_traces.jsonl
trace2train convert data/af_traces.jsonl -o out_af --max-traces 8
trace2train review -o out_af
```

## Interpretation

This is evidence for the narrow in-scope scenario: tool-call / agent behavior
failures where the fix is derivable from the trace.

It is **not**:

- a production benchmark,
- a live Langfuse validation,
- an accuracy guarantee,
- or a claim that all datasets will yield similar SFT/DPO rates.
