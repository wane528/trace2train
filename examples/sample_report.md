# trace2train inspect report

> 19 traces → 16 failed → 38% dirty (PII/dupes/noise) → 14 trainable

| Metric | Count |
|---|---:|
| traces | 19 |
| failed | 16 |
| environmental (not trainable) | 2 |
| with PII | 3 |
| duplicates | 1 |
| trainable | 14 |
| SFT candidates (upper bound) | 14 |
| DPO candidates (upper bound) | 7 |

_candidates = upper bound; `convert` only emits data where the correct answer is derivable from the trace (behavioral fixes)._

## Failure types (trainable)

| Type | Count |
|---|---:|
| wrong_tool | 8 |
| bad_args | 2 |
| over_refusal | 2 |
| policy_violation | 1 |
| lost_context | 1 |
