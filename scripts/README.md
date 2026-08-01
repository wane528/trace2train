# scripts/

Standalone helper scripts (not part of the shipped `trace2train` package).

## fetch_dataset.py — get real tool-call failure traces for validation

Downloads a public agent-trajectory dataset from HuggingFace and converts it to
trace2train's `messages` JSONL format, mapping each dataset's failure signal
onto our `error` field. Files are downloaded whole (not streamed) for
reliability.

### Setup
```bash
pip install datasets            # or: pip install -e ".[data]"
```

### Use
```bash
# AgentForge (default) — multi-turn tool-calling with real error-recovery
# branches. Best fit for trace2train's tool-call/behavior scope.
python scripts/fetch_dataset.py --dataset agentforge --limit 30 --out data/af_traces.jsonl

# ISETrace — real OS-agent trajectories with per-tool success flags
python scripts/fetch_dataset.py --dataset isetrace --limit 20

# terminalbench — coding tasks with reward=0 (needs external verification;
# useful to see convert honestly SKIP what it can't verify)
python scripts/fetch_dataset.py --dataset terminalbench --limit 20
```

### Then
```bash
trace2train inspect data/af_traces.jsonl
trace2train convert data/af_traces.jsonl -o out --max-traces 8
trace2train review -o out
```

### Options
| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `agentforge` | `agentforge`, `isetrace`, or `terminalbench` |
| `--limit` | 300 | max failed rows to keep |
| `--include-success` | off | also keep successful rows (realistic pass/fail ratio) |
| `--scan-cap` | 20000 | safety cap on rows scanned |
| `--out` | `data/real_traces.jsonl` | output path |

### Which dataset shows what

| Dataset | Failure type | Expected convert result |
|---|---|---|
| **agentforge** | tool-call / behavior (in scope) | see `../docs/validation/agentforge.md` for the exact 2026-08-01 slice evidence |
| isetrace | recovery-only | low yield — agent recovered, little to correct |
| terminalbench | needs external verification | mostly `skipped` — honest, not misleading |

Output is git-ignored (`data/`). Never commit downloaded traces.
