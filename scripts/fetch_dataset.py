"""Fetch a public agent-trajectory dataset and convert it to trace2train's
`messages` JSONL format for real-data validation (M4).

Standalone script — depends only on `datasets` (HuggingFace), not on the
trace2train package. Install with:

    pip install datasets            # or: pip install -e ".[data]"

Usage:

    python scripts/fetch_dataset.py --dataset isetrace --limit 300 --out data/real_traces.jsonl
    python scripts/fetch_dataset.py --dataset terminalbench --limit 200

Then:

    trace2train inspect data/real_traces.jsonl
    trace2train convert data/real_traces.jsonl -o out

Output format (one JSON object per line), matching trace2train's `messages`
importer:

    {"messages": [{"role": "user"|"assistant"|"tool", "content": "..."}],
     "error": "<why it failed>" | null,
     "id": "<source id>",
     "source_dataset": "<name>"}

We keep BOTH failed and (optionally) successful rows. trace2train only turns
FAILED traces into training data, but keeping successes lets `inspect` show a
realistic pass/fail ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

# ---------------------------------------------------------------------------
# Adapters: each yields normalized rows {"messages", "error", "id"} from a
# streamed HuggingFace dataset. They are defensive: malformed rows are skipped.
# ---------------------------------------------------------------------------


def _norm_messages(raw) -> list[dict]:
    """Coerce a dataset's messages into [{'role','content'}] with string content."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("from")
        content = m.get("content")
        if content is None:
            content = m.get("value")
        if role is None:
            continue
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False) if content is not None else ""
        out.append({"role": str(role), "content": content})
    return out


def adapt_isetrace(row: dict) -> dict | None:
    """valiere/ISETrace: OpenAI-format messages with per-tool `success` flags.

    A trajectory is 'failed' if any tool call errored (success is False) — real
    execution errors like "This operation was aborted". This is authentic,
    model-generated failure data (ideal for M4).
    """
    raw = row.get("messages") or []
    msgs = _norm_messages(raw)
    if not msgs:
        return None
    # find the first failed tool call in the raw (untruncated) messages
    error = None
    for m in raw:
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("success") is False:
            tool = m.get("name") or "tool"
            content = str(m.get("content") or "")[:200]
            error = f"{tool} failed: {content}"
            break
    return {
        "messages": msgs,
        "error": error,
        "id": str(row.get("session_id") or ""),
    }


def adapt_terminalbench(row: dict) -> dict | None:
    """yoonholee/terminalbench-trajectories: `reward` (1 solved/0 not) + `steps`
    (JSON string of {src,msg,tools,obs}). We flatten steps into messages."""
    steps_raw = row.get("steps")
    if isinstance(steps_raw, str):
        try:
            steps = json.loads(steps_raw)
        except json.JSONDecodeError:
            return None
    else:
        steps = steps_raw
    if not isinstance(steps, list) or not steps:
        return None

    role_map = {"user": "user", "agent": "assistant", "system": "system"}
    msgs: list[dict] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        role = role_map.get(str(st.get("src")), "assistant")
        text = str(st.get("msg") or "")
        tools = st.get("tools")
        if tools:
            text += "\n[tools] " + json.dumps(tools, ensure_ascii=False)[:400]
        if text.strip():
            msgs.append({"role": role, "content": text})
        obs = st.get("obs")
        if obs:
            msgs.append({"role": "tool", "content": str(obs)[:600]})
    if not msgs:
        return None

    reward = row.get("reward")
    failed = reward is not None and int(reward) == 0
    return {
        "messages": msgs,
        "error": f"task not solved (reward={reward})" if failed else None,
        "id": str(row.get("trial_id") or row.get("task_name") or ""),
    }


def adapt_agentforge(row: dict) -> dict | None:
    """voxozi/agentforge-multiturn-toolcall: multi-turn tool-calling with a
    30.5% slice of genuine error-recovery branches (includes_recovery flag).

    conversations are OpenAI-format (role/content, assistant carries tool_calls).
    We fold assistant tool_calls into the text so the trace is self-describing,
    and set `error` from the first failed tool result when the trajectory
    includes a recovery branch.
    """
    convs = row.get("conversations")
    if isinstance(convs, str):
        try:
            convs = json.loads(convs)
        except json.JSONDecodeError:
            return None
    if not isinstance(convs, list) or not convs:
        return None

    msgs: list[dict] = []
    first_error: str | None = None
    for m in convs:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "assistant"
        content = str(m.get("content") or "")
        # fold tool_calls into the assistant text so the call is visible
        tcs = m.get("tool_calls")
        if tcs:
            content += "\n[tool_calls] " + json.dumps(tcs, ensure_ascii=False)[:600]
        # detect a failed tool result
        if role == "tool":
            low = content.lower()
            if first_error is None and ('"status": "failed"' in low or '"error"' in low
                                        or "failed" in low):
                first_error = f"tool '{m.get('name', '?')}' returned failure: {content[:200]}"
        if content.strip():
            msgs.append({"role": role, "content": content})

    if not msgs:
        return None

    failed = bool(row.get("includes_recovery")) and first_error is not None
    return {
        "messages": msgs,
        "error": first_error if failed else None,
        "id": str(row.get("id") or ""),
    }


# Each entry: (repo_id, [files to download in order], adapter).
# We download files directly (hf_hub_download) and read them locally, which is
# far more reliable than `load_dataset(..., streaming=True)` on Windows.
ADAPTERS = {
    "agentforge": (
        "voxozi/agentforge-multiturn-toolcall",
        ["train.jsonl"],
        adapt_agentforge,
    ),
    "isetrace": (
        "valiere/ISETrace",
        ["trajectories/trajectories-00000.jsonl"],
        adapt_isetrace,
    ),
    "terminalbench": (
        "yoonholee/terminalbench-trajectories",
        ["data/train-00000-of-00002.parquet"],
        adapt_terminalbench,
    ),
}


# ---------------------------------------------------------------------------


def iter_rows(dataset_id: str, files: list[str]) -> Iterator[dict]:
    """Download each file from the HF dataset repo and yield rows locally.

    Handles .jsonl (line by line) and .parquet (via pyarrow). Downloading whole
    files avoids the flaky streaming-client lifecycle on Windows.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "The 'huggingface_hub' library is required.\n"
            "  pip install datasets   (or: pip install -e \".[data]\")"
        )

    for fname in files:
        print(f"  downloading {fname} ...")
        local = hf_hub_download(repo_id=dataset_id, filename=fname, repo_type="dataset")
        if fname.endswith(".jsonl"):
            with open(local, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        elif fname.endswith(".parquet"):
            import pyarrow.parquet as pq

            table = pq.read_table(local)
            for batch in table.to_batches():
                yield from batch.to_pylist()
        else:
            sys.exit(f"Unsupported file type: {fname}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + convert a public agent dataset.")
    ap.add_argument("--dataset", choices=list(ADAPTERS), default="agentforge")
    ap.add_argument("--limit", type=int, default=300, help="Max FAILED rows to keep")
    ap.add_argument("--out", type=Path, default=Path("data/real_traces.jsonl"))
    ap.add_argument("--include-success", action="store_true",
                    help="Also keep successful rows (up to --limit each)")
    ap.add_argument("--scan-cap", type=int, default=20000,
                    help="Safety cap on rows scanned while looking for failures")
    args = ap.parse_args()

    dataset_id, files, adapt = ADAPTERS[args.dataset]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    kept_fail = 0
    kept_ok = 0
    scanned = 0
    print(f"Fetching {dataset_id} (looking for {args.limit} failed rows)...")

    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in iter_rows(dataset_id, files):
            scanned += 1
            if scanned > args.scan_cap:
                print(f"  reached scan cap ({args.scan_cap}); stopping.")
                break
            rec = adapt(row)
            if rec is None:
                continue
            is_fail = rec["error"] is not None
            if is_fail:
                if kept_fail >= args.limit:
                    if not args.include_success or kept_ok >= args.limit:
                        break
                    continue
                kept_fail += 1
            else:
                if not args.include_success or kept_ok >= args.limit:
                    continue
                kept_ok += 1
            rec["source_dataset"] = args.dataset
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (kept_fail + kept_ok) % 50 == 0:
                print(f"  kept {kept_fail} failed / {kept_ok} ok (scanned {scanned})")

    total = kept_fail + kept_ok
    print(
        f"\nDone. Wrote {total} rows ({kept_fail} failed, {kept_ok} ok) -> {args.out}\n"
        f"Next:\n"
        f"  trace2train inspect {args.out}\n"
        f"  trace2train convert {args.out} -o out --max-traces 20"
    )
    if kept_fail == 0:
        print(
            "\n[!] No failed rows found. Try --include-success, a larger "
            "--scan-cap, or a different --dataset."
        )


if __name__ == "__main__":
    main()
