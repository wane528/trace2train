"""Export to LLaMA-Factory-compatible JSONL.

Writes SFT and DPO records to separate JSONL files in the format consumed by
LLaMA-Factory (and compatible with Unsloth/TRL workflows after trivial
mapping). A sidecar `.meta.json` records counts + provenance for audit.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import DPORecord, SFTRecord


def export_sft(records: list[SFTRecord], path: str | Path, *, append: bool = False) -> Path:
    """Write SFT records as LLaMA-Factory 'conversations' JSONL.

    `append=True` adds to an existing file (used by `--resume`) instead of
    overwriting it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as fh:
        for rec in records:
            line = rec.to_llama_factory()
            line["_provenance"] = rec.provenance.model_dump()
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


def export_dpo(records: list[DPORecord], path: str | Path, *, append: bool = False) -> Path:
    """Write DPO records as LLaMA-Factory 'chosen/rejected' JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as fh:
        for rec in records:
            line = rec.to_llama_factory()
            line["_provenance"] = rec.provenance.model_dump()
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


def write_meta(
    sft_path: Path,
    dpo_path: Path,
    *,
    total_traces: int,
    failed_traces: int,
    trainable_traces: int,
    dropped: dict,
    out_dir: str | Path,
    offline: bool = False,
    distribution: dict | None = None,
) -> Path:
    """Write a small audit file so users can see what happened and why."""
    sft_key = "raw_review_records" if offline else "sft_records"
    meta = {
        "trace2train_version": __import__("trace2train").__version__,
        "mode": "offline" if offline else "llm",
        "counts": {
            "total_traces": total_traces,
            "failed_traces": failed_traces,
            "trainable_traces": trainable_traces,
            sft_key: 0,
            "dpo_records": 0,
        },
        "dropped": dropped,
        "outputs": {
            ("raw_review" if offline else "sft"): str(sft_path),
            "dpo": str(dpo_path),
        },
    }
    if distribution is not None:
        meta["distribution"] = distribution
    if offline:
        meta["note"] = (
            "offline mode: outputs are raw unverified traces for human review, "
            "not trainable SFT data (no LLM correction was applied)"
        )
    # count records cheaply (line count of the JSONL files we just wrote)
    for key, p in ((sft_key, sft_path), ("dpo_records", dpo_path)):
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                meta["counts"][key] = sum(1 for _ in fh)

    meta_path = Path(out_dir) / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path
