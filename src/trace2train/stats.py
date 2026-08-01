"""Distribution health check for a generated training set.

`inspect` tells you how many traces are trainable. This module answers the next
question a fine-tuner actually cares about: *is the resulting dataset healthy, or
is it skewed?* A set that is 90% one failure type will over-fit the model to that
one behavior. We surface the shape (failure-type mix, sample-length spread,
decontamination hit-rate) plus explicit skew warnings — pure counting, no LLM.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .detect import DetectionResult
from .models import DPORecord, SFTRecord

# A single failure type owning more than this share of the trainable set is
# flagged: the fine-tuned model risks over-fitting that one behavior.
SKEW_THRESHOLD = 0.60


@dataclass
class LengthStats:
    """Character-length spread of a set of samples (min/median/max)."""

    count: int = 0
    min: int = 0
    median: int = 0
    max: int = 0


@dataclass
class DatasetStats:
    """Health check for one `convert` run's output."""

    sft_count: int = 0
    dpo_count: int = 0
    failure_type_dist: dict[str, int] = field(default_factory=dict)
    sft_length: LengthStats = field(default_factory=LengthStats)
    dpo_length: LengthStats = field(default_factory=LengthStats)
    dropped_total: int = 0
    dedup_leak_rate: float = 0.0  # share of considered samples removed as dup/leak
    warnings: list[str] = field(default_factory=list)

    def to_meta(self) -> dict:
        """Serialize for meta.json's `distribution` section."""

        return {
            "sft_count": self.sft_count,
            "dpo_count": self.dpo_count,
            "failure_type_dist": self.failure_type_dist,
            "sft_length": vars(self.sft_length),
            "dpo_length": vars(self.dpo_length),
            "dedup_leak_rate": round(self.dedup_leak_rate, 4),
            "warnings": self.warnings,
        }


def _median(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) // 2


def _length_stats(lengths: list[int]) -> LengthStats:
    if not lengths:
        return LengthStats()
    return LengthStats(
        count=len(lengths),
        min=min(lengths),
        median=_median(lengths),
        max=max(lengths),
    )


def _sft_len(rec: SFTRecord) -> int:
    return sum(len(t.value) for t in rec.conversations)


def _dpo_len(rec: DPORecord) -> int:
    prompt = sum(len(t.value) for t in rec.conversations)
    return prompt + len(rec.chosen) + len(rec.rejected)


def build_stats(
    sft_records: list[SFTRecord],
    dpo_records: list[DPORecord],
    kept_detections: list[DetectionResult],
    dropped: dict[str, int],
) -> DatasetStats:
    """Compute the dataset health report. Pure counting, no LLM.

    `kept_detections`: the trainable detections that produced kept records, used
    for the failure-type mix. `dropped`: the convert-time drop counters.
    """

    dist = dict(
        Counter(d.failure_type for d in kept_detections).most_common()
    )

    dropped_total = sum(dropped.values())
    considered = len(sft_records) + dropped.get("duplicate", 0) + dropped.get("leak", 0)
    decon = dropped.get("duplicate", 0) + dropped.get("leak", 0)
    dedup_leak_rate = (decon / considered) if considered else 0.0

    stats = DatasetStats(
        sft_count=len(sft_records),
        dpo_count=len(dpo_records),
        failure_type_dist=dist,
        sft_length=_length_stats([_sft_len(r) for r in sft_records]),
        dpo_length=_length_stats([_dpo_len(r) for r in dpo_records]),
        dropped_total=dropped_total,
        dedup_leak_rate=dedup_leak_rate,
    )
    stats.warnings = _warnings(stats, dist)
    return stats


def _warnings(stats: DatasetStats, dist: dict[str, int]) -> list[str]:
    """Actionable health warnings — kept few and specific, never noisy."""

    warnings: list[str] = []
    total = sum(dist.values())

    if total and dist:
        top_type, top_count = next(iter(dist.items()))
        share = top_count / total
        if share > SKEW_THRESHOLD:
            warnings.append(
                f"skewed: {top_type} is {share:.0%} of the trainable set "
                f"({top_count}/{total}) - the model may over-fit this one behavior. "
                "Consider adding traces of other failure types."
            )

    if stats.sft_count == 0 and stats.dpo_count == 0:
        warnings.append("empty: no training records were produced.")
    elif stats.sft_count < 10:
        warnings.append(
            f"small: only {stats.sft_count} SFT record(s) - usually too few to "
            "fine-tune on alone; gather more failed traces."
        )

    if stats.dedup_leak_rate > 0.30:
        warnings.append(
            f"noisy: {stats.dedup_leak_rate:.0%} of samples were dropped as "
            "duplicates/leaks - the source traces may be highly repetitive."
        )

    return warnings
