"""Dataset health-check (stats) tests — distribution, skew warnings, meta."""

from __future__ import annotations

from trace2train.detect import DetectionResult
from trace2train.models import (
    ConversationTurn,
    DPORecord,
    Role,
    Run,
    RunType,
    SFTRecord,
    Trace,
)
from trace2train.stats import SKEW_THRESHOLD, build_stats


def _det(failure_type: str) -> DetectionResult:
    run = Run(id="r", name="n", run_type=RunType.CHAIN, error="boom")
    trace = Trace(trace_id="t", root=run, runs=[run])
    return DetectionResult(trace, failed=True, reasons=["boom"], env_only=False,
                           failure_type=failure_type)


def _sft(text: str = "hello world this is a sample") -> SFTRecord:
    return SFTRecord(
        conversations=[
            ConversationTurn(from_=Role.HUMAN, value="do X"),
            ConversationTurn(from_=Role.ASSISTANT, value=text),
        ]
    )


def _dpo() -> DPORecord:
    return DPORecord(
        conversations=[ConversationTurn(from_=Role.HUMAN, value="do X")],
        chosen="good answer here",
        rejected="bad answer here",
    )


def _dropped(**kw) -> dict:
    base = {"unusable": 0, "duplicate": 0, "leak": 0, "skipped_uncertain": 0, "rejected": 0}
    base.update(kw)
    return base


def test_failure_type_distribution_counts_kept_detections():
    sfts = [_sft() for _ in range(3)]
    dets = [_det("wrong_tool"), _det("wrong_tool"), _det("bad_args")]
    stats = build_stats(sfts, [], dets, _dropped())
    assert stats.failure_type_dist == {"wrong_tool": 2, "bad_args": 1}
    assert stats.sft_count == 3


def test_skew_warning_fires_above_threshold():
    # 9 wrong_tool vs 1 bad_args → 90% > 60% threshold
    dets = [_det("wrong_tool")] * 9 + [_det("bad_args")]
    sfts = [_sft() for _ in range(10)]
    stats = build_stats(sfts, [], dets, _dropped())
    assert any("skewed" in w and "wrong_tool" in w for w in stats.warnings)


def test_no_skew_warning_when_balanced():
    dets = [_det("wrong_tool")] * 5 + [_det("bad_args")] * 5
    sfts = [_sft() for _ in range(10)]
    stats = build_stats(sfts, [], dets, _dropped())
    assert not any("skewed" in w for w in stats.warnings)
    # balanced 50/50 sits below the 0.60 threshold
    assert max(stats.failure_type_dist.values()) / 10 <= SKEW_THRESHOLD


def test_small_dataset_warning():
    stats = build_stats([_sft()], [], [_det("wrong_tool")], _dropped())
    assert any("small" in w for w in stats.warnings)


def test_empty_dataset_warning():
    stats = build_stats([], [], [], _dropped())
    assert any("empty" in w for w in stats.warnings)


def test_length_stats_computed():
    sfts = [_sft("a" * 10), _sft("b" * 20), _sft("c" * 30)]
    dets = [_det("wrong_tool") for _ in range(3)]
    stats = build_stats(sfts, [], dets, _dropped())
    # each record = len("do X")=4 + payload
    assert stats.sft_length.count == 3
    assert stats.sft_length.min < stats.sft_length.max
    assert stats.sft_length.min <= stats.sft_length.median <= stats.sft_length.max


def test_dedup_leak_rate_and_noisy_warning():
    # 6 kept, 4 removed as dup/leak → 40% > 30% → noisy warning
    sfts = [_sft() for _ in range(6)]
    dets = [_det("wrong_tool") for _ in range(6)]
    stats = build_stats(sfts, [], dets, _dropped(duplicate=3, leak=1))
    assert stats.dedup_leak_rate == 0.4
    assert any("noisy" in w for w in stats.warnings)


def test_to_meta_shape():
    sfts = [_sft() for _ in range(12)]
    dets = [_det("wrong_tool") for _ in range(12)]
    meta = build_stats(sfts, [_dpo()], dets, _dropped()).to_meta()
    assert meta["sft_count"] == 12
    assert meta["dpo_count"] == 1
    assert meta["failure_type_dist"] == {"wrong_tool": 12}
    assert "sft_length" in meta and "median" in meta["sft_length"]
    assert isinstance(meta["warnings"], list)
