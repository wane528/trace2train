"""Inspect-report tests — the numbers that make the headline must be correct."""

from __future__ import annotations

from trace2train.demo import demo_path
from trace2train.importers import load
from trace2train.inspect import build_report


def test_report_on_fixture(langsmith_file):
    report = build_report(load(langsmith_file))
    assert report.total_traces == 3
    assert report.failed == 2
    assert report.env_only == 1
    assert report.trainable == 1
    assert report.dirty.env_noise == 1


def test_headline_format(langsmith_file):
    report = build_report(load(langsmith_file))
    hl = report.headline
    assert "3 traces" in hl
    assert "trainable" in hl


def test_report_on_demo_data():
    """Demo data (tool-call/behavior failures) must produce a stable report."""
    report = build_report(load(demo_path()))
    assert report.total_traces == 19
    assert report.failed == 16
    assert report.env_only == 2
    assert report.trainable == 14
    assert report.dirty.pii == 3
    assert report.dirty.duplicate == 1
    assert 0 <= report.dirty_pct <= 100


def test_report_has_failure_types():
    """The demo must surface a tool-call failure-type breakdown."""
    report = build_report(load(demo_path()))
    assert report.failure_types  # non-empty
    assert report.failure_types.get("wrong_tool", 0) >= 1
    # counts of trainable failure types sum to the trainable count
    assert sum(report.failure_types.values()) == report.trainable


def test_messages_pii_counted(messages_file):
    report = build_report(load(messages_file))
    # m1 has "a@b.com" and is marked failed -> PII should be counted
    assert report.failed == 1
    assert report.dirty.pii == 1
