"""Decontamination tests: PII redaction, dedup, leak filter, record redaction."""

from __future__ import annotations

from trace2train.clean import (
    clean_text,
    filter_leaked,
    has_pii,
    redact_dpo,
    redact_pii,
    redact_sft,
)
from trace2train.models import (
    CleanIssue,
    ConversationTurn,
    DPORecord,
    Role,
    SFTRecord,
)


def test_redact_email_and_phone():
    out = redact_pii("mail john@x.com or call +1 415 555 0199")
    assert "john@x.com" not in out
    assert "[EMAIL]" in out
    assert "[PHONE]" in out


def test_has_pii_does_not_mutate():
    text = "reach me at a@b.com"
    assert has_pii(text)
    assert text == "reach me at a@b.com"  # unchanged


def test_dedup_via_fingerprints():
    seen: set[str] = set()
    a = clean_text("Hello World", seen_fingerprints=seen)
    b = clean_text("hello   world", seen_fingerprints=seen)  # normalized dup
    assert CleanIssue.DUPLICATE not in a.issues
    assert CleanIssue.DUPLICATE in b.issues


def test_empty_and_short_flagged():
    assert CleanIssue.EMPTY in clean_text("   ").issues
    assert CleanIssue.TOO_SHORT in clean_text("hi").issues


def test_leak_filter_by_fingerprint():
    from trace2train.models import fingerprint_text

    fp = fingerprint_text("secret eval answer")
    assert filter_leaked("secret eval answer", leak_fingerprints={fp}) == CleanIssue.LEAK
    assert filter_leaked("something else", leak_fingerprints={fp}) is None


def test_redact_sft_record():
    rec = SFTRecord(conversations=[
        ConversationTurn(from_=Role.HUMAN, value="email a@b.com"),
        ConversationTurn(from_=Role.ASSISTANT, value="ok"),
    ])
    redact_sft(rec)
    assert "a@b.com" not in rec.conversations[0].value
    assert "[EMAIL]" in rec.conversations[0].value


def test_redact_dpo_record():
    rec = DPORecord(
        conversations=[ConversationTurn(from_=Role.HUMAN, value="call +1 415 555 0199")],
        chosen="ok a@b.com",
        rejected="fail c@d.com",
    )
    redact_dpo(rec)
    assert "[PHONE]" in rec.conversations[0].value
    assert "a@b.com" not in rec.chosen
    assert "c@d.com" not in rec.rejected
