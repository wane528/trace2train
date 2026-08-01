"""Decontamination: PII redaction, dedup, and benchmark-leak filtering.

The core insight behind this module (from the product research): raw failed
traces are dirty — up to ~48% contain retries, fallbacks, cache hits, PII,
or benchmark-set leakage. Feeding them straight into a trainer poisons the
model. This stage is the actual moat.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import CleanedText, CleanIssue, DPORecord, SFTRecord, fingerprint_text

# ---------------------------------------------------------------------------
# PII patterns (pragmatic, regex-based; LLM-assisted redaction can be added)
# ---------------------------------------------------------------------------

PII_PATTERNS: list[tuple[str, str]] = [
    # emails
    (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "[EMAIL]"),
    # phone numbers — require phone-like structure to avoid matching bare
    # timestamps / IDs / ISO dates (big sources of false positives in traces).
    # Two accepted shapes:
    #   1. an international +CC number:  +1 415 555 0199 / +14155550199
    #   2. a 3-group US-style number:    (415) 555-0199 / 415-555-0199
    # ISO dates (YYYY-MM-DD) are explicitly excluded via the date lookahead.
    (
        r"(?<![\w.])(?!\d{4}[-/]\d{2}[-/]\d{2})"
        r"(?:\+\d{1,3}[\s.\-]?\d(?:[\s.\-]?\d){7,13}"
        r"|\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4})"
        r"(?![\w.])",
        "[PHONE]",
    ),
    # IPv4
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]"),
    # API keys / secrets — provider-prefixed keys, or long mixed-case+digit
    # tokens (require BOTH letters and digits to skip plain hex hashes/words).
    (
        r"\b(?:sk-[A-Za-z0-9]{16,}|(?=[A-Za-z0-9]{32,}\b)(?=[A-Za-z]*\d)"
        r"(?=\d*[A-Za-z])[A-Za-z0-9]{32,})\b",
        "[TOKEN]",
    ),
    # credit-card-ish 16-digit runs (grouped)
    (r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b", "[CARD]"),
]

MIN_SAMPLE_CHARS = 20
MAX_SAMPLE_CHARS = 20_000


def redact_pii(text: str) -> str:
    """Replace known PII shapes with placeholders."""
    out = text
    for pattern, repl in PII_PATTERNS:
        out = re.sub(pattern, repl, out)
    return out


def has_pii(text: str) -> bool:
    """Detect PII without modifying the text (used by `inspect` to count)."""
    return any(re.search(pattern, text) for pattern, _ in PII_PATTERNS)


def clean_text(
    text: str,
    *,
    seen_fingerprints: set[str] | None = None,
    redact: bool = True,
) -> CleanedText:
    """One-stop cleaning: redact PII, check emptiness/length, dedup.

    Returns the cleaned text plus the issues found. The fingerprint is
    recorded into `seen_fingerprints` when passed, enabling cross-batch dedup.
    """
    issues: list[CleanIssue] = []

    if redact:
        text = redact_pii(text)

    stripped = text.strip()
    if not stripped:
        issues.append(CleanIssue.EMPTY)
    elif len(stripped) < MIN_SAMPLE_CHARS:
        issues.append(CleanIssue.TOO_SHORT)
    elif len(stripped) > MAX_SAMPLE_CHARS:
        issues.append(CleanIssue.TOO_LONG)

    fp = fingerprint_text(stripped) if stripped else ""
    if fp and seen_fingerprints is not None:
        if fp in seen_fingerprints:
            issues.append(CleanIssue.DUPLICATE)
        seen_fingerprints.add(fp)

    return CleanedText(text=stripped, issues=issues, fingerprint=fp)


# ---------------------------------------------------------------------------
# Benchmark-leak filtering
# ---------------------------------------------------------------------------

# A small default stop-list of known benchmark/answer phrases. Real users can
# pass their own eval/golden-set fingerprints to `filter_leaked`.
DEFAULT_LEAK_PHRASES: tuple[str, ...] = ()

_LEAK_FINGERPRINT_CACHE: set[str] = set()


def filter_leaked(
    text: str,
    *,
    leak_fingerprints: set[str] | None = None,
    leak_phrases: Iterable[str] = (),
) -> CleanIssue | None:
    """Return CleanIssue.LEAK if text matches a leaked/golden sample.

    `leak_fingerprints`: exact fingerprints of your held-out eval set — any
    training sample whose fingerprint collides is a leak.
    `leak_phrases`: substring phrases known to appear only in benchmarks.
    """
    fp = fingerprint_text(text)
    if leak_fingerprints and fp in leak_fingerprints:
        return CleanIssue.LEAK

    lower = text.lower()
    for phrase in list(DEFAULT_LEAK_PHRASES) + list(leak_phrases):
        if phrase.lower() in lower:
            return CleanIssue.LEAK
    return None


# ---------------------------------------------------------------------------
# Record-level redaction (applied to the actual exported content)
# ---------------------------------------------------------------------------


def redact_sft(record: SFTRecord) -> SFTRecord:
    """Redact PII in every turn of an SFT record (mutates and returns it)."""
    for turn in record.conversations:
        turn.value = redact_pii(turn.value)
    return record


def redact_dpo(record: DPORecord) -> DPORecord:
    """Redact PII in prompt turns, chosen, and rejected of a DPO record."""
    for turn in record.conversations:
        turn.value = redact_pii(turn.value)
    record.chosen = redact_pii(record.chosen)
    record.rejected = redact_pii(record.rejected)
    return record
