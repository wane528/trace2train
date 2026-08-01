"""Failure attribution label.

Attribution names WHY a trace failed: wrong tool, bad args, lost context, etc.
This label is stored in the exported provenance for auditing and is used by
stats.py for the failure-type distribution.

Historically attribution was a separate LLM pass. It is now produced by the
same single LLM call that generates the correction (see generate.py), halving
the per-trace LLM cost. This module keeps the small value type they share.
"""

from __future__ import annotations


class Attribution:
    def __init__(self, category: str, summary: str, trainable: bool):
        self.category = category
        self.summary = summary
        self.trainable = trainable

    def __str__(self) -> str:  # for logging / CLI display
        return f"{self.category}: {self.summary} (trainable={self.trainable})"
