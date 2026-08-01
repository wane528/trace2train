"""Export-format tests: LLaMA-Factory ShareGPT shape + provenance."""

from __future__ import annotations

import json

from trace2train.export import export_dpo, export_sft
from trace2train.models import (
    ConversationTurn,
    DPORecord,
    Provenance,
    Role,
    SFTRecord,
)


def test_sft_export_shape(tmp_path):
    rec = SFTRecord(
        conversations=[
            ConversationTurn(from_=Role.HUMAN, value="q"),
            ConversationTurn(from_=Role.ASSISTANT, value="a"),
        ],
        provenance=Provenance(trace_id="t1", original_error="boom"),
    )
    path = export_sft([rec], tmp_path / "sft.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["conversations"] == [
        {"from": "human", "value": "q"},
        {"from": "gpt", "value": "a"},
    ]
    assert obj["_provenance"]["trace_id"] == "t1"


def test_dpo_export_shape(tmp_path):
    rec = DPORecord(
        conversations=[ConversationTurn(from_=Role.HUMAN, value="q")],
        chosen="good",
        rejected="bad",
    )
    path = export_dpo([rec], tmp_path / "dpo.jsonl")
    obj = json.loads(path.read_text(encoding="utf-8").strip())
    assert obj["conversations"] == [{"from": "human", "value": "q"}]
    assert obj["chosen"] == {"from": "gpt", "value": "good"}
    assert obj["rejected"] == {"from": "gpt", "value": "bad"}
