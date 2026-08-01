"""Generic messages-JSONL importer.

Lowers the barrier to entry: any tool that can dump conversations as JSONL can
feed trace2train. Each line is one conversation:

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}],
     "error": null,         # optional: marks this convo as failed
     "id": "anything"}      # optional

We wrap each conversation in a single-Run Trace so the rest of the pipeline
(detect/generate/export) works unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import Run, RunType, Trace


class MessagesImporter:
    format_id = "messages"

    def sniff(self, first_line: dict) -> bool:
        msgs = first_line.get("messages")
        if isinstance(msgs, list) and msgs:
            first = msgs[0]
            return isinstance(first, dict) and ("role" in first or "content" in first)
        return False

    def load(self, path: Path, max_traces: int | None = None) -> list[Trace]:
        traces: list[Trace] = []
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if max_traces is not None and len(traces) >= max_traces:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines

                messages = obj.get("messages") or []
                if not isinstance(messages, list) or not messages:
                    continue

                conv_id = str(obj.get("id") or f"conv-{i}")
                error = obj.get("error")

                run = Run(
                    id=conv_id,
                    name="conversation",
                    run_type=RunType.CHAIN,
                    inputs={"messages": messages},
                    outputs=self._extract_output(messages),
                    error=error if isinstance(error, str) else None,
                    raw=obj,
                )
                traces.append(
                    Trace(
                        trace_id=conv_id,
                        root=run,
                        runs=[run],
                        source_file=str(path),
                        source_format=self.format_id,
                    )
                )
        return traces

    @staticmethod
    def _extract_output(messages: list) -> dict:
        """The last assistant message is treated as the run output."""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") in ("assistant", "ai"):
                return {"output": msg.get("content", "")}
        return {}
