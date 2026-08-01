# /// script
# requires-python = ">=3.11"
# dependencies = ["langfuse>=3,<4"]
# ///
"""One-off synthetic Langfuse validation seeder (NON-SENSITIVE data only).

Seeds a small set of tool-call/agent-behavior traces into a DEDICATED Langfuse
validation project so trace2train can pull them via the real Public API v2.

- No customer data, no real prompts, no PII.
- Credentials come only from environment (LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL).
- Never prints credentials.

Run (from repo root, keys in .env or exported):
    uv run scripts/seed_langfuse_validation.py
or:
    python scripts/seed_langfuse_validation.py   # if langfuse is installed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_env_file() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    _load_env_file()
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        print("Missing Langfuse credentials in environment/.env")
        return 2

    from langfuse import Langfuse

    client = Langfuse()
    if not client.auth_check():
        print("auth_check=False")
        return 1
    print("auth_check=True")

    # Trace 1: wrong-tool behavioral failure (tool errored, agent gave bad answer).
    with client.start_as_current_observation(
        name="weather-agent",
        as_type="chain",
        input={"messages": [{"role": "user", "content": "What's the weather in Berlin?"}]},
    ) as root:
        with root.start_as_current_observation(
            name="calculator",
            as_type="tool",
            input={"expr": "weather Berlin"},
            level="ERROR",
            status_message="wrong tool: calculator used for a weather query",
        ) as tool:
            tool.update(output={"error": "invalid expression"})
        root.update(output="I got 42.")

    # Trace 2: successful generation (should NOT be trainable) + structured IO.
    with client.start_as_current_observation(
        name="qa-agent",
        as_type="chain",
        input={"messages": [{"role": "user", "content": "Capital of France?"}]},
    ) as root:
        with root.start_as_current_observation(
            name="answer",
            as_type="generation",
            model="synthetic-model",
            input={"messages": [{"role": "user", "content": "Capital of France?"}]},
        ) as gen:
            gen.update(output={"role": "assistant", "content": "Paris."})
        root.update(output="Paris.")

    # Trace 3: bad-args behavioral failure across multiple observations.
    with client.start_as_current_observation(
        name="calendar-agent",
        as_type="agent",
        input={"messages": [{"role": "user", "content": "Book a flight for 2026-09-01"}]},
    ) as root:
        with root.start_as_current_observation(
            name="search",
            as_type="retriever",
            input={"query": "flights"},
        ) as retr:
            retr.update(output=["flight-a", "flight-b"])
        with root.start_as_current_observation(
            name="create_event",
            as_type="tool",
            input={"title": "flight", "time": "09/01"},
            level="ERROR",
            status_message="bad args: time must be ISO 8601",
        ) as tool:
            tool.update(output={"error": "time must be ISO 8601"})
        root.update(output="Sorry, I could not book that.")

    client.flush()
    print("seeded_traces=3")
    print("flush=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
