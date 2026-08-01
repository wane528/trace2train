# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27"]
# ///
"""Delete ONLY the synthetic validation traces created by
`scripts/seed_langfuse_validation.py`.

Safety design:
- Matches traces by their exact seeded names only
  (weather-agent, qa-agent, calendar-agent). Nothing else is touched.
- Dry-run by default: it lists what WOULD be deleted and stops.
  Pass --confirm to actually delete.
- Never prints credentials.
- Uses only the official Public API:
  GET  /api/public/traces?name=...   (find seeded traces)
  DELETE /api/public/traces/{traceId} (delete; observations/scores cascade)

Deletion is asynchronous on Langfuse (data-warehouse removal within ~15 min),
so re-run the dry-run later to confirm the traces are gone.

Run (from repo root, keys in .env or exported):
    uv run scripts/cleanup_langfuse_validation.py           # dry-run (lists only)
    uv run scripts/cleanup_langfuse_validation.py --confirm  # actually delete
or:
    python scripts/cleanup_langfuse_validation.py [--confirm]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

import httpx

SEED_TRACE_NAMES: Final = ("weather-agent", "qa-agent", "calendar-agent")
_TIMEOUT: Final = httpx.Timeout(30.0, connect=10.0)


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


def _config() -> tuple[str, str, str]:
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    base = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").strip().rstrip("/")
    return pk, sk, base


def _find_trace_ids(client: httpx.Client) -> dict[str, list[str]]:
    """Return {trace_name: [trace_id, ...]} for the seeded names only."""
    found: dict[str, list[str]] = {}
    for name in SEED_TRACE_NAMES:
        ids: list[str] = []
        page = 1
        while True:
            resp = client.get(
                "/api/public/traces",
                params={"name": name, "limit": 50, "page": page},
            )
            resp.raise_for_status()
            body = resp.json()
            data = body.get("data") or []
            for trace in data:
                trace_id = trace.get("id")
                trace_name = trace.get("name")
                # defensive: only accept exact name matches
                if isinstance(trace_id, str) and trace_name == name:
                    ids.append(trace_id)
            meta = body.get("meta") or {}
            total_pages = meta.get("totalPages")
            if not isinstance(total_pages, int) or page >= total_pages:
                break
            page += 1
        found[name] = ids
    return found


def main(argv: list[str]) -> int:
    confirm = "--confirm" in argv[1:]
    _load_env_file()
    pk, sk, base = _config()
    if not (pk and sk):
        print("Missing Langfuse credentials in environment/.env")
        return 2

    with httpx.Client(base_url=base, auth=(pk, sk), timeout=_TIMEOUT) as client:
        # auth preflight (no secrets printed)
        health = client.get("/api/public/projects")
        if health.status_code != 200:
            print(f"auth_failed status={health.status_code}")
            return 1

        found = _find_trace_ids(client)
        total = sum(len(v) for v in found.values())
        for name, ids in found.items():
            print(f"{name}: {len(ids)} trace(s)")
        print(f"total_seed_traces_matched={total}")

        if total == 0:
            print("nothing to delete")
            return 0

        if not confirm:
            print("dry-run: no traces deleted. Re-run with --confirm to delete.")
            return 0

        deleted = 0
        failed = 0
        for name, ids in found.items():
            for trace_id in ids:
                resp = client.delete(f"/api/public/traces/{trace_id}")
                if resp.status_code in (200, 202, 204):
                    deleted += 1
                else:
                    failed += 1
                    print(f"delete_failed name={name} status={resp.status_code}")
        print(f"delete_requested={deleted} delete_failed={failed}")
        print("note: Langfuse deletion is asynchronous (~15 min); re-run dry-run to confirm.")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
