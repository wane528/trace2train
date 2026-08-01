"""Langfuse Public API v2 snapshot pull support."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol, TextIO
from uuid import uuid4

import httpx

DEFAULT_LANGFUSE_BASE_URL: Final = "https://cloud.langfuse.com"
MAX_RETRY_AFTER_SECONDS: Final = 86400
_OBSERVATIONS_PATH: Final = "/api/public/v2/observations"
_FIELDS: Final = "core,basic,io,trace_context"
_TIMEOUT: Final = httpx.Timeout(30.0, connect=10.0)
MAX_PAGES: Final = 10_000

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class LangfusePullError(Exception):
    """Raised when a Langfuse snapshot pull fails."""

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class LangfuseConfig:
    """Credentials and endpoint for Langfuse snapshot pulls."""

    public_key: str = ""
    secret_key: str = field(default="", repr=False)
    base_url: str = DEFAULT_LANGFUSE_BASE_URL

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _normalize_base_url(self.base_url))

    @classmethod
    def from_env(cls, base_url: str | None = None) -> LangfuseConfig:
        """Build config from LANGFUSE_* environment variables."""

        chosen_base_url = os.getenv("LANGFUSE_BASE_URL", DEFAULT_LANGFUSE_BASE_URL)
        if base_url is not None:
            chosen_base_url = base_url
        return cls(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            base_url=chosen_base_url,
        )


@dataclass(frozen=True, slots=True)
class PullOptions:
    """Request options for a Langfuse observation pull."""

    from_time: str | None = None
    to_time: str | None = None
    max_observations: int | None = None
    page_size: int = 100


class Sleep(Protocol):
    def __call__(self, seconds: float, /) -> object:
        ...


def pull_observations(
    output: Path,
    *,
    config: LangfuseConfig,
    options: PullOptions,
    transport: httpx.BaseTransport | None = None,
    sleep: Sleep = time.sleep,
) -> int:
    """Pull validated Langfuse observation rows into an atomic JSONL snapshot."""

    _validate_credentials(config)
    _validate_options(options)
    temp_path = output.with_name(f"{output.name}.{uuid4().hex}.tmp")
    total_rows = 0
    pending_error_message: str | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            with httpx.Client(
                base_url=config.base_url,
                timeout=_TIMEOUT,
                transport=transport,
                auth=(config.public_key, config.secret_key),
            ) as client:
                cursor: str | None = None
                seen_cursors: set[str] = set()
                page_count = 0
                retried_rate_limit = False
                while True:
                    response = client.get(_OBSERVATIONS_PATH, params=_build_params(options, cursor))
                    if response.status_code == 429:
                        if retried_rate_limit:
                            raise LangfusePullError("Langfuse rate limit persisted after retry")
                        sleep(_retry_after_seconds(response))
                        retried_rate_limit = True
                        continue
                    retried_rate_limit = False
                    page_count += 1
                    if page_count > MAX_PAGES:
                        raise LangfusePullError("Langfuse pagination exceeded maximum page count")
                    rows, next_cursor = _parse_response_document(response)
                    total_rows = _write_page_rows(handle, rows, options, total_rows)
                    if _reached_max_observations(options, total_rows) or next_cursor is None:
                        break
                    if next_cursor in seen_cursors:
                        raise LangfusePullError("Langfuse response cursor repeated")
                    seen_cursors.add(next_cursor)
                    cursor = next_cursor
        temp_path.replace(output)
        return total_rows
    except httpx.TimeoutException:
        _cleanup_temp_file(temp_path)
        pending_error_message = "Langfuse request timed out"
    except httpx.InvalidURL:
        _cleanup_temp_file(temp_path)
        pending_error_message = "Langfuse request failed"
    except httpx.NetworkError:
        _cleanup_temp_file(temp_path)
        pending_error_message = "Langfuse network request failed"
    except httpx.HTTPError:
        _cleanup_temp_file(temp_path)
        pending_error_message = "Langfuse request failed"
    except OSError:
        _cleanup_temp_file(temp_path)
        pending_error_message = "Failed to write Langfuse snapshot"
    except LangfusePullError:
        _cleanup_temp_file(temp_path)
        raise
    if pending_error_message is not None:
        raise LangfusePullError(pending_error_message)
    raise AssertionError("unreachable")


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/") or DEFAULT_LANGFUSE_BASE_URL


def _validate_credentials(config: LangfuseConfig) -> None:
    if config.public_key.strip() and config.secret_key.strip():
        return
    raise LangfusePullError("Missing Langfuse credentials")


def _validate_options(options: PullOptions) -> None:
    if options.max_observations is not None and options.max_observations < 1:
        raise LangfusePullError("max_observations must be at least 1")
    if not 1 <= options.page_size <= 1000:
        raise LangfusePullError("page_size must be between 1 and 1000")


def _build_params(options: PullOptions, cursor: str | None) -> dict[str, str | int]:
    params: dict[str, str | int] = {"fields": _FIELDS, "limit": options.page_size}
    if options.from_time is not None:
        params["fromStartTime"] = options.from_time
    if options.to_time is not None:
        params["toStartTime"] = options.to_time
    if cursor is not None:
        params["cursor"] = cursor
    return params


def _retry_after_seconds(response: httpx.Response) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return 0.0
    if not retry_after.isascii() or not retry_after.isdigit():
        return 0.0
    try:
        seconds = int(retry_after)
    except ValueError:
        return 0.0
    if seconds > MAX_RETRY_AFTER_SECONDS:
        return 0.0
    return float(seconds)


def _parse_response_document(response: httpx.Response) -> tuple[list[JsonObject], str | None]:
    _raise_for_status(response)
    payload: object | None = None
    invalid_json = False
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        invalid_json = True
    if invalid_json:
        raise LangfusePullError("Langfuse response was not valid JSON")
    if not isinstance(payload, dict):
        raise LangfusePullError("Langfuse response must be a JSON object")
    if "data" not in payload or not isinstance(payload["data"], list):
        raise LangfusePullError("Langfuse response data must be a list")
    if "meta" not in payload or not isinstance(payload["meta"], dict):
        raise LangfusePullError("Langfuse response meta must be an object")
    cursor = payload["meta"].get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise LangfusePullError("Langfuse response cursor must be a string or null")
    if cursor == "":
        raise LangfusePullError("Langfuse response cursor must not be empty")
    rows = [_validate_row(item) for item in payload["data"]]
    return rows, cursor


def _validate_row(item: object) -> JsonObject:
    if not isinstance(item, dict):
        raise LangfusePullError("Langfuse observation rows must be JSON objects")
    required_values = (item.get("id"), item.get("traceId"), item.get("startTime"))
    if not all(isinstance(value, str) and value for value in required_values):
        raise LangfusePullError("Langfuse observation rows must include id, traceId, and startTime")
    if not any(key in item for key in ("type", "parentObservationId", "isRootObservation")):
        raise LangfusePullError("Langfuse observation rows must include a discriminator")
    return item


def _write_page_rows(
    handle: TextIO,
    rows: Sequence[JsonObject],
    options: PullOptions,
    total_rows: int,
) -> int:
    written = total_rows
    for row in rows:
        if _reached_max_observations(options, written):
            break
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        written += 1
    return written


def _reached_max_observations(options: PullOptions, total_rows: int) -> bool:
    return options.max_observations is not None and total_rows >= options.max_observations


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    match response.status_code:
        case 401:
            raise LangfusePullError("Langfuse authentication failed")
        case 403:
            raise LangfusePullError("Langfuse project access denied")
        case 404:
            raise LangfusePullError(
                "Langfuse observations endpoint not found; "
                "this requires Langfuse Cloud/self-hosted v4 Public API v2"
            )
        case _:
            raise LangfusePullError(
                f"Langfuse request failed with status {response.status_code}"
            )


def _cleanup_temp_file(temp_path: Path) -> None:
    try:
        temp_path.unlink()
    except FileNotFoundError:
        return
