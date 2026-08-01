"""Langfuse snapshot pull tests."""

from __future__ import annotations

import base64
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
import pytest

from trace2train.langfuse import (
    DEFAULT_LANGFUSE_BASE_URL,
    LangfuseConfig,
    LangfusePullError,
    PullOptions,
    pull_observations,
)

PUBLIC_KEY = "pk-live-123"
SECRET_KEY = "sk-live-456"


def _config(base_url: str = DEFAULT_LANGFUSE_BASE_URL) -> LangfuseConfig:
    return LangfuseConfig(public_key=PUBLIC_KEY, secret_key=SECRET_KEY, base_url=base_url)


def _row(
    observation_id: str,
    trace_id: str,
    start_time: str,
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": observation_id,
        "traceId": trace_id,
        "startTime": start_time,
        "type": "GENERATION",
    }
    row.update(extra)
    return row


def _response(rows: Sequence[object], cursor: object) -> httpx.Response:
    return httpx.Response(200, json={"data": rows, "meta": {"cursor": cursor}})


def _exception_chain_text(error: BaseException) -> str:
    parts = [repr(error), str(error), "".join(traceback.format_exception(error))]
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current = current.__cause__ or current.__context__
        if current is None:
            break
        parts.extend((repr(current), str(current), "".join(traceback.format_exception(current))))
    return "\n".join(parts)


def _exception_chain(error: BaseException) -> list[BaseException]:
    seen: set[int] = set()
    pending = [error]
    chain: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return chain


def _timeout_with_request(message: str, request: httpx.Request) -> httpx.ReadTimeout:
    return httpx.ReadTimeout(message, request=request)


def _connect_error_with_request(message: str, request: httpx.Request) -> httpx.ConnectError:
    return httpx.ConnectError(message, request=request)


def test_langfuse_config_from_env_prefers_explicit_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "env-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "env-secret")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://env.example/")

    config = LangfuseConfig.from_env(base_url="https://explicit.example///")

    assert config.public_key == "env-public"
    assert config.secret_key == "env-secret"
    assert config.base_url == "https://explicit.example"


def test_langfuse_config_from_env_uses_default_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)

    config = LangfuseConfig.from_env()

    assert config.public_key == ""
    assert config.secret_key == ""
    assert config.base_url == DEFAULT_LANGFUSE_BASE_URL


@pytest.mark.parametrize(
    ("public_key", "secret_key"),
    [("", SECRET_KEY), ("   ", SECRET_KEY), (PUBLIC_KEY, ""), (PUBLIC_KEY, " ")],
)
def test_pull_observations_rejects_empty_or_whitespace_credentials_before_http(
    tmp_path: Path,
    public_key: str,
    secret_key: str,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _response([], None)

    with pytest.raises(LangfusePullError, match="Missing Langfuse credentials"):
        pull_observations(
            output_path,
            config=LangfuseConfig(public_key=public_key, secret_key=secret_key),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
        )

    assert called is False
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("options", "expected_message"),
    [
        (PullOptions(max_observations=0), "max_observations must be at least 1"),
        (PullOptions(page_size=0), "page_size must be between 1 and 1000"),
        (PullOptions(page_size=1001), "page_size must be between 1 and 1000"),
    ],
)
def test_pull_observations_validates_options_before_http(
    tmp_path: Path,
    options: PullOptions,
    expected_message: str,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _response([], None)

    with pytest.raises(LangfusePullError, match=expected_message):
        pull_observations(
            output_path,
            config=_config(),
            options=options,
            transport=httpx.MockTransport(handler),
        )

    assert called is False
    assert not output_path.exists()


def test_pull_observations_sends_expected_request_params_and_basic_auth(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _response([_row("obs-1", "trace-1", "2026-08-01T00:00:00Z")], None)

    count = pull_observations(
        output_path,
        config=_config("https://cloud.langfuse.com///"),
        options=PullOptions(
            from_time="2026-08-01T00:00:00Z",
            to_time="2026-08-02T00:00:00Z",
            page_size=100,
        ),
        transport=httpx.MockTransport(handler),
    )

    assert count == 1
    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == (
        "https://cloud.langfuse.com/api/public/v2/observations"
        "?fields=core%2Cbasic%2Cio%2Ctrace_context"
        "&limit=100"
        "&fromStartTime=2026-08-01T00%3A00%3A00Z"
        "&toStartTime=2026-08-02T00%3A00%3A00Z"
    )
    auth_header = request.headers["Authorization"]
    decoded = base64.b64decode(
        auth_header.split(" ", maxsplit=1)[1]
    ).decode("utf-8")
    assert decoded == f"{PUBLIC_KEY}:{SECRET_KEY}"


def test_pull_observations_truncates_current_page_and_skips_next_request(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    cursors: list[str | None] = []
    rows = [
        _row("obs-1", "trace-1", "2026-08-01T00:00:00Z"),
        _row("obs-2", "trace-1", "2026-08-01T00:00:01Z"),
        _row("obs-3", "trace-1", "2026-08-01T00:00:02Z"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        cursors.append(request.url.params.get("cursor"))
        return _response(rows, "next-1")

    count = pull_observations(
        output_path,
        config=_config(),
        options=PullOptions(max_observations=2, page_size=100),
        transport=httpx.MockTransport(handler),
    )

    assert count == 2
    assert cursors == [None]
    assert output_path.read_text(encoding="utf-8") == (
        '{"id":"obs-1","traceId":"trace-1","startTime":"2026-08-01T00:00:00Z","type":"GENERATION"}\n'
        '{"id":"obs-2","traceId":"trace-1","startTime":"2026-08-01T00:00:01Z","type":"GENERATION"}\n'
    )


def test_pull_observations_writes_two_cursor_pages_exactly_once(tmp_path: Path) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    cursors: list[str | None] = []
    first_page = [_row("obs-1", "trace-1", "2026-08-01T00:00:00Z", input="hi")]
    second_page = [_row("obs-2", "trace-1", "2026-08-01T00:00:01Z", output="bye")]

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        cursors.append(cursor)
        if cursor is None:
            return _response(first_page, "next-1")
        assert cursor == "next-1"
        return _response(second_page, None)

    count = pull_observations(
        output_path,
        config=_config(),
        options=PullOptions(),
        transport=httpx.MockTransport(handler),
    )

    assert count == 2
    assert cursors == [None, "next-1"]
    assert output_path.read_text(encoding="utf-8") == (
        '{"id":"obs-1","traceId":"trace-1","startTime":"2026-08-01T00:00:00Z","type":"GENERATION","input":"hi"}\n'
        '{"id":"obs-2","traceId":"trace-1","startTime":"2026-08-01T00:00:01Z","type":"GENERATION","output":"bye"}\n'
    )


def test_pull_observations_preserves_existing_destination_when_second_page_fails(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    original_bytes = b'{"stable":true}\n'
    output_path.write_bytes(original_bytes)

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return _response([_row("obs-1", "trace-1", "2026-08-01T00:00:00Z")], "next-1")
        return httpx.Response(200, json={"data": [123], "meta": {"cursor": None}})

    with pytest.raises(LangfusePullError, match="Langfuse observation rows must be JSON objects"):
        pull_observations(
            output_path,
            config=_config(),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
        )

    assert output_path.read_bytes() == original_bytes
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


def test_pull_observations_rejects_repeated_cursor_and_cleans_up_temp_file(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    original_bytes = b'{"stable":true}\n'
    output_path.write_bytes(original_bytes)

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return _response([_row("obs-1", "trace-1", "2026-08-01T00:00:00Z")], "next-1")
        return _response([_row("obs-2", "trace-1", "2026-08-01T00:00:01Z")], "next-1")

    with pytest.raises(LangfusePullError, match="Langfuse response cursor repeated"):
        pull_observations(
            output_path,
            config=_config(),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
        )

    assert output_path.read_bytes() == original_bytes
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


def test_pull_observations_rejects_cursor_cycle_and_cleans_up_temp_file(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    original_bytes = b'{"stable":true}\n'
    output_path.write_bytes(original_bytes)

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return _response([_row("obs-1", "trace-1", "2026-08-01T00:00:00Z")], "next-a")
        if cursor == "next-a":
            return _response([_row("obs-2", "trace-1", "2026-08-01T00:00:01Z")], "next-b")
        assert cursor == "next-b"
        return _response([_row("obs-3", "trace-1", "2026-08-01T00:00:02Z")], "next-a")

    with pytest.raises(LangfusePullError, match="Langfuse response cursor repeated"):
        pull_observations(
            output_path,
            config=_config(),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
        )

    assert output_path.read_bytes() == original_bytes
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


def test_pull_observations_rejects_empty_cursor_and_cleans_up_temp_file(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    original_bytes = b'{"stable":true}\n'
    output_path.write_bytes(original_bytes)

    def handler(request: httpx.Request) -> httpx.Response:
        return _response([_row("obs-1", "trace-1", "2026-08-01T00:00:00Z")], "")

    with pytest.raises(LangfusePullError, match="Langfuse response cursor must not be empty"):
        pull_observations(
            output_path,
            config=_config(),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
        )

    assert output_path.read_bytes() == original_bytes
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


def test_pull_observations_stops_when_pagination_exceeds_max_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    original_bytes = b'{"stable":true}\n'
    output_path.write_bytes(original_bytes)
    monkeypatch.setattr("trace2train.langfuse.MAX_PAGES", 3)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(
            [_row(f"obs-{calls}", "trace-1", f"2026-08-01T00:00:0{calls}Z")],
            f"cursor-{calls}",
        )

    with pytest.raises(
        LangfusePullError,
        match="Langfuse pagination exceeded maximum page count",
    ):
        pull_observations(
            output_path,
            config=_config(),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
        )

    assert calls == 4
    assert output_path.read_bytes() == original_bytes
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


def test_pull_observations_retries_rate_limit_once_with_numeric_retry_after(tmp_path: Path) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    sleeps: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return _response([_row("obs-1", "trace-1", "2026-08-01T00:00:00Z")], None)

    count = pull_observations(
        output_path,
        config=_config(),
        options=PullOptions(),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    assert count == 1
    assert calls == 2
    assert sleeps == [2.0]


@pytest.mark.parametrize(
    ("retry_after", "use_bytes"),
    [
        (None, False),
        ("later", False),
        ("-1", False),
        ("1.5", False),
        ("nan", False),
        ("inf", False),
        ("１２", True),
        ("9" * 10000, False),
    ],
)
def test_pull_observations_retries_429_with_zero_delay_on_invalid_retry_after(
    tmp_path: Path,
    retry_after: str | None,
    use_bytes: bool,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    sleeps: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            if retry_after is None:
                return httpx.Response(429)
            if use_bytes:
                return httpx.Response(429, headers=[(b"Retry-After", retry_after.encode("utf-8"))])
            headers = {"Retry-After": retry_after}
            return httpx.Response(429, headers=headers)
        return _response([_row("obs-1", "trace-1", "2026-08-01T00:00:00Z")], None)

    count = pull_observations(
        output_path,
        config=_config(),
        options=PullOptions(),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    assert count == 1
    assert calls == 2
    assert sleeps == [0.0]


def test_pull_observations_maps_directory_creation_failure_before_http(tmp_path: Path) -> None:
    output_path = tmp_path / "missing" / "langfuse.jsonl"
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _response([], None)

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("mkdir failed")

    class _BrokenPath(type(output_path)):
        def mkdir(self, *args: object, **kwargs: object) -> None:
            explode(*args, **kwargs)

    broken_output = _BrokenPath(output_path)

    try:
        with pytest.raises(LangfusePullError, match="Failed to write Langfuse snapshot"):
            pull_observations(
                broken_output,
                config=_config(),
                options=PullOptions(),
                transport=httpx.MockTransport(handler),
            )
    finally:
        pass

    assert called is False


def test_pull_observations_fails_after_second_rate_limit_response(tmp_path: Path) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1"})

    with pytest.raises(LangfusePullError, match="Langfuse rate limit persisted after retry"):
        pull_observations(
            output_path,
            config=_config(),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
            sleep=sleeps.append,
        )

    assert sleeps == [1.0]
    assert not output_path.exists()
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


@pytest.mark.parametrize(
    ("status_code", "expected_message"),
    [
        (401, "Langfuse authentication failed"),
        (403, "Langfuse project access denied"),
        (
            404,
            "Langfuse observations endpoint not found; "
            "this requires Langfuse Cloud/self-hosted v4 Public API v2",
        ),
        (500, "Langfuse request failed with status 500"),
        (502, "Langfuse request failed with status 502"),
        (503, "Langfuse request failed with status 503"),
    ],
)
def test_pull_observations_surfaces_sanitized_http_errors(
    tmp_path: Path,
    status_code: int,
    expected_message: str,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            text=(
                f"secret={SECRET_KEY} trace=private-content "
                f"auth={request.headers['Authorization']}"
            ),
        )

    with pytest.raises(LangfusePullError, match=expected_message) as exc_info:
        pull_observations(
            output_path,
            config=_config(),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
        )

    text = str(exc_info.value)
    assert SECRET_KEY not in text
    assert PUBLIC_KEY not in text
    assert "private-content" not in text
    assert "Authorization" not in text
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("timed out"),
        httpx.WriteTimeout("timed out"),
        httpx.ConnectTimeout("timed out"),
        httpx.PoolTimeout("timed out"),
    ],
)
def test_pull_observations_sanitizes_all_timeout_exceptions(
    tmp_path: Path,
    error: httpx.TimeoutException,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(
        LangfusePullError,
        match="Langfuse request timed out",
    ) as exc_info:
        pull_observations(
            output_path,
            config=_config(),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
        )

    text = str(exc_info.value)
    assert SECRET_KEY not in text
    assert PUBLIC_KEY not in text
    assert "timed out" in text
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


def test_pull_observations_sanitizes_network_errors(tmp_path: Path) -> None:
    output_path = tmp_path / "langfuse.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    with pytest.raises(
        LangfusePullError,
        match="Langfuse network request failed",
    ) as exc_info:
        pull_observations(
            output_path,
            config=_config(),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
        )

    text = str(exc_info.value)
    assert SECRET_KEY not in text
    assert PUBLIC_KEY not in text
    assert "network down" not in text
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


def test_pull_observations_maps_invalid_base_url_to_sanitized_error(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    output_path.write_text('{"stable":true}\n', encoding="utf-8")

    with pytest.raises(LangfusePullError, match="Langfuse request failed") as exc_info:
        pull_observations(
            output_path,
            config=_config("https://[::1"),
            options=PullOptions(),
        )

    error = exc_info.value
    chain = _exception_chain(error)
    chain_text = _exception_chain_text(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert SECRET_KEY not in chain_text
    assert PUBLIC_KEY not in chain_text
    assert "Authorization" not in chain_text
    assert all(not hasattr(link, "request") for link in chain)
    assert output_path.read_text(encoding="utf-8") == '{"stable":true}\n'
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


def test_pull_observations_does_not_chain_httpx_exceptions_with_credentials(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    errors = {
        "timeout": _timeout_with_request,
        "network": _connect_error_with_request,
    }

    for label, build_error in errors.items():
        def handler(
            request: httpx.Request,
            error_builder: Callable[[str, httpx.Request], Exception] = build_error,
            message: str = label,
        ) -> httpx.Response:
            raise error_builder(message, request)

        with pytest.raises(LangfusePullError) as exc_info:
            pull_observations(
                output_path,
                config=_config(),
                options=PullOptions(),
                transport=httpx.MockTransport(handler),
            )

        error = exc_info.value
        chain = _exception_chain(error)
        chain_text = _exception_chain_text(error)
        assert error.__cause__ is None
        assert error.__context__ is None
        assert SECRET_KEY not in chain_text
        assert PUBLIC_KEY not in chain_text
        assert "Authorization" not in chain_text
        assert all(not hasattr(link, "request") for link in chain)
        assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


@pytest.mark.parametrize(
    ("response", "expected_message"),
    [
        (
            httpx.Response(200, text="not-json"),
            "Langfuse response was not valid JSON",
        ),
        (
            httpx.Response(200, content=b'\x80{"data":[],"meta":{"cursor":null}}'),
            "Langfuse response was not valid JSON",
        ),
        (
            httpx.Response(200, json=[1, 2, 3]),
            "Langfuse response must be a JSON object",
        ),
        (
            httpx.Response(200, json={"meta": {"cursor": None}}),
            "Langfuse response data must be a list",
        ),
        (
            httpx.Response(200, json={"data": [], "meta": []}),
            "Langfuse response meta must be an object",
        ),
        (
            httpx.Response(200, json={"data": [], "meta": {"cursor": 5}}),
            "Langfuse response cursor must be a string or null",
        ),
        (
            httpx.Response(200, json={"data": [123], "meta": {"cursor": None}}),
            "Langfuse observation rows must be JSON objects",
        ),
        (
            httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "obs-1", "traceId": "trace-1", "startTime": ""}
                    ],
                    "meta": {"cursor": None},
                },
            ),
            "Langfuse observation rows must include id, traceId, and startTime",
        ),
        (
            httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "obs-1",
                            "traceId": "trace-1",
                            "startTime": "2026-08-01T00:00:00Z",
                        }
                    ],
                    "meta": {"cursor": None},
                },
            ),
            "Langfuse observation rows must include a discriminator",
        ),
    ],
)
def test_pull_observations_rejects_malformed_responses_and_cleans_up_temp_file(
    tmp_path: Path,
    response: httpx.Response,
    expected_message: str,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    output_path.write_text('{"stable":true}\n', encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return response

    with pytest.raises(LangfusePullError, match=expected_message):
        pull_observations(
            output_path,
            config=_config(),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
        )

    assert output_path.read_text(encoding="utf-8") == '{"stable":true}\n'
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


def test_pull_observations_maps_oversized_json_number_to_sanitized_error(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "langfuse.jsonl"
    output_path.write_text('{"stable":true}\n', encoding="utf-8")
    oversized_number = "1" * 5000
    body = (
        "{\"data\":[{\"id\":"
        "\"obs-1\",\"traceId\":\"trace-1\",\"startTime\":\"2026-08-01T00:00:00Z\","
        "\"type\":\"GENERATION\",\"input\":"
        f"{oversized_number}"
        "}],\"meta\":{\"cursor\":null}}"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    with pytest.raises(LangfusePullError, match="Langfuse response was not valid JSON") as exc_info:
        pull_observations(
            output_path,
            config=_config(),
            options=PullOptions(),
            transport=httpx.MockTransport(handler),
        )

    error = exc_info.value
    chain = _exception_chain(error)
    chain_text = _exception_chain_text(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert SECRET_KEY not in chain_text
    assert PUBLIC_KEY not in chain_text
    assert "Authorization" not in chain_text
    assert all(not hasattr(link, "request") for link in chain)
    assert output_path.read_text(encoding="utf-8") == '{"stable":true}\n'
    assert list(tmp_path.glob("langfuse.jsonl.*.tmp")) == []


def test_langfuse_config_and_error_repr_do_not_leak_secret() -> None:
    config = _config()
    error = LangfusePullError("safe message")

    assert SECRET_KEY not in repr(config)
    assert SECRET_KEY not in repr(error)
    assert "Authorization" not in repr(error)


def test_pull_observations_defaults_to_time_sleep() -> None:
    defaults = pull_observations.__kwdefaults__
    assert defaults is not None
    assert defaults["sleep"] is time.sleep
