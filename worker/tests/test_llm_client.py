"""Tests for the LLM chat-completion dispatcher.

All HTTP is mocked via :class:`httpx.MockTransport`. No live vMLX calls
in unit tests — a manual smoke against the live server is a separate
one-shot action, not a CI dependency.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

from clawfeed_intel.llm import (
    CallResult,
    LLMClient,
    LLMSchemaError,
    RetryConfig,
    RoutingConfig,
)
from clawfeed_intel.llm.client import (
    _hash_content,
    _hash_messages,
    _is_transient_error,
    _parse_response,
    _short_error,
    _validate_schema,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def routing() -> RoutingConfig:
    return RoutingConfig.model_validate(
        {
            "providers": {
                "vmlx": {"base_url": "http://127.0.0.1:8080/v1"},
            },
            "stages": {
                "source_planning": {
                    "provider": "vmlx",
                    "model": "mlx-community/Qwen3-8B-4bit",
                    "timeout_seconds": 30,
                },
                "relevance_filter": {
                    "provider": "vmlx",
                    "model": "mlx-community/Qwen3.5-27B-4bit",
                    "timeout_seconds": 60,
                    "batch_size": 12,
                },
            },
        }
    )


@pytest.fixture
def fast_retry() -> RetryConfig:
    """Zero-wait retry config so tests don't sleep through real backoff."""
    return RetryConfig(max_attempts=3, wait_multiplier=0, wait_min_seconds=0, wait_max_seconds=0)


@pytest.fixture
def no_retry() -> RetryConfig:
    """Single-attempt config for tests that want raw 8a behaviour."""
    return RetryConfig(max_attempts=1, wait_multiplier=0, wait_min_seconds=0, wait_max_seconds=0)


def _ok_response(
    *,
    content: str,
    model: str,
    prompt_tokens: int = 7,
    completion_tokens: int = 5,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_handler(
    responses: list[Callable[[httpx.Request], httpx.Response]],
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    """Build a MockTransport handler that walks through ``responses`` in order.

    Returns ``(handler, captured_requests)``. ``captured_requests`` is a
    mutable list filled with the requests as they arrive — useful for
    asserting on retry/repair attempt count and per-attempt body shape.
    """
    captured: list[httpx.Request] = []
    cursor = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        i = cursor["i"]
        cursor["i"] = min(i + 1, len(responses) - 1)
        return responses[i](request)

    return handler, captured


# ── Pure: _parse_response ─────────────────────────────────────────────────


def test_parse_response_extracts_content_and_model() -> None:
    payload = _ok_response(content="ok", model="m1", prompt_tokens=3, completion_tokens=4)
    result = _parse_response(payload, fallback_model="fallback")
    assert result.content == "ok"
    assert result.model == "m1"


def test_parse_response_uses_fallback_model_when_response_omits_it() -> None:
    payload = _ok_response(content="ok", model="m1")
    del payload["model"]
    result = _parse_response(payload, fallback_model="stage-default")
    assert result.model == "stage-default"


def test_parse_response_no_choices_raises() -> None:
    with pytest.raises(ValueError, match="no choices"):
        _parse_response({"choices": []}, fallback_model="m")


def test_parse_response_missing_choices_key_raises() -> None:
    with pytest.raises(ValueError, match="no choices"):
        _parse_response({}, fallback_model="m")


def test_parse_response_missing_content_raises() -> None:
    payload = {"choices": [{"message": {"role": "assistant"}}]}
    with pytest.raises(ValueError, match="missing message.content"):
        _parse_response(payload, fallback_model="m")


def test_parse_response_empty_string_content_kept() -> None:
    """An empty string is a valid (if useless) completion — don't conflate with None."""
    payload = _ok_response(content="", model="m")
    result = _parse_response(payload, fallback_model="m")
    assert result.content == ""


# ── Pure: _validate_schema ────────────────────────────────────────────────


class _ExampleSchema(BaseModel):
    keep: bool
    score: float = Field(ge=0, le=1)
    reason: str


def test_validate_schema_happy_path() -> None:
    parsed = _validate_schema('{"keep": true, "score": 0.8, "reason": "matches"}', _ExampleSchema)
    assert isinstance(parsed, _ExampleSchema)
    assert parsed.keep is True
    assert parsed.score == 0.8


def test_validate_schema_rejects_invalid_json() -> None:
    with pytest.raises(LLMSchemaError):
        _validate_schema("not json", _ExampleSchema)


def test_validate_schema_rejects_schema_violation() -> None:
    """Score out of bounds — pydantic ValidationError → LLMSchemaError."""
    with pytest.raises(LLMSchemaError):
        _validate_schema('{"keep": true, "score": 5.0, "reason": "too high"}', _ExampleSchema)


def test_validate_schema_rejects_missing_field() -> None:
    with pytest.raises(LLMSchemaError):
        _validate_schema('{"keep": true, "score": 0.5}', _ExampleSchema)


# ── Pure: _is_transient_error ─────────────────────────────────────────────


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://example/x")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


@pytest.mark.parametrize("status_code", [500, 502, 503, 504, 429])
def test_is_transient_for_5xx_and_429(status_code: int) -> None:
    assert _is_transient_error(_http_error(status_code))


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_is_not_transient_for_other_4xx(status_code: int) -> None:
    assert not _is_transient_error(_http_error(status_code))


def test_is_transient_for_connect_timeout() -> None:
    assert _is_transient_error(httpx.ConnectTimeout("connect"))


def test_is_transient_for_read_timeout() -> None:
    assert _is_transient_error(httpx.ReadTimeout("read"))


def test_is_transient_for_connect_error() -> None:
    assert _is_transient_error(httpx.ConnectError("dns failed"))


def test_is_not_transient_for_arbitrary_exception() -> None:
    assert not _is_transient_error(ValueError("not a transport error"))


# ── Pure: hashing ─────────────────────────────────────────────────────────


def test_hash_messages_is_deterministic() -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert _hash_messages(messages) == _hash_messages(messages)


def test_hash_messages_distinguishes_different_content() -> None:
    a = _hash_messages([{"role": "user", "content": "hi"}])
    b = _hash_messages([{"role": "user", "content": "bye"}])
    assert a != b


def test_hash_messages_stable_across_dict_key_order() -> None:
    """Sorting message dict keys defends against insertion-order drift."""
    a = _hash_messages([{"role": "user", "content": "hi"}])
    b = _hash_messages([{"content": "hi", "role": "user"}])
    assert a == b


def test_hash_messages_returns_64_char_hex() -> None:
    digest = _hash_messages([{"role": "user", "content": "x"}])
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_hash_content_is_deterministic() -> None:
    assert _hash_content("foo") == _hash_content("foo")


def test_hash_content_distinguishes_inputs() -> None:
    assert _hash_content("foo") != _hash_content("bar")


# ── Pure: _short_error ────────────────────────────────────────────────────


def test_short_error_includes_type_and_message() -> None:
    msg = _short_error(ValueError("boom"))
    assert msg.startswith("ValueError")
    assert "boom" in msg


def test_short_error_truncates_long_messages() -> None:
    long_msg = "x" * 500
    msg = _short_error(ValueError(long_msg))
    assert len(msg) < 320  # type-prefix + 280 chars + ellipsis
    assert msg.endswith("...")


def test_short_error_handles_empty_message() -> None:
    msg = _short_error(RuntimeError())
    assert msg == "RuntimeError"


# ── chat_completion: happy path ───────────────────────────────────────────


async def test_chat_completion_returns_call_result(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_ok_response(content="PONG", model="mlx-community/Qwen3.5-27B-4bit"),
        )

    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=no_retry)
    result = await client.chat_completion(
        "relevance_filter",
        [{"role": "user", "content": "ping"}],
    )

    assert isinstance(result, CallResult)
    assert result.content == "PONG"
    assert result.model == "mlx-community/Qwen3.5-27B-4bit"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 5
    assert result.parsed is None
    assert result.latency_ms >= 0


async def test_chat_completion_targets_correct_url(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(200, json=_ok_response(content="ok", model="m"))

    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=no_retry)
    await client.chat_completion("source_planning", [{"role": "user", "content": "hi"}])

    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"


async def test_chat_completion_body_uses_stage_model(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    """The body's ``model`` comes from the stage config, not the caller's prompt."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json=_ok_response(content="ok", model="echo"))

    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=no_retry)
    await client.chat_completion("source_planning", [{"role": "user", "content": "hi"}])

    assert captured["body"]["model"] == "mlx-community/Qwen3-8B-4bit"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    # No temperature unless the caller asks for one — leave the provider's
    # default in place by omitting the field entirely.
    assert "temperature" not in captured["body"]


async def test_chat_completion_forwards_temperature_when_set(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    """Structured-output prompts (relevance filter, cluster summary) pin a
    low temperature to keep JSON well-formed. Verify the value lands in the
    request body verbatim.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json=_ok_response(content="ok", model="m"))

    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=no_retry)
    await client.chat_completion(
        "source_planning",
        [{"role": "user", "content": "hi"}],
        temperature=0.1,
    )

    assert captured["body"]["temperature"] == 0.1


async def test_chat_completion_temperature_zero_is_forwarded(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    """``temperature=0.0`` is a valid pin (deterministic decoding) and must
    not be confused with the ``None`` default.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json=_ok_response(content="ok", model="m"))

    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=no_retry)
    await client.chat_completion(
        "source_planning",
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
    )

    assert captured["body"]["temperature"] == 0.0


async def test_chat_completion_forwards_max_tokens_when_set(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    """Batched-JSON stages must size the response budget to the batch.
    Local MLX defaults to ~1024 tokens which truncates verdict arrays.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json=_ok_response(content="ok", model="m"))

    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=no_retry)
    await client.chat_completion(
        "source_planning",
        [{"role": "user", "content": "hi"}],
        max_tokens=4096,
    )

    assert captured["body"]["max_tokens"] == 4096


async def test_chat_completion_max_tokens_omitted_when_unset(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    """The default must omit max_tokens entirely — providers' defaults
    vary and we don't want to impose a global ceiling from the client.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json=_ok_response(content="ok", model="m"))

    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=no_retry)
    await client.chat_completion("source_planning", [{"role": "user", "content": "hi"}])

    assert "max_tokens" not in captured["body"]


async def test_chat_completion_routes_per_stage(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    """Different stages → different models in the request body."""
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        seen_models.append(body["model"])
        return httpx.Response(200, json=_ok_response(content="ok", model=body["model"]))

    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=no_retry)
    await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])
    await client.chat_completion("relevance_filter", [{"role": "user", "content": "x"}])

    assert seen_models == [
        "mlx-community/Qwen3-8B-4bit",
        "mlx-community/Qwen3.5-27B-4bit",
    ]


async def test_chat_completion_trailing_slash_in_base_url_normalized(
    no_retry: RetryConfig,
) -> None:
    """``base_url`` with a trailing slash should still produce a clean endpoint URL."""
    routing = RoutingConfig.model_validate(
        {
            "providers": {"vmlx": {"base_url": "http://127.0.0.1:8080/v1/"}},
            "stages": {
                "source_planning": {
                    "provider": "vmlx",
                    "model": "m",
                    "timeout_seconds": 5,
                },
            },
        }
    )
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json=_ok_response(content="ok", model="m"))

    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=no_retry)
    await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])

    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"


# ── chat_completion: failure modes (no retry) ─────────────────────────────


async def test_chat_completion_unknown_stage_raises(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json=_ok_response(content="x", model="x"))
    )
    client = LLMClient(routing, transport=transport, retry_config=no_retry)
    with pytest.raises(KeyError):
        await client.chat_completion("not_a_stage", [{"role": "user", "content": "x"}])


async def test_chat_completion_4xx_propagates_immediately(
    routing: RoutingConfig, fast_retry: RetryConfig
) -> None:
    """400 is not transient — should fail on the first attempt without retrying."""
    handler, captured = _make_handler(
        [lambda r: httpx.Response(400, json={"error": "bad request"})]
    )
    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=fast_retry)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])
    assert len(captured) == 1, "4xx-non-429 must not retry"


async def test_chat_completion_no_choices_raises(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"choices": [], "usage": {}})
    )
    client = LLMClient(routing, transport=transport, retry_config=no_retry)
    with pytest.raises(ValueError, match="no choices"):
        await client.chat_completion(
            "source_planning",
            [{"role": "user", "content": "x"}],
        )


async def test_chat_completion_missing_content_raises(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    bad = {"choices": [{"message": {"role": "assistant"}}], "usage": {}}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=bad))
    client = LLMClient(routing, transport=transport, retry_config=no_retry)
    with pytest.raises(ValueError, match="missing message.content"):
        await client.chat_completion(
            "source_planning",
            [{"role": "user", "content": "x"}],
        )


async def test_chat_completion_handles_missing_usage(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    """vMLX always emits usage, but we shouldn't crash if a future provider doesn't."""
    response = _ok_response(content="ok", model="m")
    del response["usage"]
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=response))
    client = LLMClient(routing, transport=transport, retry_config=no_retry)
    result = await client.chat_completion(
        "source_planning",
        [{"role": "user", "content": "x"}],
    )
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


# ── chat_completion: retry behaviour ──────────────────────────────────────


async def test_retry_on_5xx_then_succeeds(routing: RoutingConfig, fast_retry: RetryConfig) -> None:
    """500 once, then 200: should retry and return the recovered result."""
    handler, captured = _make_handler(
        [
            lambda r: httpx.Response(500, text="boom"),
            lambda r: httpx.Response(200, json=_ok_response(content="recovered", model="m")),
        ]
    )
    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=fast_retry)
    result = await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])

    assert result.content == "recovered"
    assert len(captured) == 2


async def test_retry_on_429_then_succeeds(routing: RoutingConfig, fast_retry: RetryConfig) -> None:
    """429 (rate-limit) is treated as transient."""
    handler, captured = _make_handler(
        [
            lambda r: httpx.Response(429, text="slow down"),
            lambda r: httpx.Response(200, json=_ok_response(content="ok", model="m")),
        ]
    )
    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=fast_retry)
    result = await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])

    assert result.content == "ok"
    assert len(captured) == 2


async def test_retry_exhausted_propagates(routing: RoutingConfig, fast_retry: RetryConfig) -> None:
    """All 3 attempts return 5xx → final HTTPStatusError propagates."""
    handler, captured = _make_handler([lambda r: httpx.Response(500, text="boom")])
    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=fast_retry)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])
    assert len(captured) == 3, "should attempt max_attempts times"


async def test_4xx_not_429_does_not_retry(routing: RoutingConfig, fast_retry: RetryConfig) -> None:
    """403 must not retry (caller-side, won't change between attempts)."""
    handler, captured = _make_handler([lambda r: httpx.Response(403, text="nope")])
    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=fast_retry)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])
    assert len(captured) == 1


async def test_max_attempts_one_disables_retry(
    routing: RoutingConfig,
) -> None:
    """``max_attempts=1`` reproduces 8a behaviour."""
    config = RetryConfig(max_attempts=1, wait_multiplier=0, wait_min_seconds=0, wait_max_seconds=0)
    handler, captured = _make_handler([lambda r: httpx.Response(500)])
    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=config)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])
    assert len(captured) == 1


# ── chat_completion: schema validation + repair ───────────────────────────


async def test_schema_validation_happy_path_populates_parsed(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    valid_json = '{"keep": true, "score": 0.9, "reason": "ok"}'
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json=_ok_response(content=valid_json, model="m"))
    )
    client = LLMClient(routing, transport=transport, retry_config=no_retry)
    result = await client.chat_completion(
        "source_planning",
        [{"role": "user", "content": "judge this"}],
        response_schema=_ExampleSchema,
    )

    assert isinstance(result.parsed, _ExampleSchema)
    assert result.parsed.keep is True
    assert result.parsed.score == 0.9


async def test_schema_validation_repair_fires_on_invalid_json(
    routing: RoutingConfig, fast_retry: RetryConfig
) -> None:
    """First response is malformed; repair attempt returns valid JSON."""
    valid_json = '{"keep": false, "score": 0.1, "reason": "fixed"}'
    handler, captured = _make_handler(
        [
            lambda r: httpx.Response(
                200,
                json=_ok_response(content="not valid json at all", model="m"),
            ),
            lambda r: httpx.Response(
                200,
                json=_ok_response(content=valid_json, model="m"),
            ),
        ]
    )
    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=fast_retry)
    result = await client.chat_completion(
        "source_planning",
        [{"role": "user", "content": "judge"}],
        response_schema=_ExampleSchema,
    )

    assert len(captured) == 2, "repair attempt should have fired"
    assert isinstance(result.parsed, _ExampleSchema)
    assert result.parsed.keep is False
    # Surfaced ``content`` should reflect the repaired response, not the bad one.
    assert result.content == valid_json


async def test_schema_validation_repair_includes_correction_prompt(
    routing: RoutingConfig, fast_retry: RetryConfig
) -> None:
    """The repair attempt's request body should include the original messages
    plus the assistant's bad reply plus a corrective user message."""
    valid_json = '{"keep": false, "score": 0.0, "reason": "x"}'
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        captured_bodies.append(body)
        if len(captured_bodies) == 1:
            return httpx.Response(
                200,
                json=_ok_response(content="garbage", model="m"),
            )
        return httpx.Response(200, json=_ok_response(content=valid_json, model="m"))

    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=fast_retry)
    await client.chat_completion(
        "source_planning",
        [{"role": "user", "content": "judge"}],
        response_schema=_ExampleSchema,
    )

    repair_messages = captured_bodies[1]["messages"]
    # original user + assistant's bad reply + repair user prompt
    assert len(repair_messages) == 3
    assert repair_messages[0] == {"role": "user", "content": "judge"}
    assert repair_messages[1] == {"role": "assistant", "content": "garbage"}
    assert repair_messages[2]["role"] == "user"
    assert "JSON" in repair_messages[2]["content"]


async def test_schema_validation_repair_failure_raises(
    routing: RoutingConfig, fast_retry: RetryConfig
) -> None:
    """Repair attempt also returns invalid JSON → LLMSchemaError propagates."""
    handler, captured = _make_handler(
        [lambda r: httpx.Response(200, json=_ok_response(content="garbage", model="m"))]
    )
    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=fast_retry)
    with pytest.raises(LLMSchemaError, match="after repair"):
        await client.chat_completion(
            "source_planning",
            [{"role": "user", "content": "x"}],
            response_schema=_ExampleSchema,
        )
    # Repair attempt counts as a second logical request — but each request
    # may get retried up to max_attempts times. Both bodies returned
    # garbage which is non-transient (200 OK), so each fires once.
    assert len(captured) == 2


async def test_no_repair_when_no_schema(routing: RoutingConfig, fast_retry: RetryConfig) -> None:
    """Without a schema, malformed content is returned as-is — never repaired."""
    handler, captured = _make_handler(
        [lambda r: httpx.Response(200, json=_ok_response(content="garbage", model="m"))]
    )
    client = LLMClient(routing, transport=httpx.MockTransport(handler), retry_config=fast_retry)
    result = await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])

    assert result.content == "garbage"
    assert result.parsed is None
    assert len(captured) == 1, "no schema → no repair"


# ── chat_completion: DB logging ───────────────────────────────────────────


@pytest.fixture
def conn(temp_db: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection on the migrated temp DB. Module-local override of the
    conftest ``conn`` fixture which yields a throwaway ``:memory:`` connection."""
    from clawfeed_intel import db as worker_db

    with closing(worker_db.connect(temp_db)) as c:
        yield c


def _create_run(conn: sqlite3.Connection) -> int:
    from clawfeed_intel.db import create_run

    return create_run(
        conn,
        run_type="daily",
        window_start="2026-05-06T00:00:00+00:00",
        window_end="2026-05-07T00:00:00+00:00",
    )


def _list_calls(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM llm_calls ORDER BY id ASC")]


async def test_logs_successful_call_to_llm_calls(
    conn: sqlite3.Connection, routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    run_id = _create_run(conn)
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200, json=_ok_response(content="ok", model="m", prompt_tokens=11, completion_tokens=3)
        )
    )
    client = LLMClient(
        routing,
        transport=transport,
        retry_config=no_retry,
        conn=conn,
        run_id=run_id,
    )
    messages = [{"role": "user", "content": "hi"}]
    await client.chat_completion("relevance_filter", messages, prompt_version="v1")

    rows = _list_calls(conn)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "succeeded"
    assert row["stage"] == "relevance_filter"
    assert row["provider"] == "vmlx"
    assert row["model"] == "mlx-community/Qwen3.5-27B-4bit"
    assert row["run_id"] == run_id
    assert row["prompt_version"] == "v1"
    assert row["prompt_tokens"] == 11
    assert row["completion_tokens"] == 3
    assert row["input_hash"] == _hash_messages(messages)
    assert row["output_hash"] == _hash_content("ok")
    assert row["error"] is None


async def test_logs_failed_call_to_llm_calls(
    conn: sqlite3.Connection, routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    run_id = _create_run(conn)
    transport = httpx.MockTransport(lambda r: httpx.Response(400, text="bad"))
    client = LLMClient(
        routing, transport=transport, retry_config=no_retry, conn=conn, run_id=run_id
    )
    messages = [{"role": "user", "content": "x"}]

    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_completion("source_planning", messages)

    rows = _list_calls(conn)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "failed"
    assert row["stage"] == "source_planning"
    assert row["error"] is not None
    assert "HTTPStatusError" in row["error"]
    assert row["input_hash"] == _hash_messages(messages)
    assert row["output_hash"] is None
    assert row["prompt_tokens"] == 0
    assert row["completion_tokens"] == 0


async def test_no_db_logging_when_conn_omitted(
    routing: RoutingConfig, no_retry: RetryConfig
) -> None:
    """Without a connection, the client must not raise on missing DB."""
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json=_ok_response(content="ok", model="m"))
    )
    client = LLMClient(routing, transport=transport, retry_config=no_retry)
    result = await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])
    assert result.content == "ok"


async def test_logging_failure_does_not_mask_call_result(
    routing: RoutingConfig, no_retry: RetryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the DB write fails, the call result is still returned. The audit log
    is best-effort — it must not crash a successful inference."""

    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json=_ok_response(content="ok", model="m"))
    )

    from clawfeed_intel.llm import client as client_module

    def boom(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("DB exploded")

    monkeypatch.setattr(client_module.db, "record_llm_call", boom)

    # Pass a sentinel non-None conn so the client tries to log.
    fake_conn = object()
    llm = LLMClient(
        routing,
        transport=transport,
        retry_config=no_retry,
        conn=fake_conn,  # type: ignore[arg-type]
        run_id=1,
    )
    result = await llm.chat_completion("source_planning", [{"role": "user", "content": "x"}])
    assert result.content == "ok"


async def test_retry_with_logging_records_one_row_per_logical_call(
    conn: sqlite3.Connection, routing: RoutingConfig, fast_retry: RetryConfig
) -> None:
    """Three HTTP attempts under the hood → one row in llm_calls."""
    run_id = _create_run(conn)
    handler, captured = _make_handler(
        [
            lambda r: httpx.Response(500),
            lambda r: httpx.Response(500),
            lambda r: httpx.Response(200, json=_ok_response(content="recovered", model="m")),
        ]
    )
    client = LLMClient(
        routing,
        transport=httpx.MockTransport(handler),
        retry_config=fast_retry,
        conn=conn,
        run_id=run_id,
    )
    await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])

    assert len(captured) == 3
    rows = _list_calls(conn)
    assert len(rows) == 1
    assert rows[0]["status"] == "succeeded"


# ── Step 12b: gemini_cli dispatch via LLMClient ─────────────────────────────


@pytest.fixture
def routing_with_gemini() -> RoutingConfig:
    """Routing config that has both providers and routes final_compose
    to gemini_cli. CLI dispatch tests use this; the actual subprocess
    is monkeypatched in each test so nothing is invoked for real.
    """
    return RoutingConfig.model_validate(
        {
            "providers": {
                "vmlx": {"base_url": "http://127.0.0.1:8080/v1"},
                "gemini_cli": {"script_path": "/fake/gemini"},
            },
            "stages": {
                "source_planning": {
                    "provider": "vmlx",
                    "model": "stub-vmlx",
                    "timeout_seconds": 30,
                },
                "final_compose": {
                    "provider": "gemini_cli",
                    "model": "gemini-3-pro-preview",
                    "timeout_seconds": 300,
                },
            },
        }
    )


async def test_cli_dispatch_happy_path(
    monkeypatch: pytest.MonkeyPatch, routing_with_gemini: RoutingConfig
) -> None:
    """Stage routed to gemini_cli → LLMClient calls
    ``gemini_cli_completion`` and returns its content + model.
    """
    from clawfeed_intel.llm import gemini_cli as gem_module
    from clawfeed_intel.llm.gemini_cli import GeminiCliResult

    captured: dict[str, Any] = {}

    async def fake_completion(config, *, messages, model):
        captured["config"] = config
        captured["messages"] = messages
        captured["model"] = model
        return GeminiCliResult(
            content="# Compose output",
            model=model,
            latency_ms=1234,
            prompt_tokens=111,
            completion_tokens=222,
            attempts=1,
        )

    monkeypatch.setattr(gem_module, "gemini_cli_completion", fake_completion)
    monkeypatch.setattr("clawfeed_intel.llm.client.gemini_cli_completion", fake_completion)

    client = LLMClient(routing_with_gemini)
    result = await client.chat_completion(
        "final_compose",
        messages=[{"role": "user", "content": "compose this"}],
    )

    assert result.content == "# Compose output"
    assert result.model == "gemini-3-pro-preview"
    assert result.prompt_tokens == 111
    assert result.completion_tokens == 222
    assert captured["model"] == "gemini-3-pro-preview"
    assert captured["config"].script_path == "/fake/gemini"


async def test_cli_dispatch_writes_audit_row_with_gemini_provider(
    monkeypatch: pytest.MonkeyPatch,
    conn: sqlite3.Connection,
    routing_with_gemini: RoutingConfig,
) -> None:
    """Audit row stamps ``provider='gemini_cli'`` so the dashboard can
    distinguish frontier calls from local-model calls.
    """
    from clawfeed_intel.llm.gemini_cli import GeminiCliResult

    async def fake_completion(config, *, messages, model):
        return GeminiCliResult(
            content="ok", model=model, latency_ms=10, prompt_tokens=1, completion_tokens=1
        )

    monkeypatch.setattr("clawfeed_intel.llm.client.gemini_cli_completion", fake_completion)

    run_id = _create_run(conn)
    client = LLMClient(routing_with_gemini, conn=conn, run_id=run_id)
    await client.chat_completion("final_compose", messages=[{"role": "user", "content": "x"}])

    rows = _list_calls(conn)
    assert len(rows) == 1
    assert rows[0]["provider"] == "gemini_cli"
    assert rows[0]["model"] == "gemini-3-pro-preview"
    assert rows[0]["status"] == "succeeded"


async def test_cli_dispatch_with_response_schema_raises(
    routing_with_gemini: RoutingConfig,
) -> None:
    """Asking for schema validation on a CLI stage is a programmer error.

    The CLI emits free-form text; pretending to do JSON-schema repair
    would hide the actual contract from the caller.
    """
    from pydantic import BaseModel

    class StubSchema(BaseModel):
        x: int

    client = LLMClient(routing_with_gemini)
    with pytest.raises(ValueError, match="does not support response_schema"):
        await client.chat_completion(
            "final_compose",
            messages=[{"role": "user", "content": "x"}],
            response_schema=StubSchema,
        )


async def test_cli_dispatch_failure_records_audit_row(
    monkeypatch: pytest.MonkeyPatch,
    conn: sqlite3.Connection,
    routing_with_gemini: RoutingConfig,
) -> None:
    """When the CLI provider fails, the audit row records failure +
    error string so post-hoc inspection can find what broke.
    """
    from clawfeed_intel.llm.gemini_cli import GeminiCliExitError

    async def fake_completion(config, *, messages, model):
        raise GeminiCliExitError(1, "oauth refresh failed")

    monkeypatch.setattr("clawfeed_intel.llm.client.gemini_cli_completion", fake_completion)

    run_id = _create_run(conn)
    client = LLMClient(routing_with_gemini, conn=conn, run_id=run_id)
    with pytest.raises(GeminiCliExitError):
        await client.chat_completion("final_compose", messages=[{"role": "user", "content": "x"}])

    rows = _list_calls(conn)
    assert len(rows) == 1
    assert rows[0]["provider"] == "gemini_cli"
    assert rows[0]["status"] == "failed"
    assert "oauth refresh failed" in rows[0]["error"]


async def test_cli_dispatch_raises_when_provider_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage that routes to gemini_cli but the providers block doesn't
    declare it → deploy-time bug, raised loudly at first call.
    """
    routing = RoutingConfig.model_validate(
        {
            "providers": {"vmlx": {"base_url": "http://x/v1"}},
            "stages": {
                "final_compose": {
                    "provider": "gemini_cli",
                    "model": "gemini-3-pro-preview",
                    "timeout_seconds": 300,
                },
            },
        }
    )
    client = LLMClient(routing)
    with pytest.raises(ValueError, match="providers.gemini_cli is not declared"):
        await client.chat_completion("final_compose", messages=[{"role": "user", "content": "x"}])


async def test_stage_config_override_dispatches_against_override(
    monkeypatch: pytest.MonkeyPatch, routing: RoutingConfig
) -> None:
    """``stage_config_override`` is what compose_brief's Tier-2 fallback uses.

    Routing has source_planning → vmlx. We dispatch against
    ``source_planning`` but pass an override pointing at a different
    vmlx model; the HTTP request body must carry the override's
    model, not the routed model.
    """
    from clawfeed_intel.llm import StageConfig

    body_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        body_models.append(body["model"])
        return httpx.Response(200, json=_ok_response(content="ok", model=body["model"]))

    client = LLMClient(routing, transport=httpx.MockTransport(handler))
    override = StageConfig(provider="vmlx", model="override-model", timeout_seconds=30)
    result = await client.chat_completion(
        "source_planning",
        messages=[{"role": "user", "content": "x"}],
        stage_config_override=override,
    )

    assert body_models == ["override-model"]
    assert result.model == "override-model"
