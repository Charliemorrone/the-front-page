"""Tests for the LLM chat-completion dispatcher.

All HTTP is mocked via :class:`httpx.MockTransport`. No live vMLX calls
in unit tests — a manual smoke against the live server is a separate
one-shot action, not a CI dependency.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from clawfeed_intel.llm import CallResult, LLMClient, RoutingConfig
from clawfeed_intel.llm.client import _parse_response


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


# ── Pure response parser (fixture-testable, no HTTP layer) ────────────────


def test_parse_response_happy_path() -> None:
    payload = _ok_response(content="ok", model="m1", prompt_tokens=3, completion_tokens=4)
    result = _parse_response(payload, fallback_model="fallback", latency_ms=42)
    assert result == CallResult(
        content="ok",
        model="m1",
        latency_ms=42,
        prompt_tokens=3,
        completion_tokens=4,
    )


def test_parse_response_uses_fallback_model_when_response_omits_it() -> None:
    payload = _ok_response(content="ok", model="m1")
    del payload["model"]
    result = _parse_response(payload, fallback_model="stage-default", latency_ms=0)
    assert result.model == "stage-default"


def test_parse_response_missing_usage_degrades_to_zero() -> None:
    payload = _ok_response(content="ok", model="m")
    del payload["usage"]
    result = _parse_response(payload, fallback_model="m", latency_ms=0)
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


def test_parse_response_no_choices_raises() -> None:
    with pytest.raises(ValueError, match="no choices"):
        _parse_response({"choices": []}, fallback_model="m", latency_ms=0)


def test_parse_response_missing_choices_key_raises() -> None:
    with pytest.raises(ValueError, match="no choices"):
        _parse_response({}, fallback_model="m", latency_ms=0)


def test_parse_response_missing_content_raises() -> None:
    payload = {"choices": [{"message": {"role": "assistant"}}]}
    with pytest.raises(ValueError, match="missing message.content"):
        _parse_response(payload, fallback_model="m", latency_ms=0)


def test_parse_response_empty_string_content_kept() -> None:
    """An empty string is a valid (if useless) completion — don't conflate with None."""
    payload = _ok_response(content="", model="m")
    result = _parse_response(payload, fallback_model="m", latency_ms=0)
    assert result.content == ""


# ── chat_completion: happy path ───────────────────────────────────────────


async def test_chat_completion_returns_call_result(routing: RoutingConfig) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_ok_response(content="PONG", model="mlx-community/Qwen3.5-27B-4bit"),
        )

    client = LLMClient(routing, transport=httpx.MockTransport(handler))
    result = await client.chat_completion(
        "relevance_filter",
        [{"role": "user", "content": "ping"}],
    )

    assert isinstance(result, CallResult)
    assert result.content == "PONG"
    assert result.model == "mlx-community/Qwen3.5-27B-4bit"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 5
    assert result.latency_ms >= 0


async def test_chat_completion_targets_correct_url(routing: RoutingConfig) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(200, json=_ok_response(content="ok", model="m"))

    client = LLMClient(routing, transport=httpx.MockTransport(handler))
    await client.chat_completion("source_planning", [{"role": "user", "content": "hi"}])

    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"


async def test_chat_completion_body_uses_stage_model(routing: RoutingConfig) -> None:
    """The body's ``model`` comes from the stage config, not the caller's prompt."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json=_ok_response(content="ok", model="echo"))

    client = LLMClient(routing, transport=httpx.MockTransport(handler))
    await client.chat_completion("source_planning", [{"role": "user", "content": "hi"}])

    assert captured["body"]["model"] == "mlx-community/Qwen3-8B-4bit"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]


async def test_chat_completion_routes_per_stage(routing: RoutingConfig) -> None:
    """Different stages → different models in the request body."""
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        seen_models.append(body["model"])
        return httpx.Response(200, json=_ok_response(content="ok", model=body["model"]))

    client = LLMClient(routing, transport=httpx.MockTransport(handler))
    await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])
    await client.chat_completion("relevance_filter", [{"role": "user", "content": "x"}])

    assert seen_models == [
        "mlx-community/Qwen3-8B-4bit",
        "mlx-community/Qwen3.5-27B-4bit",
    ]


async def test_chat_completion_trailing_slash_in_base_url_normalized() -> None:
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

    client = LLMClient(routing, transport=httpx.MockTransport(handler))
    await client.chat_completion("source_planning", [{"role": "user", "content": "x"}])

    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"


# ── Failure modes ─────────────────────────────────────────────────────────


async def test_chat_completion_unknown_stage_raises(routing: RoutingConfig) -> None:
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json=_ok_response(content="x", model="x"))
    )
    client = LLMClient(routing, transport=transport)
    with pytest.raises(KeyError):
        await client.chat_completion("not_a_stage", [{"role": "user", "content": "x"}])


async def test_chat_completion_5xx_propagates(routing: RoutingConfig) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    client = LLMClient(routing, transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_completion(
            "source_planning",
            [{"role": "user", "content": "x"}],
        )


async def test_chat_completion_4xx_propagates(routing: RoutingConfig) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(400, json={"error": "bad request"}))
    client = LLMClient(routing, transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_completion(
            "source_planning",
            [{"role": "user", "content": "x"}],
        )


async def test_chat_completion_no_choices_raises(routing: RoutingConfig) -> None:
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"choices": [], "usage": {}})
    )
    client = LLMClient(routing, transport=transport)
    with pytest.raises(ValueError, match="no choices"):
        await client.chat_completion(
            "source_planning",
            [{"role": "user", "content": "x"}],
        )


async def test_chat_completion_missing_content_raises(routing: RoutingConfig) -> None:
    bad = {"choices": [{"message": {"role": "assistant"}}], "usage": {}}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=bad))
    client = LLMClient(routing, transport=transport)
    with pytest.raises(ValueError, match="missing message.content"):
        await client.chat_completion(
            "source_planning",
            [{"role": "user", "content": "x"}],
        )


async def test_chat_completion_handles_missing_usage(routing: RoutingConfig) -> None:
    """vMLX always emits usage, but we shouldn't crash if a future provider doesn't."""
    response = _ok_response(content="ok", model="m")
    del response["usage"]
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=response))
    client = LLMClient(routing, transport=transport)
    result = await client.chat_completion(
        "source_planning",
        [{"role": "user", "content": "x"}],
    )
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
