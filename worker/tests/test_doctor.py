"""Tests for the active vMLX probes used by ``clawfeed-intel doctor``."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from clawfeed_intel.doctor import (
    PROBE_CHAT_STAGE,
    PROBE_PROMPT,
    ProbeResult,
    _server_root,
    probe_chat,
    probe_health,
    probe_models,
    run_doctor_probes,
)
from clawfeed_intel.llm import RoutingConfig


@pytest.fixture
def routing() -> RoutingConfig:
    return RoutingConfig.model_validate(
        {
            "providers": {"vmlx": {"base_url": "http://127.0.0.1:8080/v1"}},
            "stages": {
                "relevance_filter": {
                    "provider": "vmlx",
                    "model": "mlx-community/Qwen3.5-27B-4bit",
                    "timeout_seconds": 60,
                },
            },
        }
    )


def _ok_chat(content: str = "PONG") -> dict[str, Any]:
    return {
        "id": "chatcmpl-probe",
        "object": "chat.completion",
        "model": "mlx-community/Qwen3.5-27B-4bit",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 1,
            "total_tokens": 9,
        },
    }


# ── Pure: _server_root ────────────────────────────────────────────────────


def test_server_root_strips_v1_suffix() -> None:
    assert _server_root("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"


def test_server_root_strips_v1_with_trailing_slash() -> None:
    assert _server_root("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080"


def test_server_root_keeps_root_without_v1() -> None:
    assert _server_root("http://127.0.0.1:8080") == "http://127.0.0.1:8080"


def test_server_root_keeps_root_with_trailing_slash() -> None:
    assert _server_root("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"


# ── probe_health ──────────────────────────────────────────────────────────


async def test_probe_health_ok(routing: RoutingConfig) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    result = await probe_health(routing.providers.vmlx, transport=transport)

    assert result.ok
    assert result.name == "health"
    assert "status=ok" in result.detail
    assert captured["url"] == "http://127.0.0.1:8080/health"
    assert result.latency_ms >= 0


async def test_probe_health_unexpected_status_value_fails(routing: RoutingConfig) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"status": "degraded"}))
    result = await probe_health(routing.providers.vmlx, transport=transport)
    assert not result.ok
    assert "degraded" in result.detail


async def test_probe_health_unexpected_payload_shape_fails(routing: RoutingConfig) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=["ok"]))
    result = await probe_health(routing.providers.vmlx, transport=transport)
    assert not result.ok


async def test_probe_health_5xx_fails(routing: RoutingConfig) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    result = await probe_health(routing.providers.vmlx, transport=transport)
    assert not result.ok
    assert "HTTPStatusError" in result.detail


async def test_probe_health_connect_error_fails(routing: RoutingConfig) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("vmlx down")

    transport = httpx.MockTransport(handler)
    result = await probe_health(routing.providers.vmlx, transport=transport)
    assert not result.ok
    assert "ConnectError" in result.detail


# ── probe_models ──────────────────────────────────────────────────────────


async def test_probe_models_ok(routing: RoutingConfig) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "mlx-community/Qwen3-8B-4bit"},
                    {"id": "mlx-community/Qwen3.5-27B-4bit"},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    result = await probe_models(routing.providers.vmlx, transport=transport)

    assert result.ok
    assert "2 model(s)" in result.detail
    assert "Qwen3-8B-4bit" in result.detail
    assert captured["url"] == "http://127.0.0.1:8080/v1/models"


async def test_probe_models_truncates_long_lists(routing: RoutingConfig) -> None:
    """Don't dump a 30-model registry verbatim into one terminal line."""
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={"data": [{"id": f"model-{i}"} for i in range(10)]},
        )
    )
    result = await probe_models(routing.providers.vmlx, transport=transport)
    assert result.ok
    assert "10 model(s)" in result.detail
    assert "+5 more" in result.detail


async def test_probe_models_empty_list_succeeds(routing: RoutingConfig) -> None:
    """Empty registry is a degenerate but non-error response."""
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []}))
    result = await probe_models(routing.providers.vmlx, transport=transport)
    assert result.ok
    assert "0" in result.detail


async def test_probe_models_unexpected_shape_fails(routing: RoutingConfig) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"models": []}))
    result = await probe_models(routing.providers.vmlx, transport=transport)
    assert not result.ok
    assert "unexpected response shape" in result.detail


async def test_probe_models_non_dict_items_skipped(routing: RoutingConfig) -> None:
    """Be defensive about API-shape drift."""
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"data": [{"id": "good"}, "junk", {}, {"id": ""}]})
    )
    result = await probe_models(routing.providers.vmlx, transport=transport)
    assert result.ok
    assert "1 model(s)" in result.detail
    assert "good" in result.detail


async def test_probe_models_5xx_fails(routing: RoutingConfig) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    result = await probe_models(routing.providers.vmlx, transport=transport)
    assert not result.ok


# ── probe_chat ────────────────────────────────────────────────────────────


async def test_probe_chat_ok(routing: RoutingConfig) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_ok_chat("PONG"))

    transport = httpx.MockTransport(handler)
    result = await probe_chat(routing, "relevance_filter", transport=transport)

    assert result.ok
    assert result.name == "chat:relevance_filter"
    assert "PONG" in result.detail
    assert "Qwen3.5-27B-4bit" in result.detail
    assert "prompt=" in result.detail
    assert "compl=" in result.detail
    assert len(captured) == 1
    assert str(captured[0].url) == "http://127.0.0.1:8080/v1/chat/completions"


async def test_probe_chat_includes_pong_prompt(routing: RoutingConfig) -> None:
    """The probe should ask for PONG specifically — that's the contract
    documented in the architecture doc / doctor-command spec."""
    import json as _json

    captured_body: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(_json.loads(request.read().decode()))
        return httpx.Response(200, json=_ok_chat("PONG"))

    transport = httpx.MockTransport(handler)
    await probe_chat(routing, "relevance_filter", transport=transport)

    assert captured_body["messages"] == [{"role": "user", "content": PROBE_PROMPT}]


async def test_probe_chat_long_response_truncated(routing: RoutingConfig) -> None:
    """If the model rambles, the doctor line stays readable."""
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json=_ok_chat("PONG " + "x" * 200))
    )
    result = await probe_chat(routing, "relevance_filter", transport=transport)
    assert result.ok
    # The truncated snippet is enclosed in quotes and ends with `...`
    assert "..." in result.detail


async def test_probe_chat_5xx_fails_without_retry(routing: RoutingConfig) -> None:
    """Doctor disables retry — one HTTP attempt, period."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    result = await probe_chat(routing, "relevance_filter", transport=transport)
    assert not result.ok
    assert "HTTPStatusError" in result.detail
    assert len(captured) == 1, "doctor probe must not retry — operator wants raw connectivity"


async def test_probe_chat_unknown_stage_fails(routing: RoutingConfig) -> None:
    """A misconfigured stage should surface as a probe failure, not crash the doctor."""
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=_ok_chat("PONG")))
    result = await probe_chat(routing, "made_up_stage", transport=transport)
    assert not result.ok
    assert "KeyError" in result.detail


# ── run_doctor_probes ─────────────────────────────────────────────────────


def _all_ok_handler() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        if path.endswith("/v1/models"):
            return httpx.Response(200, json={"data": [{"id": "m1"}]})
        if path.endswith("/v1/chat/completions"):
            return httpx.Response(200, json=_ok_chat())
        return httpx.Response(404, text=f"unexpected: {path}")

    return httpx.MockTransport(handler)


async def test_run_doctor_probes_returns_three_results(routing: RoutingConfig) -> None:
    results = await run_doctor_probes(routing, transport=_all_ok_handler())
    assert len(results) == 3
    assert [r.name for r in results] == ["health", "models", f"chat:{PROBE_CHAT_STAGE}"]
    assert all(r.ok for r in results)


async def test_run_doctor_probes_individual_failure_does_not_short_circuit(
    routing: RoutingConfig,
) -> None:
    """Health fails — models and chat still run."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(503)
        if request.url.path.endswith("/v1/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        if request.url.path.endswith("/v1/chat/completions"):
            return httpx.Response(200, json=_ok_chat())
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    results = await run_doctor_probes(routing, transport=transport)

    assert len(results) == 3
    assert [r.ok for r in results] == [False, True, True]


async def test_run_doctor_probes_all_failures_surface_independently(
    routing: RoutingConfig,
) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    results = await run_doctor_probes(routing, transport=transport)
    assert len(results) == 3
    assert all(not r.ok for r in results)


# ── ProbeResult shape ─────────────────────────────────────────────────────


def test_probe_result_is_immutable() -> None:
    """Results are passed around the CLI; mutation would risk audit drift."""
    r = ProbeResult(name="x", ok=True, latency_ms=10, detail="hi")
    with pytest.raises(Exception):
        r.ok = False  # type: ignore[misc]
