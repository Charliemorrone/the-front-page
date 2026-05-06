"""LLM client chokepoint.

One async entrypoint dispatches every chat-completion in the pipeline.
The stage name resolves to a provider + model + timeout via
:class:`RoutingConfig`; the body shape is OpenAI-compatible so vMLX (and
later OpenClaw, step 11) accept the same request format.

Phase 1 step 8a is happy-path only. Retries (tenacity), per-call timeout
classification, JSON-schema validation, the bounded JSON-repair fallback,
and ``llm_calls`` logging all land in step 8b.

Two-layer split mirrored from the fetcher modules:

- :func:`_parse_response` is pure. Given an OpenAI-compatible payload, it
  returns a :class:`CallResult` or raises ``ValueError``. Fixture-testable
  without an HTTP layer.
- :class:`LLMClient.chat_completion` is the async wrapper that adds the
  HTTP call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from .routing import RoutingConfig, StageConfig


@dataclass(frozen=True)
class CallResult:
    """One model response, normalized.

    ``latency_ms`` is wall-clock around the HTTP call only — it excludes
    JSON parse and excludes any retry attempts (those are added in 8b).
    Token counts come from the ``usage`` block; missing usage degrades to
    zero rather than raising, since not every future provider is
    guaranteed to populate it.
    """

    content: str
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int


class LLMClient:
    """Async chat-completion dispatcher routed through :class:`RoutingConfig`.

    The optional ``transport`` argument lets tests inject
    :class:`httpx.MockTransport` without touching the network. Production
    callers omit it and httpx uses its default transport.
    """

    def __init__(
        self,
        routing: RoutingConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._routing = routing
        self._transport = transport

    async def chat_completion(
        self,
        stage: str,
        messages: list[dict[str, str]],
    ) -> CallResult:
        """Dispatch one chat-completion against the configured stage.

        Raises:
            KeyError: stage not in the routing config.
            httpx.HTTPStatusError: provider returned 4xx/5xx.
            httpx.HTTPError: transport-level failure.
            ValueError: response payload missing required fields.
        """
        stage_config = self._routing.resolve(stage)
        url = self._build_url(stage_config)
        body = {"model": stage_config.model, "messages": messages}

        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=stage_config.timeout_seconds,
        ) as client:
            start = time.perf_counter()
            response = await client.post(url, json=body)
            latency_ms = int((time.perf_counter() - start) * 1000)
            response.raise_for_status()
            payload = response.json()

        return _parse_response(
            payload,
            fallback_model=stage_config.model,
            latency_ms=latency_ms,
        )

    def _build_url(self, stage_config: StageConfig) -> str:
        # Step 11 will branch on stage_config.provider here to pick between
        # vmlx (HTTP) and openclaw (WebSocket gateway).
        del stage_config
        provider = self._routing.providers.vmlx
        return f"{provider.base_url.rstrip('/')}/chat/completions"


def _parse_response(
    payload: dict[str, Any],
    *,
    fallback_model: str,
    latency_ms: int,
) -> CallResult:
    """Extract content + usage from an OpenAI-compatible response.

    ``fallback_model`` is the stage's configured model name, used when the
    server omits ``model`` in the response (vMLX always echoes it back, but
    other providers may not).
    """
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise ValueError("response choice missing message.content")
    usage = payload.get("usage") or {}
    return CallResult(
        content=content,
        model=payload.get("model") or fallback_model,
        latency_ms=latency_ms,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )
