"""Active health probes for the intelligence pipeline.

`clawfeed-intel doctor` runs three independent probes against the vMLX
provider declared in :mod:`config/model-routing.yaml`:

1. ``GET /health``  — server is up and reports ``{"status": "ok"}``.
2. ``GET /v1/models`` — model registry is reachable; cached models listed.
3. ``POST /v1/chat/completions`` — tiny "Reply with PONG" prompt against
   the configured ``relevance_filter`` model. This is the canonical
   "can-we-actually-do-inference" check; the architecture doc named the
   filter stage as the load-bearing daily-brief dependency.

Probes are independent: an individual failure doesn't short-circuit the
others. The CLI surfaces every result so an operator sees the full
picture in one run, and exits non-zero if any probe failed.

Tests use :class:`httpx.MockTransport` for the HTTP layer; the live-vMLX
smoke is a manual one-shot, not a CI dependency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from .llm import LLMClient, RetryConfig, RoutingConfig
from .llm.routing import VmlxProviderConfig

PROBE_PROMPT = "Reply with the single word PONG."

# vMLX cold-loads cached models on first request; the architecture-doc
# 27B placeholder takes ~3-5s wall to warm. Give the chat probe a real
# budget rather than the 5s default so an operator doesn't see false
# negatives just because the model wasn't already in memory.
PROBE_HEALTH_TIMEOUT_SECONDS = 5.0
PROBE_MODELS_TIMEOUT_SECONDS = 5.0
PROBE_CHAT_STAGE = "relevance_filter"


@dataclass(frozen=True)
class ProbeResult:
    """One probe outcome — surfaced as a single CLI line."""

    name: str
    ok: bool
    latency_ms: int
    detail: str


def _server_root(base_url: str) -> str:
    """Strip a trailing ``/v1`` suffix from a base URL.

    vMLX exposes ``/health`` at the server root and ``/v1/*`` for the
    OpenAI-compatible endpoints, so the routing config's ``base_url``
    (e.g. ``http://127.0.0.1:8080/v1``) needs the suffix dropped to
    address the health endpoint.
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return base


async def probe_health(
    provider: VmlxProviderConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProbeResult:
    """``GET /health`` — server up + reports ``ok``."""
    health_url = _server_root(provider.base_url) + "/health"
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            transport=transport,
            timeout=PROBE_HEALTH_TIMEOUT_SECONDS,
        ) as client:
            response = await client.get(health_url)
            response.raise_for_status()
            payload = response.json()
        latency = int((time.perf_counter() - start) * 1000)
    except Exception as exc:
        latency = int((time.perf_counter() - start) * 1000)
        return ProbeResult(
            name="health",
            ok=False,
            latency_ms=latency,
            detail=f"{type(exc).__name__}: {exc}",
        )

    if isinstance(payload, dict) and payload.get("status") == "ok":
        return ProbeResult(
            name="health",
            ok=True,
            latency_ms=latency,
            detail=f"status=ok ({health_url})",
        )
    return ProbeResult(
        name="health",
        ok=False,
        latency_ms=latency,
        detail=f"unexpected response shape: {payload!r}",
    )


async def probe_models(
    provider: VmlxProviderConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProbeResult:
    """``GET /v1/models`` — list available models on the server.

    vMLX hot-loads cached models on demand, so the response may include
    only the currently-loaded model rather than the full on-disk
    inventory. We don't gate on a particular model being present here;
    the chat probe is the canonical "can we actually use this model"
    test.
    """
    models_url = provider.base_url.rstrip("/") + "/models"
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            transport=transport,
            timeout=PROBE_MODELS_TIMEOUT_SECONDS,
        ) as client:
            response = await client.get(models_url)
            response.raise_for_status()
            payload = response.json()
        latency = int((time.perf_counter() - start) * 1000)
    except Exception as exc:
        latency = int((time.perf_counter() - start) * 1000)
        return ProbeResult(
            name="models",
            ok=False,
            latency_ms=latency,
            detail=f"{type(exc).__name__}: {exc}",
        )

    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return ProbeResult(
            name="models",
            ok=False,
            latency_ms=latency,
            detail=f"unexpected response shape: {payload!r}",
        )

    model_ids = [m.get("id") for m in items if isinstance(m, dict) and m.get("id")]
    summary = ", ".join(model_ids[:5])
    if len(model_ids) > 5:
        summary += f", … (+{len(model_ids) - 5} more)"
    return ProbeResult(
        name="models",
        ok=True,
        latency_ms=latency,
        detail=f"{len(model_ids)} model(s): {summary}" if model_ids else "0 models",
    )


async def probe_chat(
    routing: RoutingConfig,
    stage: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProbeResult:
    """Tiny chat-completion against ``stage``'s configured model.

    Uses :class:`LLMClient` directly so the probe goes through the same
    chokepoint as production calls. Retry is disabled
    (``max_attempts=1``) — doctor wants the raw connectivity answer, not
    a softened one. A persistent transient error here is still a
    "vMLX is in a bad state" signal worth surfacing.
    """
    client = LLMClient(
        routing,
        transport=transport,
        retry_config=RetryConfig(
            max_attempts=1,
            wait_multiplier=0,
            wait_min_seconds=0,
            wait_max_seconds=0,
        ),
    )
    start = time.perf_counter()
    try:
        result = await client.chat_completion(
            stage,
            [{"role": "user", "content": PROBE_PROMPT}],
        )
        latency = int((time.perf_counter() - start) * 1000)
    except Exception as exc:
        latency = int((time.perf_counter() - start) * 1000)
        return ProbeResult(
            name=f"chat:{stage}",
            ok=False,
            latency_ms=latency,
            detail=f"{type(exc).__name__}: {exc}",
        )

    snippet = result.content.strip().replace("\n", " ")
    if len(snippet) > 60:
        snippet = snippet[:57] + "..."
    detail = (
        f"{result.model} → {snippet!r} "
        f"(prompt={result.prompt_tokens} compl={result.completion_tokens})"
    )
    return ProbeResult(
        name=f"chat:{stage}",
        ok=True,
        latency_ms=latency,
        detail=detail,
    )


async def run_doctor_probes(
    routing: RoutingConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[ProbeResult]:
    """Run health → models → chat probes in sequence.

    Returns results in execution order. Individual failures don't
    short-circuit — every probe runs every time so an operator sees the
    full picture (e.g. "health works but the filter model is broken").
    """
    provider = routing.providers.vmlx
    return [
        await probe_health(provider, transport=transport),
        await probe_models(provider, transport=transport),
        await probe_chat(routing, PROBE_CHAT_STAGE, transport=transport),
    ]
