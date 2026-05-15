"""LLM client chokepoint.

One async entrypoint dispatches every chat-completion in the pipeline.
The stage name resolves to a provider + model + timeout via
:class:`RoutingConfig`; the body shape is OpenAI-compatible so vMLX (and
later OpenClaw, step 11) accept the same request format.

Phase 1 step 8b adds the reliability layer on top of step 8a's happy
path:

- **Tenacity retry** on transient errors (5xx, 429, connect/read/write/
  pool timeouts, connect errors). Up to ``max_attempts`` attempts with
  exponential backoff. Non-transient 4xx errors fail immediately.
- **JSON-schema validation** when ``response_schema`` is supplied.
  Content is parsed as JSON and validated against the supplied pydantic
  model. On failure, one bounded repair attempt is sent before raising
  :class:`LLMSchemaError`.
- **DB logging** of every logical call (success or failure) to
  ``llm_calls`` via :func:`db.record_llm_call`. Hashes only — no prompt
  or response text stored.

Two-layer split mirrors the fetcher modules:

- :func:`_parse_response`, :func:`_validate_schema`, :func:`_hash_messages`,
  :func:`_hash_content`, :func:`_is_transient_error` are pure. Fixture-
  testable without HTTP or a SQLite connection.
- :class:`LLMClient.chat_completion` is the async wrapper that adds the
  HTTP call, retries, repair flow, and DB logging.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .. import db
from .gemini_cli import (
    GeminiCliProviderConfig as GeminiCliRuntimeConfig,
    GeminiCliResult,
    gemini_cli_completion,
)
from .routing import GeminiCliProviderConfig, RoutingConfig, StageConfig

log = logging.getLogger(__name__)


_REPAIR_PROMPT = (
    "Your previous response could not be parsed as JSON matching the required "
    "schema. Reply with valid JSON only — no markdown fencing, no commentary, "
    "no preamble."
)


# ── Public errors ─────────────────────────────────────────────────────────────


class LLMSchemaError(ValueError):
    """Response failed JSON parse or pydantic schema validation.

    Raised after the bounded repair attempt has also failed. The original
    parse/validation exception is chained via ``__cause__``.
    """


# ── Public config ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for transient HTTP failures.

    ``wait_min_seconds`` / ``wait_max_seconds`` clamp the exponential
    backoff. Tests that assert retry counts should set both to ``0`` to
    keep the suite fast — production callers use the defaults.
    """

    max_attempts: int = 3
    wait_multiplier: float = 1.0
    wait_min_seconds: float = 1.0
    wait_max_seconds: float = 10.0


_DEFAULT_RETRY = RetryConfig()


# ── Public result ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CallResult:
    """One model response, normalized.

    ``latency_ms`` is total wall-clock for the logical call — including
    every retry attempt and the repair attempt if one fired. Token counts
    are summed across the same set. ``parsed`` is populated only when
    ``chat_completion`` was called with a ``response_schema``.
    """

    content: str
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    parsed: BaseModel | None = None


# ── Internal accumulator ──────────────────────────────────────────────────────


@dataclass
class _Metrics:
    """Mutable token counters, summed across HTTP attempts within one call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add_usage(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)


# ── Client ────────────────────────────────────────────────────────────────────


class LLMClient:
    """Async chat-completion dispatcher routed through :class:`RoutingConfig`.

    Optional constructor args:

    - ``transport``: lets tests inject :class:`httpx.MockTransport`.
      Production omits it.
    - ``conn`` + ``run_id``: when both supplied, every call writes one row
      to ``llm_calls``. Either-or-neither (passing ``conn`` without
      ``run_id`` is permitted and just records ``run_id=NULL``; passing
      ``run_id`` without ``conn`` skips DB logging entirely).
    - ``retry_config``: per-stage retry policy. Defaults to
      :data:`_DEFAULT_RETRY` (3 attempts, 1–10s exponential backoff).
    """

    def __init__(
        self,
        routing: RoutingConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        conn: sqlite3.Connection | None = None,
        run_id: int | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._routing = routing
        self._transport = transport
        self._conn = conn
        self._run_id = run_id
        self._retry_config = retry_config or _DEFAULT_RETRY

    @property
    def routing(self) -> RoutingConfig:
        """Public read-only access to the routing config.

        Callers like :func:`~clawfeed_intel.pipeline.compose.compose_brief`
        need to inspect ``StageConfig.fallback`` to drive the multi-tier
        resilience chain. Exposing the routing object publicly is cleaner
        than reaching at ``_routing`` from outside the package.
        """
        return self._routing

    async def chat_completion(
        self,
        stage: str,
        messages: list[dict[str, str]],
        *,
        response_schema: type[BaseModel] | None = None,
        prompt_version: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage_config_override: StageConfig | None = None,
    ) -> CallResult:
        """Dispatch one chat-completion against the configured stage.

        ``temperature`` is forwarded into the OpenAI body when supplied;
        when omitted the provider's default sampling temperature is used
        (vMLX defaults to ~0.7 for Qwen-style models). Structured-output
        stages typically want ``0.0``–``0.1`` to keep JSON well-formed
        under load. The ``gemini_cli`` provider ignores ``temperature``
        and ``max_tokens`` — sampling is configured server-side by
        Gemini and the CLI doesn't surface either knob.

        ``max_tokens`` is forwarded only on the HTTP path; see above.

        ``stage_config_override`` lets a caller (currently the compose
        stage's Tier-2 fallback path) dispatch against a synthesized
        stage config without registering it in the YAML. The audit row
        still records the caller-supplied ``stage`` string for grouping;
        ``provider`` and ``model`` columns reflect the override so the
        log distinguishes the fallback call from the primary.

        Raises:
            KeyError: stage not in the routing config.
            httpx.HTTPStatusError: HTTP provider returned a non-transient
                4xx, or 5xx/429 after exhausting retries.
            httpx.HTTPError: HTTP transport-level failure after exhausting
                retries.
            GeminiCliError: CLI provider failed after its internal retry.
            LLMSchemaError: HTTP response (and the repair retry) failed
                JSON parse or schema validation. CLI provider does not
                support schema validation.
            ValueError: response payload missing required fields, or
                ``response_schema`` supplied with a CLI provider stage.
        """
        stage_config = stage_config_override or self._routing.resolve(stage)
        metrics = _Metrics()
        input_hash = _hash_messages(messages)
        wall_start = time.perf_counter()

        # Branch on provider type. The CLI path has its own retry +
        # timeout machinery inside :func:`gemini_cli_completion`, so we
        # don't wrap it in tenacity here; doing so would double-count
        # attempts and mask provider-level diagnostics.
        if stage_config.provider == "gemini_cli":
            if response_schema is not None:
                raise ValueError(
                    "gemini_cli provider does not support response_schema; "
                    "the CLI emits free-form text, not structured JSON"
                )
            try:
                result = await self._cli_call(stage_config, messages, metrics)
            except BaseException as exc:
                wall_ms = int((time.perf_counter() - wall_start) * 1000)
                self._record(
                    stage=stage,
                    stage_config=stage_config,
                    input_hash=input_hash,
                    output_hash=None,
                    latency_ms=wall_ms,
                    metrics=metrics,
                    status="failed",
                    error=_short_error(exc),
                    prompt_version=prompt_version,
                )
                raise
            wall_ms = int((time.perf_counter() - wall_start) * 1000)
            self._record(
                stage=stage,
                stage_config=stage_config,
                input_hash=input_hash,
                output_hash=_hash_content(result.content),
                latency_ms=wall_ms,
                metrics=metrics,
                status="succeeded",
                error=None,
                prompt_version=prompt_version,
            )
            # Replace the provider's wall-clock with our own (which
            # includes the audit overhead) so latency_ms stays
            # consistent with the HTTP path.
            return CallResult(
                content=result.content,
                model=result.model,
                latency_ms=wall_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                parsed=None,
            )

        try:
            content, model = await self._call_with_retries(
                stage_config, messages, metrics, temperature, max_tokens
            )
            parsed, content_out = await self._validate_and_repair(
                stage_config,
                messages,
                content,
                response_schema,
                metrics,
                temperature,
                max_tokens,
            )
        except BaseException as exc:
            wall_ms = int((time.perf_counter() - wall_start) * 1000)
            self._record(
                stage=stage,
                stage_config=stage_config,
                input_hash=input_hash,
                output_hash=None,
                latency_ms=wall_ms,
                metrics=metrics,
                status="failed",
                error=_short_error(exc),
                prompt_version=prompt_version,
            )
            raise

        wall_ms = int((time.perf_counter() - wall_start) * 1000)
        self._record(
            stage=stage,
            stage_config=stage_config,
            input_hash=input_hash,
            output_hash=_hash_content(content_out),
            latency_ms=wall_ms,
            metrics=metrics,
            status="succeeded",
            error=None,
            prompt_version=prompt_version,
        )

        return CallResult(
            content=content_out,
            model=model,
            latency_ms=wall_ms,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            parsed=parsed,
        )

    async def _call_with_retries(
        self,
        stage_config: StageConfig,
        messages: list[dict[str, str]],
        metrics: _Metrics,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, str]:
        """Make one HTTP call, retrying transient failures.

        Returns ``(content, model)`` from the successful response. Token
        usage is folded into ``metrics`` as a side effect — only on
        success, since failed attempts don't carry usage.
        """
        cfg = self._retry_config
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_transient_error),
            stop=stop_after_attempt(cfg.max_attempts),
            wait=wait_exponential(
                multiplier=cfg.wait_multiplier,
                min=cfg.wait_min_seconds,
                max=cfg.wait_max_seconds,
            ),
            reraise=True,
        )

        payload: dict[str, Any] = {}
        async for attempt in retrying:
            with attempt:
                payload = await self._http_call_once(
                    stage_config, messages, temperature, max_tokens
                )

        metrics.add_usage(payload)
        result = _parse_response(payload, fallback_model=stage_config.model)
        return result.content, result.model

    async def _validate_and_repair(
        self,
        stage_config: StageConfig,
        messages: list[dict[str, str]],
        content: str,
        response_schema: type[BaseModel] | None,
        metrics: _Metrics,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[BaseModel | None, str]:
        """Validate ``content`` against the optional schema; repair once.

        Returns ``(parsed, validated_content)``. ``parsed`` is ``None``
        when no schema was requested; ``validated_content`` is the
        original response in that case. When a repair fires successfully,
        ``validated_content`` is the repaired content (so the output hash
        and the surfaced ``CallResult.content`` reflect what was actually
        accepted, not the original malformed response).

        Raises :class:`LLMSchemaError` when validation fails on both the
        original response and the repair attempt.
        """
        if response_schema is None:
            return None, content

        try:
            return _validate_schema(content, response_schema), content
        except LLMSchemaError as first_failure:
            log.info("%s: schema validation failed, attempting one repair", stage_config.model)
            repair_messages = [
                *messages,
                {"role": "assistant", "content": content},
                {"role": "user", "content": _REPAIR_PROMPT},
            ]
            repaired_content, _ = await self._call_with_retries(
                stage_config, repair_messages, metrics, temperature, max_tokens
            )
            try:
                return (
                    _validate_schema(repaired_content, response_schema),
                    repaired_content,
                )
            except LLMSchemaError as second_failure:
                # Surface the second failure as the cause so callers see
                # the final state of the response, not the original problem.
                raise LLMSchemaError(
                    f"schema validation failed after repair: {second_failure}"
                ) from first_failure

    async def _http_call_once(
        self,
        stage_config: StageConfig,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """One HTTP attempt. Raises on 4xx/5xx; retry handled by caller."""
        url = self._build_url(stage_config)
        body: dict[str, Any] = {"model": stage_config.model, "messages": messages}
        if temperature is not None:
            # Forward only when caller pinned a value; providers' defaults
            # vary, and we don't want this client to impose a global default.
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=stage_config.timeout_seconds,
        ) as client:
            response = await client.post(url, json=body)
            response.raise_for_status()
            return response.json()

    def _build_url(self, stage_config: StageConfig) -> str:
        # Step 12b: HTTP path is vmlx only. The gemini_cli provider is
        # dispatched through ``_cli_call`` (subprocess, not HTTP) and
        # never reaches this method.
        del stage_config
        provider = self._routing.providers.vmlx
        return f"{provider.base_url.rstrip('/')}/chat/completions"

    async def _cli_call(
        self,
        stage_config: StageConfig,
        messages: list[dict[str, str]],
        metrics: _Metrics,
    ) -> GeminiCliResult:
        """Dispatch one ``gemini_cli`` completion.

        The provider's config (``executable_path``, timeouts, etc.) is
        read off ``self._routing.providers.gemini_cli`` — the pydantic
        YAML shape. ``StageConfig.retries`` / ``retry_backoff_seconds``
        are read off the stage and override the provider defaults when
        present, so per-stage retry policy is YAML-tunable.

        Raises:
            ValueError: ``gemini_cli`` not declared in the providers
                block (deploy bug — a stage routed to it but the
                provider config is missing).
            GeminiCliError: subprocess failed after its retries.
        """
        provider_yaml = self._routing.providers.gemini_cli
        if provider_yaml is None:
            raise ValueError(
                "stage routes to gemini_cli provider but providers.gemini_cli "
                "is not declared in model-routing.yaml"
            )
        # Convert the pydantic YAML shape to the dataclass the async
        # function consumes. Stage-level overrides (retries,
        # retry_backoff_seconds) win when supplied.
        runtime = _build_gemini_runtime_config(provider_yaml, stage_config)
        result = await gemini_cli_completion(
            runtime,
            messages=messages,
            model=stage_config.model,
        )
        # Fold token usage into the same accumulator the HTTP path uses
        # so :func:`db.record_llm_call` sees a non-zero count when the
        # Gemini stream emits a usage event.
        metrics.prompt_tokens += result.prompt_tokens
        metrics.completion_tokens += result.completion_tokens
        return result

    def _record(
        self,
        *,
        stage: str,
        stage_config: StageConfig,
        input_hash: str,
        output_hash: str | None,
        latency_ms: int,
        metrics: _Metrics,
        status: str,
        error: str | None,
        prompt_version: str | None,
    ) -> None:
        """Persist one ``llm_calls`` row when a connection is configured.

        DB write is best-effort: if logging fails (e.g. SQLite locked by
        another writer beyond busy_timeout), we log the secondary failure
        and return rather than mask the original call result. The audit
        trail is valuable but must not crash a successful inference.
        """
        if self._conn is None:
            return
        try:
            db.record_llm_call(
                self._conn,
                stage=stage,
                provider=stage_config.provider,
                model=stage_config.model,
                status=status,
                latency_ms=latency_ms,
                run_id=self._run_id,
                prompt_version=prompt_version,
                input_hash=input_hash,
                output_hash=output_hash,
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=metrics.completion_tokens,
                error=error,
            )
        except Exception:
            log.exception(
                "%s: failed to record llm_calls row (status=%s); call result preserved",
                stage,
                status,
            )


# ── Pure helpers ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ParsedResponse:
    content: str
    model: str


def _parse_response(
    payload: dict[str, Any],
    *,
    fallback_model: str,
) -> _ParsedResponse:
    """Extract content + model from an OpenAI-compatible response.

    Token usage is read by the caller via :meth:`_Metrics.add_usage` so
    aggregation across retries lives in one place.
    """
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise ValueError("response choice missing message.content")
    return _ParsedResponse(
        content=content,
        model=payload.get("model") or fallback_model,
    )


def _validate_schema(content: str, schema: type[BaseModel]) -> BaseModel:
    """Parse JSON and validate against ``schema``.

    Wraps both ``json.JSONDecodeError`` and pydantic's
    ``ValidationError`` in :class:`LLMSchemaError` so callers handle one
    type. Pydantic's ``model_validate_json`` does both steps in one call
    but the error types differ, so we still need the wrapping.
    """
    try:
        return schema.model_validate_json(content)
    except (ValidationError, ValueError) as exc:
        # ``ValueError`` covers ``json.JSONDecodeError`` (it's a subclass)
        # and any other parse-time failures pydantic might surface.
        raise LLMSchemaError(str(exc)) from exc


def _is_transient_error(exc: BaseException) -> bool:
    """Predicate for tenacity: is this a retryable HTTP error?

    Retryable: 5xx, 429, connect/read/write/pool timeouts, connect errors.
    Not retryable: 4xx other than 429 (those reflect caller-side problems
    that won't change between attempts).
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= 500 or status == 429
    return isinstance(
        exc,
        (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.ConnectError,
        ),
    )


def _hash_messages(messages: list[dict[str, str]]) -> str:
    """SHA-256 over canonical message JSON.

    ``sort_keys=True`` makes the hash stable regardless of dict insertion
    order; ``separators=(",", ":")`` strips whitespace so two equivalent
    inputs produce the same digest.
    """
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_content(content: str) -> str:
    """SHA-256 over the response content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _build_gemini_runtime_config(
    provider_yaml: GeminiCliProviderConfig,
    stage_config: StageConfig,
) -> GeminiCliRuntimeConfig:
    """Convert the pydantic provider config into the dataclass runtime shape.

    Stage-level ``retries`` / ``retry_backoff_seconds`` override the
    provider defaults when present — lets a YAML deployment tune the
    compose stage's retry policy without changing other stages or
    providers. ``hard_timeout_seconds`` is taken from the provider
    config so it applies uniformly to every stage routed at gemini_cli;
    ``StageConfig.timeout_seconds`` mirrors it as documentation but is
    not separately consulted (a per-stage CLI timeout that differs
    from the provider-wide CLI timeout would be a confusing config).
    """
    retries = stage_config.retries if stage_config.retries is not None else provider_yaml.retries
    backoff = (
        stage_config.retry_backoff_seconds
        if stage_config.retry_backoff_seconds is not None
        else provider_yaml.retry_backoff_seconds
    )
    return GeminiCliRuntimeConfig(
        script_path=provider_yaml.script_path,
        executable_path=provider_yaml.executable_path,
        approval_mode=provider_yaml.approval_mode,
        output_format=provider_yaml.output_format,
        idle_timeout_seconds=provider_yaml.idle_timeout_seconds,
        hard_timeout_seconds=provider_yaml.hard_timeout_seconds,
        retries=retries,
        retry_backoff_seconds=backoff,
    )


def _short_error(exc: BaseException) -> str:
    """One-line description of an exception for the ``llm_calls.error`` column."""
    name = type(exc).__name__
    msg = str(exc)
    if not msg:
        return name
    # Trim to keep the audit log readable; full trace lives in stderr logs.
    if len(msg) > 280:
        msg = msg[:277] + "..."
    return f"{name}: {msg}"


# Internal helpers exported only for unit tests. Public surface is
# defined in ``llm/__init__.py``.
__all__ = (
    "CallResult",
    "LLMClient",
    "LLMSchemaError",
    "RetryConfig",
)
