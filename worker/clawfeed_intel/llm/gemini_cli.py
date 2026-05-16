"""Gemini CLI subprocess provider for final-compose calls.

Architecture-doc Phase 5 requires final composition by a frontier-class
model. The 2026-05-15 amendment routes that through the Gemini CLI
(``gemini-2.5-pro`` by default — see ``config/model-routing.yaml`` for
the Step 12c correction note) invoked as a subprocess rather than
through the OpenClaw WebSocket gateway. See
``docs/personal-intelligence-brief-architecture.md`` Decision 4
amendment for the full rationale.

**Auth and cost model (load-bearing).** The Gemini CLI is signed into
the operator's **Gemini Pro subscription** via OAuth at install time
(``gemini`` interactive command); the CLI manages the refresh-token
lifecycle internally. **This is not the pay-per-token Gemini API.**
The worker holds no API key, never makes a
``generativelanguage.googleapis.com`` HTTP call directly, and incurs
no per-token cost. Quota is the Pro plan's subscription ceiling
(comfortably covers one daily-brief compose + several topical-search
composes per day). A ``GeminiCliExitError`` mentioning quota /
auth / login is recovered by re-authenticating the CLI interactively
(`docs/runbook.md` "Gemini CLI auth expired"), NOT by swapping to a
direct API integration — see the architecture-doc note for the
explicit decision against API billing.

This module is Step 12a: the provider in isolation. Step 12b wires it
into :class:`~clawfeed_intel.llm.client.LLMClient` and
:func:`~clawfeed_intel.pipeline.compose.compose_brief`.

Why subprocess + ``-o stream-json`` rather than buffered output:
Gemini CLI occasionally stalls mid-response on long generations — the
underlying API streams tokens but the local subprocess can stop
emitting without erroring, producing an indefinite hang. Streaming
JSON output gives us per-line events; we time the gap between events
to detect a stall **far faster** than waiting out a wall-clock
timeout. When the gap exceeds ``idle_timeout_seconds`` (default 60s)
or total wall time exceeds ``hard_timeout_seconds`` (default 300s),
we send ``SIGTERM`` with a 5-second grace period, then ``SIGKILL``
if the child hasn't exited. The retry path picks up cleanly.

Why an explicit ``executable_path`` for the node binary: the local
``node@22`` on PATH is broken (linked against an absent
``libsimdjson.30.dylib``). The provider must invoke the working
``node`` v25.9.0 at ``/opt/homebrew/bin/node`` directly so the
shebang's PATH-based resolution doesn't pick the wrong binary.
Tracked as an environmental risk in the status doc; this is the
mitigation, not the fix.

Two-layer split mirrors the rest of the LLM package:

- **Pure layer**: :func:`_flatten_messages`,
  :func:`_extract_event_text`, :func:`_build_argv`. Fixture-testable
  without an actual subprocess.
- **Async layer**: :func:`gemini_cli_completion` runs the subprocess
  with the timeout + kill + retry machinery and returns
  :class:`GeminiCliResult`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


_GRACE_PERIOD_SECONDS = 5.0
"""SIGTERM grace period before escalating to SIGKILL.

Five seconds is enough for Gemini CLI to flush any in-flight stdout
and shut down cleanly; long enough to avoid spurious SIGKILLs on a
busy machine, short enough that a stuck child doesn't hold the run.
"""


# ── Public errors ─────────────────────────────────────────────────────────────


class GeminiCliError(RuntimeError):
    """Base class for Gemini CLI provider failures.

    Caller code (Step 12b's three-tier fallback chain) catches this
    base class and routes to Tier-2 vMLX fallback. The subclasses
    exist so logs and audit trails can distinguish between stall,
    hard timeout, non-zero exit, and malformed output without parsing
    error messages.
    """


class GeminiCliStallError(GeminiCliError):
    """Stream went silent for longer than ``idle_timeout_seconds``."""


class GeminiCliTimeoutError(GeminiCliError):
    """Total wall-clock exceeded ``hard_timeout_seconds``."""


class GeminiCliExitError(GeminiCliError):
    """Subprocess exited non-zero. ``stderr_tail`` captures the last
    chunk of stderr for the audit trail; the full stderr is logged.
    """

    def __init__(self, returncode: int, stderr_tail: str) -> None:
        super().__init__(f"gemini exited with code {returncode}: {stderr_tail}")
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class GeminiCliOutputError(GeminiCliError):
    """The stream produced no usable content.

    Raised when the subprocess exits 0 but yielded zero text events —
    indicates the response shape changed or the model refused. The
    fallback chain handles this the same way it handles a crash.
    """


# ── Public config ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GeminiCliProviderConfig:
    """Provider config for the Gemini CLI subprocess.

    ``executable_path`` and ``script_path`` are separated so we can
    invoke the CLI with a specific node binary (bypassing a broken
    PATH-resolved shebang). When ``executable_path`` is ``None``, we
    invoke ``script_path`` directly (relying on its shebang) — the
    "user fixed their PATH" case.

    The two timeouts handle distinct failure modes:

    - ``idle_timeout_seconds``: gap between consecutive stream events.
      A stall mid-stream trips this fast (default 60s) so the retry
      path doesn't have to wait out the full wall-clock cap.
    - ``hard_timeout_seconds``: total wall-clock cap (default 300s).
      Catches genuinely long but non-stalled generations.

    ``retries`` is the number of *additional* attempts after the first
    failure; ``retries=1`` means up to 2 total attempts. ``retries=0``
    disables retry entirely (used in tests that assert the first
    attempt's failure shape).
    """

    script_path: str
    executable_path: str | None = None
    approval_mode: str = "plan"
    output_format: str = "stream-json"
    idle_timeout_seconds: float = 60.0
    hard_timeout_seconds: float = 300.0
    retries: int = 1
    retry_backoff_seconds: float = 10.0


# ── Public result ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GeminiCliResult:
    """One successful Gemini CLI call, normalized.

    ``content`` is the accumulated text across all stream events.
    ``latency_ms`` is total wall-clock for the logical call including
    retries. ``prompt_tokens`` / ``completion_tokens`` come from a
    final usage event if Gemini emits one; both default to 0 when no
    such event appears. ``attempts`` records how many subprocess
    invocations were needed (1 on the happy path, up to
    ``1 + retries`` on a recovered failure).
    """

    content: str
    model: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    raw_events: tuple[dict[str, Any], ...] = field(default_factory=tuple)


# ── Pure helpers ──────────────────────────────────────────────────────────────


_ROLE_BANNER = {
    "system": "=== SYSTEM INSTRUCTIONS ===",
    "user": "=== USER ===",
    "assistant": "=== ASSISTANT ===",
}


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    """Collapse OpenAI-style messages into one Gemini-CLI-shaped prompt.

    Gemini CLI takes a single prompt string; we preserve the role
    structure with banners so the model knows which content is
    instructions vs the query. Unknown roles surface their literal
    role name (defensive against future role types).

    Raises:
        ValueError: empty messages list. The compose stage never calls
            us with an empty list — :func:`compose_brief` routes the
            zero-items case to :func:`render_empty_brief` first — so a
            ValueError here is a programmer error, not a runtime case.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    parts: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        banner = _ROLE_BANNER.get(role, f"=== {role.upper() or 'UNKNOWN'} ===")
        content = (msg.get("content") or "").strip()
        parts.append(banner)
        parts.append(content)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _build_argv(
    config: GeminiCliProviderConfig,
    *,
    prompt: str,
    model: str,
) -> list[str]:
    """Construct the argv for ``asyncio.create_subprocess_exec``.

    The prompt rides on ``-p`` rather than stdin. Python's
    ``create_subprocess_exec`` uses ``execve`` directly (no shell), so
    there's no shell-escape risk even with multi-kilobyte prompts.
    macOS ``ARG_MAX`` is ~1 MiB which comfortably covers a ~100 KiB
    brief composition prompt.

    ``--approval-mode plan`` is the read-only setting — the CLI's
    tool-use surface is blocked. Pure prose generation; no actions
    the model could take that might affect the host.
    """
    head: list[str]
    if config.executable_path:
        head = [config.executable_path, config.script_path]
    else:
        head = [config.script_path]
    return [
        *head,
        "-p",
        prompt,
        "-o",
        config.output_format,
        "-m",
        model,
        "--approval-mode",
        config.approval_mode,
    ]


def _extract_event_text(event: dict[str, Any]) -> str:
    """Pull text content out of one stream-json event.

    Gemini CLI v0.36.0 (verified live, 2026-05-15) emits events shaped
    like ``{"type": "message", "role": "<user|assistant>", "content":
    "..."}`` alongside an ``init`` event and a final ``result`` event.
    Critically, the CLI echoes the flattened user prompt back as a
    ``role: "user"`` message **before** the assistant reply; treating
    that echo as response content would prepend the entire prompt
    (system instructions + cluster summaries) to the brief. So when
    an event carries a ``role`` field, we only accept its content
    when the role is ``assistant``. Events without a ``role`` field
    fall through to the permissive multi-shape inspection so a
    future CLI version that drops the role tag still works.

    The helper inspects ``text``, ``content``, ``delta``, and nested
    ``message.content`` in priority order. Unknown shapes yield
    empty string so the caller treats them as a heartbeat/metadata
    event rather than erroring.
    """
    if not isinstance(event, dict):
        return ""
    role = event.get("role")
    if isinstance(role, str) and role.strip().lower() != "assistant":
        # Filters the user-echo (and any future system-banner) events.
        return ""
    # Direct text fields, in priority order.
    for key in ("text", "content", "delta"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    # Nested message.content shape.
    message = event.get("message")
    if isinstance(message, dict):
        nested = message.get("content")
        if isinstance(nested, str) and nested:
            return nested
    return ""


def _extract_event_usage(event: dict[str, Any]) -> tuple[int, int] | None:
    """Pull (prompt_tokens, completion_tokens) out of a usage event.

    Returns ``None`` when the event isn't a usage summary. Gemini CLI
    v0.36.0 emits its token counts as ``{"type": "result", "stats":
    {"input_tokens": ..., "output_tokens": ...}}`` (verified live
    2026-05-15); we also accept the more standard ``usage``-nested
    shape and flat ``prompt_tokens`` / ``completion_tokens`` keys so
    a future CLI emission shape doesn't silently drop accounting.
    Returns 0/0 (a tuple, not None) for usage events with missing or
    non-integer values so the audit row still distinguishes "received
    a usage event" from "didn't".
    """
    if not isinstance(event, dict):
        return None
    candidate: dict[str, Any] | None = None
    for key in ("usage", "stats"):
        block = event.get(key)
        if isinstance(block, dict):
            candidate = block
            break
    source = candidate if candidate is not None else event
    if not (
        "prompt_tokens" in source
        or "completion_tokens" in source
        or "input_tokens" in source
        or "output_tokens" in source
    ):
        return None
    pt = source.get("prompt_tokens", source.get("input_tokens", 0))
    ct = source.get("completion_tokens", source.get("output_tokens", 0))
    try:
        return (int(pt), int(ct))
    except (TypeError, ValueError):
        return (0, 0)


def _short_stderr_tail(stderr: bytes, *, limit: int = 280) -> str:
    """Trim stderr to a single audit-log-friendly line."""
    text = stderr.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    # Take the last line — it's usually the actionable error message.
    last_line = text.splitlines()[-1] if text.splitlines() else text
    if len(last_line) > limit:
        last_line = last_line[: limit - 3] + "..."
    return last_line


# ── Async layer ──────────────────────────────────────────────────────────────


async def gemini_cli_completion(
    config: GeminiCliProviderConfig,
    *,
    messages: list[dict[str, str]],
    model: str,
) -> GeminiCliResult:
    """Run one Gemini CLI completion, with stall detection + retry.

    The call dispatches the CLI subprocess, reads its stream-json
    output line-by-line under both an idle-gap and a hard wall-clock
    deadline, accumulates text from each event, and returns a
    :class:`GeminiCliResult` on success.

    On any failure mode (stall, hard timeout, non-zero exit, output
    empty after stream end), the function retries up to
    ``config.retries`` additional times with ``retry_backoff_seconds``
    sleep between attempts. After the last attempt, the most recent
    error is re-raised.

    Raises:
        GeminiCliStallError: idle gap between events exceeded
            ``idle_timeout_seconds`` on the final attempt.
        GeminiCliTimeoutError: total wall-clock exceeded
            ``hard_timeout_seconds`` on the final attempt.
        GeminiCliExitError: subprocess exited non-zero on the final
            attempt.
        GeminiCliOutputError: subprocess exited 0 but emitted no text
            events on the final attempt.
        ValueError: empty ``messages`` list (programmer error).
    """
    prompt = _flatten_messages(messages)
    argv = _build_argv(config, prompt=prompt, model=model)

    wall_start = time.perf_counter()
    last_error: GeminiCliError | None = None

    attempts_made = 0
    total_attempts = 1 + max(0, config.retries)
    for attempt_idx in range(total_attempts):
        attempts_made = attempt_idx + 1
        try:
            content, raw_events, usage = await _run_one_subprocess(config, argv)
            latency_ms = int((time.perf_counter() - wall_start) * 1000)
            return GeminiCliResult(
                content=content,
                model=model,
                latency_ms=latency_ms,
                prompt_tokens=usage[0],
                completion_tokens=usage[1],
                attempts=attempts_made,
                raw_events=tuple(raw_events),
            )
        except GeminiCliError as exc:
            last_error = exc
            if attempt_idx == total_attempts - 1:
                # Last attempt — surface the failure.
                break
            log.warning(
                "gemini_cli attempt %d/%d failed (%s); retrying in %.1fs",
                attempts_made,
                total_attempts,
                type(exc).__name__,
                config.retry_backoff_seconds,
            )
            await asyncio.sleep(config.retry_backoff_seconds)

    assert last_error is not None  # at least one attempt must have run
    raise last_error


async def _run_one_subprocess(
    config: GeminiCliProviderConfig,
    argv: list[str],
) -> tuple[str, list[dict[str, Any]], tuple[int, int]]:
    """Spawn one subprocess attempt; read its stream; enforce timeouts.

    Returns ``(content, raw_events, (prompt_tokens, completion_tokens))``.
    On any failure mode, terminates the process cleanly (SIGTERM with
    grace, then SIGKILL) before raising.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    deadline = time.monotonic() + config.hard_timeout_seconds
    content_parts: list[str] = []
    raw_events: list[dict[str, Any]] = []
    usage: tuple[int, int] = (0, 0)

    assert proc.stdout is not None  # PIPE configured above

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GeminiCliTimeoutError(
                    f"gemini exceeded hard timeout of {config.hard_timeout_seconds:.0f}s"
                )
            per_line_wait = min(config.idle_timeout_seconds, remaining)
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=per_line_wait)
            except asyncio.TimeoutError as exc:
                if time.monotonic() >= deadline:
                    raise GeminiCliTimeoutError(
                        f"gemini exceeded hard timeout of {config.hard_timeout_seconds:.0f}s"
                    ) from exc
                raise GeminiCliStallError(
                    f"gemini stream went idle for >{config.idle_timeout_seconds:.0f}s"
                ) from exc
            if not line:
                # EOF — subprocess closed stdout. Wait for exit code below.
                break
            event = _parse_event(line)
            if event is None:
                continue
            raw_events.append(event)
            text = _extract_event_text(event)
            if text:
                content_parts.append(text)
            event_usage = _extract_event_usage(event)
            if event_usage is not None:
                usage = event_usage

        # Drain stderr + collect exit code. ``wait_for`` keeps us bounded
        # in case the process closed stdout but is still alive.
        try:
            stderr_bytes = await asyncio.wait_for(
                proc.stderr.read() if proc.stderr is not None else _empty_bytes(),
                timeout=max(0.5, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError:
            stderr_bytes = b""
        returncode = await asyncio.wait_for(
            proc.wait(),
            timeout=max(0.5, deadline - time.monotonic()),
        )

        if returncode != 0:
            tail = _short_stderr_tail(stderr_bytes)
            raise GeminiCliExitError(returncode, tail)

        content = "".join(content_parts).strip()
        if not content:
            raise GeminiCliOutputError(
                f"gemini exited 0 but emitted no text events (raw events: {len(raw_events)})"
            )

        return content, raw_events, usage

    except BaseException:
        await _kill_process(proc)
        raise


async def _empty_bytes() -> bytes:
    """Awaitable returning empty bytes — used when stderr is None."""
    return b""


def _parse_event(line: bytes) -> dict[str, Any] | None:
    """Decode one stream-json line.

    Non-JSON lines (debug noise, blank lines, partial flushes) are
    treated as heartbeats — logged at debug, not surfaced as errors.
    The architecture-doc rule "always publish a useful brief" says
    we tolerate format drift; the live smoke at 12c will tighten
    this if Gemini's actual emission shape needs different handling.
    """
    text = line.strip()
    if not text:
        return None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        log.debug("gemini stream non-json line skipped: %r", text[:80])
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


async def _kill_process(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM → grace period → SIGKILL.

    Always drains the child cleanly so we never leak a zombie. Any
    further exceptions during shutdown are logged but not raised —
    the caller is already in an except-and-reraise path, and shadowing
    the original error would be worse than missing the kill detail.
    """
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    except Exception:
        log.exception("gemini_cli: terminate() raised; continuing to wait()")

    try:
        await asyncio.wait_for(proc.wait(), timeout=_GRACE_PERIOD_SECONDS)
        return
    except asyncio.TimeoutError:
        pass

    try:
        proc.kill()
    except ProcessLookupError:
        return
    except Exception:
        log.exception("gemini_cli: kill() raised; continuing to wait()")

    try:
        await asyncio.wait_for(proc.wait(), timeout=_GRACE_PERIOD_SECONDS)
    except asyncio.TimeoutError:
        log.error("gemini_cli: process still alive after SIGKILL; abandoning")


__all__ = (
    "GeminiCliError",
    "GeminiCliExitError",
    "GeminiCliOutputError",
    "GeminiCliProviderConfig",
    "GeminiCliResult",
    "GeminiCliStallError",
    "GeminiCliTimeoutError",
    "gemini_cli_completion",
)
