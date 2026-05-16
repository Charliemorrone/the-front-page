"""Tests for the Gemini CLI subprocess provider (Step 12a).

The provider invokes ``gemini`` as a subprocess and reads stream-json
events from its stdout. Tests must not invoke a live Gemini API — we
monkeypatch :func:`asyncio.create_subprocess_exec` to return a
controllable :class:`FakeProcess` and exercise:

- happy-path stream → result with accumulated content
- event-shape permissiveness (text / content / delta / nested message)
- final usage event populates token counts
- malformed JSON line tolerated as heartbeat
- stall detection (idle gap > ``idle_timeout_seconds``) → kill + raise
- hard timeout (total wall > ``hard_timeout_seconds``) → kill + raise
- non-zero exit → raise with stderr tail
- output empty after stream end → raise
- retry recovers a transient failure
- retries exhausted → final error surfaced
- ``retries=0`` skips retry entirely
- SIGTERM grace → SIGKILL when the child resists terminate

Pure helpers (_flatten_messages, _build_argv, _extract_event_text,
_extract_event_usage) get their own focused tests so behavior is
pinned without spinning up the subprocess machinery.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from clawfeed_intel.llm import gemini_cli
from clawfeed_intel.llm.gemini_cli import (
    GeminiCliError,
    GeminiCliExitError,
    GeminiCliOutputError,
    GeminiCliProviderConfig,
    GeminiCliResult,
    GeminiCliStallError,
    GeminiCliTimeoutError,
    _build_argv,
    _extract_event_text,
    _extract_event_usage,
    _flatten_messages,
    gemini_cli_completion,
)


# ── Fake subprocess machinery ─────────────────────────────────────────────────


class _FakeStdout:
    """In-memory stream that yields one line at a time and signals EOF.

    Set ``hang_after`` to the index at which ``readline`` should block
    forever (used for stall + hard-timeout tests). The hang is an
    asyncio.sleep(3600) so the caller's ``asyncio.wait_for`` cancels
    it cleanly when the timeout fires.
    """

    def __init__(self, lines: list[bytes], *, hang_after: int | None = None) -> None:
        self._lines = list(lines)
        self._hang_after = hang_after
        self._idx = 0
        self.eof_seen = False

    async def readline(self) -> bytes:
        if self._hang_after is not None and self._idx >= self._hang_after:
            await asyncio.sleep(3600)
            return b""
        if self._idx >= len(self._lines):
            self.eof_seen = True
            return b""
        line = self._lines[self._idx]
        self._idx += 1
        return line


class _FakeStderr:
    """One-shot bytes buffer mimicking ``StreamReader.read()``."""

    def __init__(self, content: bytes = b"") -> None:
        self._content = content
        self._read = False

    async def read(self) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._content


class FakeProcess:
    """Drop-in replacement for ``asyncio.subprocess.Process`` in tests.

    Behavior knobs:
        ``returncode``: clean-exit code when stdout drains naturally.
        ``hang_after_line``: index past which ``stdout.readline`` blocks
            forever — exercises stall + hard timeout paths.
        ``respect_terminate``: when False, ``terminate()`` is recorded
            but ``wait()`` continues to block until ``kill()`` is
            called — exercises the SIGTERM → SIGKILL escalation.
    """

    def __init__(
        self,
        *,
        stdout_lines: list[bytes] | None = None,
        stderr_bytes: bytes = b"",
        returncode: int = 0,
        hang_after_line: int | None = None,
        respect_terminate: bool = True,
    ) -> None:
        self.stdout = _FakeStdout(stdout_lines or [], hang_after=hang_after_line)
        self.stderr = _FakeStderr(stderr_bytes)
        self._target_returncode = returncode
        self._respect_terminate = respect_terminate
        self.returncode: int | None = None
        self.terminate_called = False
        self.kill_called = False
        self._exit_signal = asyncio.Event()

    def terminate(self) -> None:
        self.terminate_called = True
        if self._respect_terminate:
            self.returncode = -15
            self._exit_signal.set()

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -9
        self._exit_signal.set()

    async def wait(self) -> int:
        if self.returncode is not None:
            return self.returncode
        if self.stdout.eof_seen:
            self.returncode = self._target_returncode
            return self.returncode
        await self._exit_signal.wait()
        assert self.returncode is not None
        return self.returncode


def _events_to_lines(events: list[dict[str, Any]]) -> list[bytes]:
    return [json.dumps(e).encode("utf-8") + b"\n" for e in events]


def _patch_exec(
    monkeypatch: pytest.MonkeyPatch,
    proc_factory,
) -> list[list[str]]:
    """Patch ``asyncio.create_subprocess_exec`` to return a fake process.

    ``proc_factory`` may be a callable (per-attempt scenario) or an
    iterator of FakeProcesses (one per attempt; used for retry tests).
    Returns a list captured argvs so tests can assert dispatch shape.
    """
    captured_argv: list[list[str]] = []

    if callable(proc_factory):
        factory = proc_factory
    else:
        iterator = iter(proc_factory)

        def factory():
            return next(iterator)

    async def fake_exec(*args, **kwargs):
        captured_argv.append(list(args))
        return factory()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return captured_argv


def _config(**overrides: Any) -> GeminiCliProviderConfig:
    """Build a config with test-friendly fast defaults.

    Tests use sub-second timeouts so the suite stays under a few
    seconds total even when exercising the stall + hard-timeout paths.
    Production defaults (60s / 300s) are validated by the dataclass
    defaults test.
    """
    defaults = {
        "script_path": "/fake/gemini",
        "executable_path": "/fake/node",
        "approval_mode": "plan",
        "output_format": "stream-json",
        "idle_timeout_seconds": 0.5,
        "hard_timeout_seconds": 2.0,
        "retries": 0,
        "retry_backoff_seconds": 0.0,
    }
    defaults.update(overrides)
    return GeminiCliProviderConfig(**defaults)


def _messages(user_text: str = "Compose the brief.") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are the composer."},
        {"role": "user", "content": user_text},
    ]


# ── Pure helpers ──────────────────────────────────────────────────────────────


def test_flatten_messages_emits_role_banners() -> None:
    out = _flatten_messages(
        [
            {"role": "system", "content": "system body"},
            {"role": "user", "content": "user body"},
        ]
    )
    assert "=== SYSTEM INSTRUCTIONS ===" in out
    assert "=== USER ===" in out
    assert "system body" in out
    assert "user body" in out
    # System block precedes user block.
    assert out.index("system body") < out.index("user body")


def test_flatten_messages_unknown_role_uses_literal_banner() -> None:
    out = _flatten_messages([{"role": "tool", "content": "x"}])
    assert "=== TOOL ===" in out


def test_flatten_messages_empty_role_uses_unknown_banner() -> None:
    out = _flatten_messages([{"role": "", "content": "x"}])
    assert "=== UNKNOWN ===" in out


def test_flatten_messages_strips_whitespace() -> None:
    out = _flatten_messages([{"role": "user", "content": "  hello  \n"}])
    assert "hello" in out
    assert "  hello  " not in out


def test_flatten_messages_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _flatten_messages([])


def test_build_argv_with_executable_path() -> None:
    cfg = _config(executable_path="/usr/bin/node", script_path="/bin/gemini")
    argv = _build_argv(cfg, prompt="hi", model="gemini-2.5-pro")
    assert argv[0:2] == ["/usr/bin/node", "/bin/gemini"]
    assert "-p" in argv
    assert "hi" in argv
    assert "-m" in argv and "gemini-2.5-pro" in argv
    assert "--approval-mode" in argv and "plan" in argv
    assert "-o" in argv and "stream-json" in argv


def test_build_argv_without_executable_path_uses_script_only() -> None:
    cfg = _config(executable_path=None, script_path="/bin/gemini")
    argv = _build_argv(cfg, prompt="hi", model="gemini-2.5-pro")
    assert argv[0] == "/bin/gemini"
    # No node binary prepended.
    assert argv[1] != "/bin/gemini"


def test_extract_event_text_text_field_wins() -> None:
    assert _extract_event_text({"text": "alpha", "content": "beta"}) == "alpha"


def test_extract_event_text_content_field_fallback() -> None:
    assert _extract_event_text({"content": "beta"}) == "beta"


def test_extract_event_text_delta_field_fallback() -> None:
    assert _extract_event_text({"delta": "gamma"}) == "gamma"


def test_extract_event_text_nested_message_content() -> None:
    assert _extract_event_text({"message": {"content": "nested"}}) == "nested"


def test_extract_event_text_empty_when_unknown_shape() -> None:
    assert _extract_event_text({"event_type": "heartbeat"}) == ""


def test_extract_event_text_handles_non_dict() -> None:
    assert _extract_event_text("not a dict") == ""  # type: ignore[arg-type]


def test_extract_event_usage_nested_under_usage_key() -> None:
    out = _extract_event_usage({"usage": {"prompt_tokens": 42, "completion_tokens": 17}})
    assert out == (42, 17)


def test_extract_event_usage_flat_keys() -> None:
    out = _extract_event_usage({"input_tokens": 100, "output_tokens": 50})
    assert out == (100, 50)


def test_extract_event_usage_none_for_non_usage_event() -> None:
    assert _extract_event_usage({"text": "hello"}) is None


def test_extract_event_usage_non_integer_returns_zero() -> None:
    # Defensive: missing values shouldn't blow up the audit log.
    out = _extract_event_usage({"prompt_tokens": "lots", "completion_tokens": None})
    assert out == (0, 0)


# ── Live event-shape regression guards (Gemini CLI v0.36.0, 2026-05-15) ───────
#
# These tests pin the actual event shapes captured from a live
# invocation against the operator's Gemini Pro account during Step 12c.
# They are load-bearing because the original provider, written before
# the live smoke, would have leaked the user-prompt echo into the
# composed brief and silently lost usage accounting. See the Step 12c
# build-log entry in docs/current-project-status.md for the captured
# raw events used here.


def test_extract_event_text_skips_user_role_echo() -> None:
    """Load-bearing: Gemini CLI echoes the user prompt back as a
    ``role: "user"`` message before the assistant reply. Treating
    that as content would prepend the full prompt to the brief.
    """
    echo = {
        "type": "message",
        "timestamp": "2026-05-15T17:35:14.514Z",
        "role": "user",
        "content": "Reply with exactly: PONG\n",
    }
    assert _extract_event_text(echo) == ""


def test_extract_event_text_accepts_assistant_role() -> None:
    assistant = {
        "type": "message",
        "timestamp": "2026-05-15T17:35:18.414Z",
        "role": "assistant",
        "content": "PONG",
        "delta": True,
    }
    assert _extract_event_text(assistant) == "PONG"


def test_extract_event_text_omitted_role_falls_through() -> None:
    """Back-compat: events without a ``role`` field still surface
    content. The role guard only filters when role is explicitly set
    to something non-assistant, so a future CLI emission that drops
    the role tag still works.
    """
    assert _extract_event_text({"text": "hello", "type": "content"}) == "hello"


def test_extract_event_text_init_event_is_empty() -> None:
    """Gemini's init event carries metadata only — no content fields."""
    init = {
        "type": "init",
        "timestamp": "2026-05-15T17:35:14.513Z",
        "session_id": "d5d25e59-2b9f-4e5f-9628-f05507203889",
        "model": "gemini-2.5-pro",
    }
    assert _extract_event_text(init) == ""


def test_extract_event_text_result_event_is_empty() -> None:
    """Gemini's final result event carries stats but no text content."""
    result = {
        "type": "result",
        "timestamp": "2026-05-15T17:35:18.575Z",
        "status": "success",
        "stats": {"total_tokens": 6775, "input_tokens": 6683, "output_tokens": 2},
    }
    assert _extract_event_text(result) == ""


def test_extract_event_text_skips_system_role() -> None:
    """Defensive: any non-assistant role is filtered, not just user."""
    sys_event = {"type": "message", "role": "system", "content": "you are an assistant"}
    assert _extract_event_text(sys_event) == ""


def test_extract_event_usage_from_stats_block() -> None:
    """Load-bearing: Gemini CLI puts token counts under ``stats``,
    not ``usage``. Without this fallback the audit log would record
    0/0 for every Gemini call and the dashboard's per-call accounting
    would be wrong.
    """
    out = _extract_event_usage({"stats": {"input_tokens": 6683, "output_tokens": 2}})
    assert out == (6683, 2)


def test_extract_event_usage_live_result_event_shape() -> None:
    """Full captured shape from the 2026-05-15 live PONG probe."""
    result = {
        "type": "result",
        "timestamp": "2026-05-15T17:35:18.575Z",
        "status": "success",
        "stats": {
            "total_tokens": 6775,
            "input_tokens": 6683,
            "output_tokens": 2,
            "cached": 0,
            "input": 6683,
            "duration_ms": 4062,
            "tool_calls": 0,
            "models": {
                "gemini-2.5-pro": {
                    "total_tokens": 6775,
                    "input_tokens": 6683,
                    "output_tokens": 2,
                    "cached": 0,
                    "input": 6683,
                }
            },
        },
    }
    assert _extract_event_usage(result) == (6683, 2)


def test_extract_event_usage_usage_block_still_preferred_when_present() -> None:
    """``usage`` (the more standard shape) is tried before ``stats``."""
    event = {
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "stats": {"input_tokens": 999, "output_tokens": 888},
    }
    assert _extract_event_usage(event) == (100, 50)


def test_gemini_cli_completion_handles_live_v036_event_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end pin against the Gemini CLI v0.36.0 stream shape.

    Replays the exact event sequence captured from a live PONG probe:
    init → user-echo → assistant → result. The expected content is
    just ``"PONG"`` (the assistant reply) — NOT the user-echo content,
    which would otherwise leak the full prompt into the brief.
    """
    events = [
        {
            "type": "init",
            "timestamp": "2026-05-15T17:35:14.513Z",
            "session_id": "d5d25e59-2b9f-4e5f-9628-f05507203889",
            "model": "gemini-2.5-pro",
        },
        {
            "type": "message",
            "timestamp": "2026-05-15T17:35:14.514Z",
            "role": "user",
            "content": "Reply with exactly: PONG\n",
        },
        {
            "type": "message",
            "timestamp": "2026-05-15T17:35:18.414Z",
            "role": "assistant",
            "content": "PONG",
            "delta": True,
        },
        {
            "type": "result",
            "timestamp": "2026-05-15T17:35:18.575Z",
            "status": "success",
            "stats": {
                "total_tokens": 6775,
                "input_tokens": 6683,
                "output_tokens": 2,
                "cached": 0,
                "input": 6683,
                "duration_ms": 4062,
                "tool_calls": 0,
            },
        },
    ]
    proc = FakeProcess(stdout_lines=_events_to_lines(events), returncode=0)
    _patch_exec(monkeypatch, lambda: proc)

    result = asyncio.run(
        gemini_cli_completion(_config(), messages=_messages(), model="gemini-2.5-pro")
    )

    # Load-bearing assertions: content is the assistant reply only,
    # tokens come from the result event's stats block.
    assert result.content == "PONG"
    assert result.prompt_tokens == 6683
    assert result.completion_tokens == 2
    assert result.model == "gemini-2.5-pro"


# ── Async happy path ──────────────────────────────────────────────────────────


def test_gemini_cli_completion_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "content", "text": "# Daily Brief\n\n"},
        {"type": "content", "text": "## Section\n"},
        {"type": "content", "text": "body."},
        {"usage": {"prompt_tokens": 1234, "completion_tokens": 567}},
    ]
    proc = FakeProcess(stdout_lines=_events_to_lines(events), returncode=0)
    argvs = _patch_exec(monkeypatch, lambda: proc)

    cfg = _config()
    result = asyncio.run(gemini_cli_completion(cfg, messages=_messages(), model="gemini-2.5-pro"))

    assert isinstance(result, GeminiCliResult)
    assert result.content == "# Daily Brief\n\n## Section\nbody."
    assert result.model == "gemini-2.5-pro"
    assert result.prompt_tokens == 1234
    assert result.completion_tokens == 567
    assert result.attempts == 1
    assert len(argvs) == 1


def test_happy_path_uses_executable_path_and_passes_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [{"text": "ok"}]
    proc = FakeProcess(stdout_lines=_events_to_lines(events), returncode=0)
    argvs = _patch_exec(monkeypatch, lambda: proc)

    cfg = _config(executable_path="/exec/node", script_path="/exec/gemini")
    asyncio.run(gemini_cli_completion(cfg, messages=_messages(), model="gemini-2.5-pro"))

    argv = argvs[0]
    assert argv[:2] == ["/exec/node", "/exec/gemini"]
    assert "gemini-2.5-pro" in argv


def test_malformed_json_lines_treated_as_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-JSON noise on the stream must not abort the response.

    Gemini CLI may emit progress lines, blank lines, or partial flushes
    interspersed with the stream-json events. Treating these as
    heartbeats means we tolerate format drift without losing content.
    """
    lines = [
        b"some debug noise from the CLI\n",
        b"\n",
        json.dumps({"text": "good content"}).encode() + b"\n",
        b"{ partial json\n",
        json.dumps({"text": " more good content"}).encode() + b"\n",
    ]
    proc = FakeProcess(stdout_lines=lines, returncode=0)
    _patch_exec(monkeypatch, lambda: proc)

    result = asyncio.run(
        gemini_cli_completion(_config(), messages=_messages(), model="gemini-2.5-pro")
    )
    assert result.content == "good content more good content"


# ── Stall + hard-timeout ─────────────────────────────────────────────────────


def test_stall_detected_after_idle_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """One event, then the stream hangs → GeminiCliStallError fires fast.

    Verifies the per-token-idle threshold is enforced separately from
    the wall-clock cap: with idle=0.2s and hard=2.0s, a true stall
    raises in ~0.2s, not 2.0s.
    """
    events = [{"text": "start"}]
    proc = FakeProcess(
        stdout_lines=_events_to_lines(events),
        hang_after_line=1,
        respect_terminate=True,
    )
    _patch_exec(monkeypatch, lambda: proc)

    cfg = _config(idle_timeout_seconds=0.2, hard_timeout_seconds=5.0, retries=0)

    started = time.perf_counter()
    with pytest.raises(GeminiCliStallError, match="idle"):
        asyncio.run(gemini_cli_completion(cfg, messages=_messages(), model="x"))
    elapsed = time.perf_counter() - started

    # Should raise quickly — way under the hard timeout.
    assert elapsed < 1.5
    assert proc.terminate_called


def test_hard_timeout_raises_when_stream_never_idles_but_runs_too_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wall-clock cap fires even when the idle threshold is never tripped.

    Setting ``idle_timeout_seconds > hard_timeout_seconds`` means the
    idle check is effectively disabled — only the hard wall-clock can
    trip. The stream hangs on first read, hard cap fires.
    """
    proc = FakeProcess(stdout_lines=[], hang_after_line=0, respect_terminate=True)
    _patch_exec(monkeypatch, lambda: proc)

    cfg = _config(idle_timeout_seconds=10.0, hard_timeout_seconds=0.3, retries=0)

    with pytest.raises(GeminiCliTimeoutError, match="hard timeout"):
        asyncio.run(gemini_cli_completion(cfg, messages=_messages(), model="x"))


def test_stall_path_terminates_then_kills_when_terminate_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the child resists SIGTERM, the provider escalates to SIGKILL.

    Models the worst case: subprocess is hung deep enough that SIGTERM
    doesn't make it exit. The provider must escalate so we never leak
    a zombie holding the asyncio loop.
    """
    proc = FakeProcess(
        stdout_lines=[],
        hang_after_line=0,
        respect_terminate=False,
    )
    _patch_exec(monkeypatch, lambda: proc)

    cfg = _config(idle_timeout_seconds=0.1, hard_timeout_seconds=2.0, retries=0)

    # Speed the grace period for testing — patch the module constant.
    monkeypatch.setattr(gemini_cli, "_GRACE_PERIOD_SECONDS", 0.1)

    with pytest.raises(GeminiCliStallError):
        asyncio.run(gemini_cli_completion(cfg, messages=_messages(), model="x"))

    assert proc.terminate_called
    assert proc.kill_called


# ── Exit-code + empty-output ────────────────────────────────────────────────


def test_non_zero_exit_raises_with_stderr_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = FakeProcess(
        stdout_lines=[],
        stderr_bytes=b"Error: auth refresh failed\n",
        returncode=1,
    )
    _patch_exec(monkeypatch, lambda: proc)

    with pytest.raises(GeminiCliExitError) as excinfo:
        asyncio.run(gemini_cli_completion(_config(), messages=_messages(), model="gemini-2.5-pro"))

    err = excinfo.value
    assert err.returncode == 1
    assert "auth refresh failed" in err.stderr_tail


def test_empty_content_raises_output_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stream exits 0 with only metadata events → GeminiCliOutputError.

    The CLI shouldn't emit zero content events on success; if it does,
    the fallback chain treats it the same as a crash.
    """
    events = [{"event_type": "heartbeat"}, {"event_type": "done"}]
    proc = FakeProcess(stdout_lines=_events_to_lines(events), returncode=0)
    _patch_exec(monkeypatch, lambda: proc)

    with pytest.raises(GeminiCliOutputError, match="no text events"):
        asyncio.run(gemini_cli_completion(_config(), messages=_messages(), model="gemini-2.5-pro"))


# ── Retry behaviour ──────────────────────────────────────────────────────────


def test_retry_recovers_after_one_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """First attempt stalls; second attempt succeeds; result reflects both.

    ``attempts`` increments so the audit row can record how many
    subprocess invocations were needed for a successful brief.
    """
    failing = FakeProcess(stdout_lines=[], hang_after_line=0, respect_terminate=True)
    succeeding = FakeProcess(
        stdout_lines=_events_to_lines([{"text": "recovered content"}]),
        returncode=0,
    )
    argvs = _patch_exec(monkeypatch, [failing, succeeding])

    cfg = _config(retries=1, retry_backoff_seconds=0.0)

    result = asyncio.run(gemini_cli_completion(cfg, messages=_messages(), model="gemini-2.5-pro"))

    assert result.content == "recovered content"
    assert result.attempts == 2
    assert len(argvs) == 2
    assert failing.terminate_called


def test_retries_exhausted_raises_final_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every attempt fails → the last attempt's exception is what surfaces.

    Caller (compose-stage fallback chain) sees a single ``GeminiCliError``
    and routes to Tier-2 vMLX fallback.
    """
    fail_a = FakeProcess(stdout_lines=[], stderr_bytes=b"first fail\n", returncode=1)
    fail_b = FakeProcess(stdout_lines=[], stderr_bytes=b"second fail\n", returncode=2)
    _patch_exec(monkeypatch, [fail_a, fail_b])

    cfg = _config(retries=1, retry_backoff_seconds=0.0)

    with pytest.raises(GeminiCliExitError) as excinfo:
        asyncio.run(gemini_cli_completion(cfg, messages=_messages(), model="x"))

    assert excinfo.value.returncode == 2  # last attempt's code, not first
    assert "second fail" in excinfo.value.stderr_tail


def test_retries_zero_surfaces_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """``retries=0`` means single attempt — first failure is final."""
    proc = FakeProcess(stdout_lines=[], stderr_bytes=b"oauth\n", returncode=1)
    argvs = _patch_exec(monkeypatch, lambda: proc)

    cfg = _config(retries=0, retry_backoff_seconds=0.0)

    with pytest.raises(GeminiCliExitError):
        asyncio.run(gemini_cli_completion(cfg, messages=_messages(), model="x"))

    assert len(argvs) == 1


def test_retry_inherits_subclass_relationships() -> None:
    """All Gemini CLI errors derive from GeminiCliError.

    Lets the compose-stage fallback handler catch the base class
    without enumerating every subtype.
    """
    for cls in (
        GeminiCliStallError,
        GeminiCliTimeoutError,
        GeminiCliExitError,
        GeminiCliOutputError,
    ):
        assert issubclass(cls, GeminiCliError)


# ── Config defaults ───────────────────────────────────────────────────────────


def test_provider_config_defaults_match_design() -> None:
    """The architecture-doc-defined defaults are pinned here.

    Tests in the file use overrides for speed; production callers
    rely on these defaults. A bump must be deliberate.
    """
    cfg = GeminiCliProviderConfig(script_path="/x")
    assert cfg.idle_timeout_seconds == 60.0
    assert cfg.hard_timeout_seconds == 300.0
    assert cfg.approval_mode == "plan"
    assert cfg.output_format == "stream-json"
    assert cfg.retries == 1
    assert cfg.retry_backoff_seconds == 10.0
