"""Tests for the ``clawfeed-intel run topic`` Phase 7a scaffold.

Mirrors ``test_cli_run_daily.py``'s structure: stub the probe layer
and the orchestrator entrypoint so no live vMLX or live fetcher HTTP
fires under CI. The CLI's contract for topic runs matches the daily
preflight posture (exit 3 on routing / vMLX failure) plus a topic-
specific boundary check (exit 2 on empty query).
"""

from __future__ import annotations

import argparse

import pytest

from clawfeed_intel import cli, doctor
from clawfeed_intel.doctor import ProbeResult
from clawfeed_intel.llm import RoutingConfig


def _routing() -> RoutingConfig:
    return RoutingConfig.model_validate(
        {
            "providers": {"vmlx": {"base_url": "http://127.0.0.1:8080/v1"}},
            "stages": {
                "relevance_filter": {
                    "provider": "vmlx",
                    "model": "m",
                    "timeout_seconds": 60,
                },
            },
        }
    )


def _patch_probes(monkeypatch: pytest.MonkeyPatch, results: list[ProbeResult]) -> None:
    async def fake_probes(*_args: object, **_kwargs: object) -> list[ProbeResult]:
        return list(results)

    monkeypatch.setattr(doctor, "run_doctor_probes", fake_probes)
    monkeypatch.setattr(cli, "load_routing", _routing)


def _track_run_topic(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Replace ``cli.run_topic`` with a stub recording (query, window_days)."""
    calls: list[tuple[str, int]] = []

    def stub(query: str, *, window_days: int) -> int:
        calls.append((query, window_days))
        return 5151

    monkeypatch.setattr(cli, "run_topic", stub)
    return calls


def _topic_args(query: str = "Khosla Ventures", window_days: int = 30) -> argparse.Namespace:
    return argparse.Namespace(
        cmd="run",
        run_type="topic",
        query=query,
        window_days=window_days,
    )


def test_run_topic_proceeds_when_all_probes_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_probes(
        monkeypatch,
        [
            ProbeResult("health", True, 5, "status=ok"),
            ProbeResult("models", True, 10, "1 model(s): m"),
            ProbeResult("chat:relevance_filter", True, 100, "m → 'PONG'"),
        ],
    )
    calls = _track_run_topic(monkeypatch)

    rc = cli.cmd_run_topic(_topic_args())
    captured = capsys.readouterr()

    assert rc == 0
    assert calls == [("Khosla Ventures", 30)]
    assert "published topic digest 5151" in captured.out
    # Happy path is silent on stderr — preflight noise only when something fails.
    assert captured.err == ""


def test_run_topic_respects_custom_window_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_probes(
        monkeypatch,
        [
            ProbeResult("health", True, 5, "status=ok"),
            ProbeResult("models", True, 10, "1 model(s): m"),
            ProbeResult("chat:relevance_filter", True, 100, "m → 'PONG'"),
        ],
    )
    calls = _track_run_topic(monkeypatch)

    rc = cli.cmd_run_topic(_topic_args(window_days=7))
    assert rc == 0
    assert calls == [("Khosla Ventures", 7)]


def test_run_topic_aborts_with_exit_3_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_probes(
        monkeypatch,
        [
            ProbeResult("health", True, 5, "status=ok"),
            ProbeResult("models", True, 10, "1 model(s): m"),
            ProbeResult(
                "chat:relevance_filter",
                False,
                5000,
                "ConnectError: vMLX unreachable",
            ),
        ],
    )
    calls = _track_run_topic(monkeypatch)

    rc = cli.cmd_run_topic(_topic_args())
    captured = capsys.readouterr()

    assert rc == 3
    assert calls == []
    assert "preflight: vMLX is not ready" in captured.err
    assert "[FAIL] chat:relevance_filter" in captured.err


def test_run_topic_aborts_with_exit_3_when_routing_invalid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _broken_load() -> RoutingConfig:
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(cli, "load_routing", _broken_load)
    calls = _track_run_topic(monkeypatch)

    rc = cli.cmd_run_topic(_topic_args())
    captured = capsys.readouterr()

    assert rc == 3
    assert calls == []
    assert "preflight: routing config" in captured.err
    assert "FileNotFoundError" in captured.err


def test_run_topic_rejects_empty_query_with_exit_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Empty / whitespace-only --query is a caller error, not a
    preflight failure. Exit 2 matches the existing convention for
    ``ValueError``-shaped misuse (bad window spec on run daily).
    """
    calls = _track_run_topic(monkeypatch)

    rc = cli.cmd_run_topic(_topic_args(query="   "))
    captured = capsys.readouterr()

    assert rc == 2
    assert calls == []
    assert "--query must not be empty" in captured.err


def test_main_wires_run_topic_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the argparse + dispatch wiring end-to-end so a regression
    in the subparser would surface as a failing test rather than as a
    silent ``main()`` falling through to the help text + exit 2.
    """
    _patch_probes(
        monkeypatch,
        [
            ProbeResult("health", True, 5, "status=ok"),
            ProbeResult("models", True, 10, "1 model(s): m"),
            ProbeResult("chat:relevance_filter", True, 100, "m → 'PONG'"),
        ],
    )
    calls = _track_run_topic(monkeypatch)

    rc = cli.main(["run", "topic", "--query", "Anthropic", "--window-days", "14"])
    assert rc == 0
    assert calls == [("Anthropic", 14)]
