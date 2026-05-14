"""Tests for the ``clawfeed-intel run daily`` preflight contract.

The CLI runs the same vMLX probes ``doctor`` runs before kicking off the
real pipeline. If any probe fails, the command must abort *before*
``run_daily`` is called — fetch/cluster/filter must not start, no
``intel_runs`` row must land, and the exit code must be ``3`` so cron /
OpenClaw can tell preflight failure apart from the existing ``2`` used
for a bad window spec.

Tests stub the probe layer and the orchestrator entrypoint so no live
vMLX or live fetcher HTTP runs under CI — same pattern
``test_cli_doctor.py`` uses.
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


def _track_run_daily(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace ``cli.run_daily`` with a stub that records its call argument.

    Returning a sentinel digest id lets the all-pass test verify the
    handler's success path; recording the argument lets every failure
    test assert the real orchestrator was *not* invoked.
    """
    calls: list[str] = []

    def stub(window: str) -> int:
        calls.append(window)
        return 4242

    monkeypatch.setattr(cli, "run_daily", stub)
    return calls


def test_run_daily_proceeds_when_all_probes_pass(
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
    calls = _track_run_daily(monkeypatch)

    args = argparse.Namespace(cmd="run", run_type="daily", window="24h")
    rc = cli.cmd_run_daily(args)
    captured = capsys.readouterr()

    assert rc == 0
    assert calls == ["24h"]
    assert "published digest 4242" in captured.out
    # Stderr must stay empty on the happy path — preflight prints nothing
    # when every probe is green so cron logs aren't noisy on success.
    assert captured.err == ""


def test_run_daily_aborts_with_exit_3_when_a_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
    temp_db,  # noqa: ANN001 — pytest fixture (Path)
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failed preflight: exit 3, no intel_runs row, doctor-style stderr."""
    from clawfeed_intel import db as worker_db

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
    calls = _track_run_daily(monkeypatch)
    # Point both DB bindings at the temp DB so any (hypothetical) accidental
    # write would land where we can inspect it instead of polluting the
    # real ``data/digest.db``.
    monkeypatch.setattr(cli, "DB_PATH", temp_db)
    monkeypatch.setattr(worker_db, "DB_PATH", temp_db)

    args = argparse.Namespace(cmd="run", run_type="daily", window="24h")
    rc = cli.cmd_run_daily(args)
    captured = capsys.readouterr()

    assert rc == 3
    assert calls == []  # orchestrator never invoked
    # Stderr carries both the operator-facing reason and the same probe-line
    # format ``cmd_doctor`` prints, so the view is identical.
    assert "preflight: vMLX is not ready" in captured.err
    assert "[FAIL] chat:relevance_filter" in captured.err
    assert "[ ok ] health" in captured.err
    # And no run row should exist in the DB.
    import sqlite3

    with sqlite3.connect(temp_db) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM intel_runs").fetchone()
    assert row[0] == 0


def test_run_daily_aborts_with_exit_3_when_routing_config_invalid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _broken_load() -> RoutingConfig:
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(cli, "load_routing", _broken_load)
    calls = _track_run_daily(monkeypatch)

    args = argparse.Namespace(cmd="run", run_type="daily", window="24h")
    rc = cli.cmd_run_daily(args)
    captured = capsys.readouterr()

    assert rc == 3
    assert calls == []
    assert "preflight: routing config" in captured.err
    assert "FileNotFoundError" in captured.err
