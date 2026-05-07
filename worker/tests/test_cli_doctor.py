"""Tests for the ``clawfeed-intel doctor`` exit-code contract.

The command's exit code is load-bearing: cron uses it to decide whether
to kick off the daily run. These tests pin that contract by mocking the
probe layer and exercising :func:`cli.cmd_doctor` directly.
"""

from __future__ import annotations

import argparse
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from clawfeed_intel import cli, db, doctor
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


def _patch_probes(
    monkeypatch: pytest.MonkeyPatch,
    results: list[ProbeResult] | Callable[[], Awaitable[list[ProbeResult]]],
) -> None:
    """Stub out network-dependent helpers inside the doctor module."""

    async def fake_probes(*_args: object, **_kwargs: object) -> list[ProbeResult]:
        if callable(results):
            return await results()  # type: ignore[no-any-return]
        return list(results)

    monkeypatch.setattr(doctor, "run_doctor_probes", fake_probes)
    monkeypatch.setattr(cli, "load_routing", _routing)


def _isolate_db(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
    """Point both ``cli.DB_PATH`` and ``db.DB_PATH`` at *db_path*.

    ``cli.DB_PATH`` is the binding the doctor command's existence check
    consults; ``db.DB_PATH`` is what :func:`db.connect` reads when no
    explicit path is supplied. Both must move together.
    """
    monkeypatch.setattr(cli, "DB_PATH", db_path)
    monkeypatch.setattr(db, "DB_PATH", db_path)


def test_doctor_exits_zero_when_all_probes_pass(
    monkeypatch: pytest.MonkeyPatch, temp_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _isolate_db(monkeypatch, temp_db)
    _patch_probes(
        monkeypatch,
        [
            ProbeResult("health", True, 5, "status=ok"),
            ProbeResult("models", True, 10, "1 model(s): m"),
            ProbeResult("chat:relevance_filter", True, 100, "m → 'PONG'"),
        ],
    )

    args = argparse.Namespace(cmd="doctor")
    rc = cli.cmd_doctor(args)
    out = capsys.readouterr().out

    assert rc == 0
    # All probes appear in the output as "[ ok ]" lines.
    assert "[ ok ] health" in out
    assert "[ ok ] models" in out
    assert "[ ok ] chat:relevance_filter" in out


def test_doctor_exits_nonzero_on_any_probe_failure(
    monkeypatch: pytest.MonkeyPatch, temp_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _isolate_db(monkeypatch, temp_db)
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

    args = argparse.Namespace(cmd="doctor")
    rc = cli.cmd_doctor(args)
    out = capsys.readouterr().out

    assert rc == 1
    assert "[FAIL] chat:relevance_filter" in out


def test_doctor_exits_nonzero_when_routing_config_invalid(
    monkeypatch: pytest.MonkeyPatch, temp_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _isolate_db(monkeypatch, temp_db)

    def _broken_load() -> RoutingConfig:
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(cli, "load_routing", _broken_load)

    args = argparse.Namespace(cmd="doctor")
    rc = cli.cmd_doctor(args)
    out = capsys.readouterr().out

    assert rc == 1
    assert "FAIL" in out
    assert "FileNotFoundError" in out


def test_doctor_exits_nonzero_when_probes_raise_unexpectedly(
    monkeypatch: pytest.MonkeyPatch, temp_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bug in the probes shouldn't result in a misleading exit-zero."""
    _isolate_db(monkeypatch, temp_db)
    monkeypatch.setattr(cli, "load_routing", _routing)

    async def boom(*_args: object, **_kwargs: object) -> list[ProbeResult]:
        raise RuntimeError("unexpected internal failure")

    monkeypatch.setattr(doctor, "run_doctor_probes", boom)

    args = argparse.Namespace(cmd="doctor")
    rc = cli.cmd_doctor(args)
    out = capsys.readouterr().out

    assert rc == 1
    assert "FAIL" in out
    assert "RuntimeError" in out


def test_doctor_exits_nonzero_when_db_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fresh checkout without migrations applied → DB doesn't exist → fail."""
    _isolate_db(monkeypatch, tmp_path / "missing.db")
    _patch_probes(
        monkeypatch,
        [
            ProbeResult("health", True, 5, "status=ok"),
            ProbeResult("models", True, 10, "1 model(s): m"),
            ProbeResult("chat:relevance_filter", True, 100, "ok"),
        ],
    )

    args = argparse.Namespace(cmd="doctor")
    rc = cli.cmd_doctor(args)

    assert rc == 1


def test_doctor_exits_nonzero_when_db_missing_required_tables(
    monkeypatch: pytest.MonkeyPatch, temp_db: Path
) -> None:
    """Migrated DB but with a required table dropped → also fail."""
    import sqlite3

    with sqlite3.connect(temp_db) as conn:
        conn.execute("DROP TABLE llm_calls")

    _isolate_db(monkeypatch, temp_db)
    _patch_probes(
        monkeypatch,
        [
            ProbeResult("health", True, 5, "status=ok"),
            ProbeResult("models", True, 10, "1 model(s): m"),
            ProbeResult("chat:relevance_filter", True, 100, "ok"),
        ],
    )

    args = argparse.Namespace(cmd="doctor")
    rc = cli.cmd_doctor(args)

    assert rc == 1
