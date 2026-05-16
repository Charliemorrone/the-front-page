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

    args = argparse.Namespace(cmd="run", run_type="daily", window="24h", dry_run=False)
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

    args = argparse.Namespace(cmd="run", run_type="daily", window="24h", dry_run=False)
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

    args = argparse.Namespace(cmd="run", run_type="daily", window="24h", dry_run=False)
    rc = cli.cmd_run_daily(args)
    captured = capsys.readouterr()

    assert rc == 3
    assert calls == []
    assert "preflight: routing config" in captured.err
    assert "FileNotFoundError" in captured.err


# ── --dry-run mode (Phase 6c.1) ──────────────────────────────────────────────


def _routing_with_gemini() -> RoutingConfig:
    """Production-shaped routing: vmlx + gemini_cli + final_compose fallback.

    Mirrors the shipped ``config/model-routing.yaml`` shape so the
    dry-run report reflects what an operator would see in real use.
    """
    return RoutingConfig.model_validate(
        {
            "providers": {
                "vmlx": {"base_url": "http://127.0.0.1:8080/v1"},
                "gemini_cli": {
                    "script_path": "/fake/gemini",
                    "executable_path": "/fake/node",
                },
            },
            "stages": {
                "relevance_filter": {
                    "provider": "vmlx",
                    "model": "stub-filter",
                    "timeout_seconds": 60,
                    "batch_size": 12,
                },
                "cluster_summary": {
                    "provider": "vmlx",
                    "model": "stub-summary",
                    "timeout_seconds": 60,
                },
                "final_compose": {
                    "provider": "gemini_cli",
                    "model": "gemini-3-pro-preview",
                    "timeout_seconds": 300,
                    "fallback": {"provider": "vmlx", "model": "stub-fallback"},
                },
            },
        }
    )


def _setup_dry_run_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,  # noqa: ANN001 — pytest tmp_path fixture
    *,
    routing: RoutingConfig | None = None,
    probes: list[ProbeResult] | None = None,
    gemini_files_exist: bool = True,
    plan_raises: Exception | None = None,
):
    """Wire a dry-run-test-friendly environment.

    - Routing returns either the provided config or ``_routing_with_gemini()``.
    - vMLX probes return the supplied list (default: all-pass).
    - ``build_source_plan`` returns an empty-plan stub unless a raise
      is requested.
    - Gemini CLI script/executable files are created in ``tmp_path``
      unless the test wants them missing.
    - ``launchagent.log_paths`` + ``launchagent.plist_path`` are
      redirected under ``tmp_path``.
    - ``run_daily`` is stubbed (the dry-run path must never invoke it).
    """
    from pathlib import Path

    from clawfeed_intel import cli, db, doctor, launchagent
    from clawfeed_intel.sources import ProfileConfig, SourcePlan

    cfg = routing if routing is not None else _routing_with_gemini()

    monkeypatch.setattr(cli, "load_routing", lambda: cfg)

    async def fake_probes(*_a, **_kw):
        return list(
            probes
            or [
                ProbeResult("health", True, 5, "status=ok"),
                ProbeResult("models", True, 10, "1 model"),
                ProbeResult("chat:relevance_filter", True, 100, "stub-filter → 'PONG'"),
            ]
        )

    monkeypatch.setattr(doctor, "run_doctor_probes", fake_probes)

    def fake_plan(_conn):
        if plan_raises is not None:
            raise plan_raises
        return SourcePlan(
            profile=ProfileConfig(),
            categories=[],
            dynamic_search=[],
            warnings=[],
        )

    monkeypatch.setattr(cli, "build_source_plan", fake_plan)

    # Stub the DB connection so build_source_plan's signature is satisfied
    # without needing a real SQLite file. ``closing`` accepts any object
    # with a ``close()`` method.
    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(db, "connect", lambda *a, **kw: _FakeConn())

    # Gemini CLI binaries — create real files in tmp_path so
    # ``Path(...).is_file()`` returns True; the routing config points at
    # them via monkeypatched provider attributes.
    if gemini_files_exist and cfg.providers.gemini_cli is not None:
        gem_dir = tmp_path / "gemini-binaries"
        gem_dir.mkdir(exist_ok=True)
        fake_node = gem_dir / "node"
        fake_script = gem_dir / "gemini"
        fake_node.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_script.write_text("#!/bin/sh\n", encoding="utf-8")
        cfg.providers.gemini_cli.__dict__["executable_path"] = str(fake_node)
        cfg.providers.gemini_cli.__dict__["script_path"] = str(fake_script)

    # Redirect launchagent's filesystem touchpoints into tmp_path.
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir(exist_ok=True)
    monkeypatch.setattr(
        launchagent,
        "Path",
        type(
            "_P",
            (),
            {
                "home": staticmethod(lambda: fake_home),
                "__call__": staticmethod(Path),
            },
        )(),
    )
    monkeypatch.setattr(launchagent, "REPO_ROOT", fake_repo)

    # Sentinel run_daily so the dry-run path's failure to invoke is asserted.
    invoked: list[str] = []

    def stub_run_daily(window):
        invoked.append(window)
        return 999

    monkeypatch.setattr(cli, "run_daily", stub_run_daily)
    return invoked


def _dry_run_args() -> argparse.Namespace:
    return argparse.Namespace(cmd="run", run_type="daily", window="24h", dry_run=True)


def test_dry_run_short_circuits_run_daily(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--dry-run`` must never invoke the real ``run_daily`` —
    that's the whole point of the flag.
    """
    invoked = _setup_dry_run_environment(monkeypatch, tmp_path)
    rc = cli.cmd_run_daily(_dry_run_args())
    captured = capsys.readouterr()

    assert rc == 0
    assert invoked == []  # run_daily NOT called
    assert "preflight passed" in captured.out


def test_dry_run_reports_all_sections(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dry-run output covers routing, vmlx, gemini_cli, source plan,
    log dir, and launchd — six discoverable sections an operator can
    scan top-to-bottom.
    """
    _setup_dry_run_environment(monkeypatch, tmp_path)
    cli.cmd_run_daily(_dry_run_args())
    out = capsys.readouterr().out
    for section in (
        "[routing]",
        "[vmlx]",
        "[gemini_cli]",
        "[source_plan]",
        "[log_dir]",
        "[launchd]",
    ):
        assert section in out


def test_dry_run_reports_final_compose_routing_with_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """final_compose stage routing is named explicitly so an operator
    knows whether the run will use Gemini (Tier 1) or the fallback.
    """
    _setup_dry_run_environment(monkeypatch, tmp_path)
    cli.cmd_run_daily(_dry_run_args())
    out = capsys.readouterr().out
    assert "final_compose: gemini_cli/gemini-3-pro-preview" in out
    assert "fallback to vmlx/stub-fallback" in out


def test_dry_run_exits_3_when_routing_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _broken() -> RoutingConfig:
        raise FileNotFoundError("/no/such/routing.yaml")

    monkeypatch.setattr(cli, "load_routing", _broken)
    rc = cli.cmd_run_daily(_dry_run_args())
    err = capsys.readouterr().err

    assert rc == 3
    assert "config failed to load" in err
    assert "FileNotFoundError" in err


def test_dry_run_exits_3_when_vmlx_probe_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing vMLX probe aborts the dry-run before reaching the
    Gemini / source-plan / launchd sections.
    """
    _setup_dry_run_environment(
        monkeypatch,
        tmp_path,
        probes=[
            ProbeResult("health", False, 5000, "ConnectError"),
            ProbeResult("models", True, 10, "1 model"),
            ProbeResult("chat:relevance_filter", True, 100, "ok"),
        ],
    )
    rc = cli.cmd_run_daily(_dry_run_args())
    out = capsys.readouterr()
    assert rc == 3
    assert "vMLX preflight failed" in out.err
    # Sections AFTER vmlx must not have printed.
    assert "[gemini_cli]" not in out.out


def test_dry_run_exits_3_when_gemini_binaries_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``gemini_cli`` provider configured but executable file missing →
    exit 3. Without this, the first 06:15 launchd fire would discover
    the broken config the hard way.
    """
    _setup_dry_run_environment(monkeypatch, tmp_path, gemini_files_exist=False)
    rc = cli.cmd_run_daily(_dry_run_args())
    err = capsys.readouterr().err
    assert rc == 3
    assert "binaries missing" in err


def test_dry_run_warns_without_failing_when_gemini_provider_unconfigured(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A vmlx-only deployment (no gemini_cli provider in YAML) is a
    valid but unusual config after the 2026-05-15 amendment. Surface
    it as a warning, not an error.
    """
    vmlx_only = RoutingConfig.model_validate(
        {
            "providers": {"vmlx": {"base_url": "http://127.0.0.1:8080/v1"}},
            "stages": {
                "relevance_filter": {
                    "provider": "vmlx",
                    "model": "m",
                    "timeout_seconds": 60,
                    "batch_size": 12,
                },
                "final_compose": {
                    "provider": "vmlx",
                    "model": "compose-vmlx",
                    "timeout_seconds": 300,
                },
            },
        }
    )
    _setup_dry_run_environment(monkeypatch, tmp_path, routing=vmlx_only)
    rc = cli.cmd_run_daily(_dry_run_args())
    out = capsys.readouterr().out
    assert rc == 0
    assert "[warn] provider not declared" in out


def test_dry_run_exits_3_when_source_plan_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup_dry_run_environment(
        monkeypatch,
        tmp_path,
        plan_raises=ValueError("intel-sources.yaml: missing required key 'categories'"),
    )
    rc = cli.cmd_run_daily(_dry_run_args())
    err = capsys.readouterr().err
    assert rc == 3
    assert "resolver raised" in err
    assert "ValueError" in err


def test_dry_run_reports_launchd_not_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the LaunchAgent isn't installed yet, the dry-run prints an
    [info] line with the install command — the operator's next action.
    """
    _setup_dry_run_environment(monkeypatch, tmp_path)
    cli.cmd_run_daily(_dry_run_args())
    out = capsys.readouterr().out
    assert "[info] not installed" in out
    assert "cron install --install" in out


def test_dry_run_reports_launchd_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the LaunchAgent IS installed, the dry-run reports the path
    with an [ ok ] marker.
    """
    from clawfeed_intel import launchagent

    _setup_dry_run_environment(monkeypatch, tmp_path)
    target = launchagent.plist_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("body", encoding="utf-8")
    cli.cmd_run_daily(_dry_run_args())
    out = capsys.readouterr().out
    assert f"[ ok ] installed at {target}" in out
