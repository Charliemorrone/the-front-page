"""Tests for the launchd LaunchAgent module (Phase 6b).

The pure layer (``render_plist``, ``plist_path``, ``log_paths``,
``gui_domain_*``) is fixture-testable without touching launchctl or
the operator's real ``~/Library/LaunchAgents/``.

The subprocess wrappers (``bootstrap_agent`` / ``bootout_agent`` /
``print_agent``) are tested by stubbing ``launchagent._default_runner``
so no real launchctl invocation happens during ``uv run pytest``.
The CLI integration tests redirect ``plist_path`` to a tmp dir via
the ``home`` parameter — the operator's real LaunchAgents directory
is never touched.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from xml.etree import ElementTree

import pytest

from clawfeed_intel import cli, launchagent


# ── Helpers ──────────────────────────────────────────────────────────────────


class _FakeRunner:
    """Recording subprocess stub that returns a controllable result.

    ``returncode`` + ``stdout`` + ``stderr`` are what each subsequent
    call resolves to; pop the queue between calls if a test needs
    different responses across multiple invocations. ``calls`` is the
    captured argv list — used by tests to assert what launchctl was
    invoked with.
    """

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.calls: list[list[str]] = []
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    def __call__(self, args, **kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        return subprocess.CompletedProcess(
            args=args,
            returncode=self._returncode,
            stdout=self._stdout,
            stderr=self._stderr,
        )


def _patch_subprocess(monkeypatch: pytest.MonkeyPatch, runner: _FakeRunner) -> None:
    """Redirect every launchctl invocation through the recording stub."""
    monkeypatch.setattr(launchagent, "_default_runner", runner)


def _patch_home(monkeypatch: pytest.MonkeyPatch, fake_home: Path) -> None:
    """Redirect plist_path()'s ``Path.home()`` lookup to a tmp dir."""
    monkeypatch.setattr(launchagent, "Path", _HomePatchPath(fake_home))


class _HomePatchPath:
    """Tiny shim that returns ``fake_home`` for ``Path.home()`` and
    delegates everything else to the real :class:`pathlib.Path`.

    We can't monkeypatch ``Path.home`` directly (immutable on stdlib
    classes); the cleanest swap is replacing the ``Path`` name inside
    the module.
    """

    def __init__(self, fake_home: Path) -> None:
        self._fake_home = fake_home

    def home(self) -> Path:
        return self._fake_home

    def __call__(self, *args, **kwargs) -> Path:
        return Path(*args, **kwargs)


# ── Pure: render_plist ───────────────────────────────────────────────────────


def test_render_plist_is_valid_xml() -> None:
    """Output must parse cleanly — launchd rejects malformed plists."""
    text = launchagent.render_plist()
    root = ElementTree.fromstring(text)
    assert root.tag == "plist"


def test_render_plist_contains_label() -> None:
    text = launchagent.render_plist()
    assert "<key>Label</key>" in text
    assert f"<string>{launchagent.LABEL}</string>" in text


def test_render_plist_default_schedule_is_06_15() -> None:
    """Architecture-doc Phase 6 default — 06:15 local. Pin it."""
    text = launchagent.render_plist()
    assert "<key>Hour</key>\n        <integer>6</integer>" in text
    assert "<key>Minute</key>\n        <integer>15</integer>" in text


def test_render_plist_custom_schedule(tmp_path: Path) -> None:
    text = launchagent.render_plist(hour=23, minute=45)
    assert "<integer>23</integer>" in text
    assert "<integer>45</integer>" in text


def test_render_plist_rejects_invalid_hour() -> None:
    with pytest.raises(ValueError, match="hour must be"):
        launchagent.render_plist(hour=24)


def test_render_plist_rejects_invalid_minute() -> None:
    with pytest.raises(ValueError, match="minute must be"):
        launchagent.render_plist(minute=-1)


def test_render_plist_uses_provided_repo_root(tmp_path: Path) -> None:
    """``WorkingDirectory`` should resolve to the supplied root."""
    text = launchagent.render_plist(repo_root=tmp_path)
    expected = tmp_path.resolve()
    assert f"<string>{expected}</string>" in text


def test_render_plist_is_deterministic() -> None:
    """Same inputs → byte-identical output (load-bearing for idempotent install)."""
    a = launchagent.render_plist(hour=8, minute=30)
    b = launchagent.render_plist(hour=8, minute=30)
    assert a == b


def test_render_plist_pins_uv_path_when_supplied(tmp_path: Path) -> None:
    """An operator can pin ``uv_path`` to override PATH-resolved default."""
    text = launchagent.render_plist(uv_path="/custom/uv")
    assert "<string>/custom/uv</string>" in text


def test_render_plist_program_args_invoke_daily(tmp_path: Path) -> None:
    """The job runs `clawfeed-intel run daily --window 24h`. Pin it."""
    text = launchagent.render_plist()
    for token in ("clawfeed-intel", "run", "daily", "--window", "24h"):
        assert f"<string>{token}</string>" in text


def test_render_plist_run_at_load_is_false() -> None:
    """Loading the agent must not fire the daily run immediately —
    that would re-run the brief mid-day every time we install.
    """
    text = launchagent.render_plist()
    assert "<key>RunAtLoad</key>\n    <false/>" in text


def test_render_plist_path_env_includes_homebrew() -> None:
    """Without /opt/homebrew/bin on PATH, `uv` and the working `node`
    used by the Gemini CLI provider aren't findable at job-fire time.
    """
    text = launchagent.render_plist()
    assert "/opt/homebrew/bin" in text


def test_render_plist_log_paths_point_under_repo(tmp_path: Path) -> None:
    text = launchagent.render_plist(repo_root=tmp_path)
    expected_logs = (tmp_path / "data" / "logs").resolve()
    assert f"<string>{expected_logs}/daily-brief.out.log</string>" in text
    assert f"<string>{expected_logs}/daily-brief.err.log</string>" in text


# ── Pure: paths + gui domain ─────────────────────────────────────────────────


def test_plist_path_lives_under_library_launchagents(tmp_path: Path) -> None:
    out = launchagent.plist_path(tmp_path)
    assert out == tmp_path / "Library" / "LaunchAgents" / f"{launchagent.LABEL}.plist"


def test_log_paths_under_data_logs(tmp_path: Path) -> None:
    paths = launchagent.log_paths(tmp_path)
    assert paths.out.name == "daily-brief.out.log"
    assert paths.err.name == "daily-brief.err.log"
    assert paths.directory == (tmp_path / "data" / "logs").resolve()


def test_gui_domain_target_format() -> None:
    assert launchagent.gui_domain_target(uid=502) == "gui/502"


def test_gui_domain_label_format() -> None:
    assert launchagent.gui_domain_label(uid=502) == f"gui/502/{launchagent.LABEL}"


def test_is_installed_false_when_plist_missing(tmp_path: Path) -> None:
    assert launchagent.is_installed(tmp_path) is False


def test_is_installed_true_when_plist_present(tmp_path: Path) -> None:
    target = launchagent.plist_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("anything", encoding="utf-8")
    assert launchagent.is_installed(tmp_path) is True


# ── write_plist_atomic ───────────────────────────────────────────────────────


def test_write_plist_atomic_creates_directory(tmp_path: Path) -> None:
    """The ``LaunchAgents`` parent dir may not exist on a fresh user."""
    written = launchagent.write_plist_atomic("plist body", home=tmp_path)
    assert written.read_text() == "plist body"
    assert written.parent.is_dir()


def test_write_plist_atomic_overwrites_existing(tmp_path: Path) -> None:
    launchagent.write_plist_atomic("first", home=tmp_path)
    launchagent.write_plist_atomic("second", home=tmp_path)
    assert launchagent.plist_path(tmp_path).read_text() == "second"


def test_write_plist_atomic_leaves_no_tmp_file(tmp_path: Path) -> None:
    launchagent.write_plist_atomic("body", home=tmp_path)
    parent = launchagent.plist_path(tmp_path).parent
    leftover = [p for p in parent.iterdir() if p.suffix.endswith(".tmp")]
    assert leftover == []


def test_ensure_log_dir_creates_data_logs(tmp_path: Path) -> None:
    target = launchagent.ensure_log_dir(tmp_path)
    assert target.is_dir()
    assert target == (tmp_path / "data" / "logs").resolve()


# ── Subprocess wrappers ──────────────────────────────────────────────────────


def test_bootstrap_invokes_launchctl_with_gui_domain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _FakeRunner(returncode=0)
    _patch_subprocess(monkeypatch, runner)
    plist = tmp_path / "x.plist"
    launchagent.bootstrap_agent(plist, uid=502)
    assert runner.calls == [["launchctl", "bootstrap", "gui/502", str(plist)]]


def test_bootout_invokes_launchctl_with_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _FakeRunner(returncode=0)
    _patch_subprocess(monkeypatch, runner)
    launchagent.bootout_agent(uid=502)
    assert runner.calls == [["launchctl", "bootout", f"gui/502/{launchagent.LABEL}"]]


def test_print_invokes_launchctl_print(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _FakeRunner(returncode=0, stdout="state = running")
    _patch_subprocess(monkeypatch, runner)
    result = launchagent.print_agent(uid=502)
    assert runner.calls == [["launchctl", "print", f"gui/502/{launchagent.LABEL}"]]
    assert "state = running" in result.stdout


def test_bootstrap_returns_completed_process_for_inspection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI layer reads ``returncode`` + ``stderr`` to decide whether
    to surface the failure to the operator. Don't raise on non-zero.
    """
    runner = _FakeRunner(returncode=5, stderr="domain locked")
    _patch_subprocess(monkeypatch, runner)
    result = launchagent.bootstrap_agent(tmp_path / "x.plist", uid=502)
    assert result.returncode == 5
    assert "domain locked" in result.stderr


# ── CLI: cron install ────────────────────────────────────────────────────────


def _cron_install_args(
    *, install: bool = False, hour: int = 6, minute: int = 15
) -> argparse.Namespace:
    return argparse.Namespace(
        cmd="cron",
        cron_action="install",
        install=install,
        hour=hour,
        minute=minute,
    )


def _redirect_cli_to_tmp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner: _FakeRunner | None = None,
) -> _FakeRunner:
    """Wire up tmp_path as the operator's fake home + stub launchctl.

    Returns the recording runner so tests can assert what launchctl
    calls happened. Each test gets a fresh runner unless one is
    supplied.
    """
    fake_runner = runner or _FakeRunner(returncode=0)
    _patch_subprocess(monkeypatch, fake_runner)
    monkeypatch.setattr(launchagent, "Path", _HomePatchPath(tmp_path))
    return fake_runner


def test_cron_install_dry_run_prints_preview_and_does_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without ``--install`` the CLI prints the plist + exit 0; no
    filesystem mutation, no launchctl invocation.
    """
    runner = _redirect_cli_to_tmp(monkeypatch, tmp_path)
    exit_code = cli.cmd_cron_install(_cron_install_args(install=False))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "would write LaunchAgent" in captured.out
    assert "would invoke: launchctl bootstrap" in captured.out
    assert "--- plist preview ---" in captured.out
    assert "re-run with --install" in captured.out
    assert runner.calls == []
    assert not launchagent.plist_path(tmp_path).exists()


def test_cron_install_writes_plist_and_invokes_launchctl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--install`` writes the plist atomically + calls bootstrap.

    The bootout-first call is best-effort cleanup — bootstrap of an
    already-loaded service errors on macOS, so we always boot out
    first. Test asserts both calls in order.
    """
    runner = _redirect_cli_to_tmp(monkeypatch, tmp_path)
    exit_code = cli.cmd_cron_install(_cron_install_args(install=True))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert launchagent.plist_path(tmp_path).is_file()
    assert "wrote" in captured.out
    assert "loaded" in captured.out

    # Two launchctl calls: bootout first, then bootstrap.
    assert len(runner.calls) == 2
    assert runner.calls[0][:2] == ["launchctl", "bootout"]
    assert runner.calls[1][:2] == ["launchctl", "bootstrap"]


def test_cron_install_idempotent_writes_byte_identical_plist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running install with the same args produces the same plist
    bytes. Load-bearing: an operator who runs install twice shouldn't
    end up with a drifting config.
    """
    _redirect_cli_to_tmp(monkeypatch, tmp_path)
    cli.cmd_cron_install(_cron_install_args(install=True))
    first = launchagent.plist_path(tmp_path).read_text()
    cli.cmd_cron_install(_cron_install_args(install=True))
    second = launchagent.plist_path(tmp_path).read_text()
    assert first == second


def test_cron_install_surfaces_launchctl_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When bootstrap fails (e.g. invalid plist, domain locked) the
    CLI exits non-zero and prints stderr. Plist is still on disk —
    the operator may want to inspect it.
    """

    # The first runner call (bootout) is the implicit cleanup; the
    # second (bootstrap) is what we want to fail.
    class _SequencedRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []
            self._results = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(
                    args=[], returncode=2, stdout="", stderr="launchctl: input/output error"
                ),
            ]

        def __call__(self, args, **kwargs) -> subprocess.CompletedProcess[str]:
            self.calls.append(list(args))
            return self._results.pop(0)

    runner = _SequencedRunner()
    monkeypatch.setattr(launchagent, "_default_runner", runner)
    monkeypatch.setattr(launchagent, "Path", _HomePatchPath(tmp_path))

    exit_code = cli.cmd_cron_install(_cron_install_args(install=True))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "launchctl bootstrap failed" in captured.err
    assert "input/output error" in captured.err
    # The plist remains on disk for the operator to inspect.
    assert launchagent.plist_path(tmp_path).is_file()


def test_cron_install_respects_hour_minute_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _redirect_cli_to_tmp(monkeypatch, tmp_path)
    cli.cmd_cron_install(_cron_install_args(install=True, hour=20, minute=0))
    text = launchagent.plist_path(tmp_path).read_text()
    assert "<integer>20</integer>" in text
    assert "<integer>0</integer>" in text


# ── CLI: cron uninstall ──────────────────────────────────────────────────────


def _cron_uninstall_args(*, remove: bool = False) -> argparse.Namespace:
    return argparse.Namespace(cmd="cron", cron_action="uninstall", remove=remove)


def test_cron_uninstall_dry_run_when_installed_prints_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _redirect_cli_to_tmp(monkeypatch, tmp_path)
    # Pre-install so the dry-run reports the present plist.
    launchagent.plist_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    launchagent.plist_path(tmp_path).write_text("body", encoding="utf-8")

    exit_code = cli.cmd_cron_uninstall(_cron_uninstall_args(remove=False))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "would invoke" in captured.out
    assert "would remove" in captured.out
    assert runner.calls == []
    assert launchagent.plist_path(tmp_path).is_file()


def test_cron_uninstall_dry_run_when_not_installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _redirect_cli_to_tmp(monkeypatch, tmp_path)
    exit_code = cli.cmd_cron_uninstall(_cron_uninstall_args(remove=False))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "no LaunchAgent installed" in captured.out


def test_cron_uninstall_remove_actually_removes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _redirect_cli_to_tmp(monkeypatch, tmp_path)
    target = launchagent.plist_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("body", encoding="utf-8")

    exit_code = cli.cmd_cron_uninstall(_cron_uninstall_args(remove=True))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert not target.exists()
    assert "removed" in captured.out
    assert runner.calls and runner.calls[0][:2] == ["launchctl", "bootout"]


def test_cron_uninstall_idempotent_when_not_installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--remove`` on a clean system is a no-op exit-0, not an error."""
    runner = _redirect_cli_to_tmp(monkeypatch, tmp_path)
    exit_code = cli.cmd_cron_uninstall(_cron_uninstall_args(remove=True))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "no LaunchAgent installed" in captured.out
    assert runner.calls == []


# ── CLI: cron status ─────────────────────────────────────────────────────────


def _cron_status_args() -> argparse.Namespace:
    return argparse.Namespace(cmd="cron", cron_action="status")


def test_cron_status_reports_not_installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _redirect_cli_to_tmp(monkeypatch, tmp_path)
    exit_code = cli.cmd_cron_status(_cron_status_args())
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "not installed" in captured.out


def test_cron_status_reports_installed_with_launchctl_dump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _redirect_cli_to_tmp(
        monkeypatch,
        tmp_path,
        _FakeRunner(returncode=0, stdout="state = running\nnext firing = 2026-05-16 06:15:00\n"),
    )
    launchagent.plist_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    launchagent.plist_path(tmp_path).write_text("body", encoding="utf-8")

    exit_code = cli.cmd_cron_status(_cron_status_args())
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "installed at" in captured.out
    assert "state = running" in captured.out
    assert "next firing" in captured.out
    assert runner.calls and runner.calls[0][:2] == ["launchctl", "print"]


def test_cron_status_plist_present_but_launchctl_unloaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``launchctl print`` returns non-zero when the agent isn't loaded.

    Plist file presence + launchctl-not-loaded means the operator
    wrote the file but never ran ``cron install --install``; the
    status output guides them to that next step.
    """
    _redirect_cli_to_tmp(
        monkeypatch,
        tmp_path,
        _FakeRunner(returncode=113, stderr="Could not find service"),
    )
    launchagent.plist_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    launchagent.plist_path(tmp_path).write_text("body", encoding="utf-8")

    exit_code = cli.cmd_cron_status(_cron_status_args())
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "may not be loaded" in captured.out
    assert "Could not find service" in captured.out


# ── Main wiring ──────────────────────────────────────────────────────────────


def test_main_dispatches_cron_install_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``cli.main(["cron", "install"])`` reaches ``cmd_cron_install``."""
    _redirect_cli_to_tmp(monkeypatch, tmp_path)
    exit_code = cli.main(["cron", "install"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "would write" in captured.out


def test_main_dispatches_cron_uninstall_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _redirect_cli_to_tmp(monkeypatch, tmp_path)
    exit_code = cli.main(["cron", "uninstall"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "no LaunchAgent installed" in captured.out


def test_main_dispatches_cron_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _redirect_cli_to_tmp(monkeypatch, tmp_path)
    exit_code = cli.main(["cron", "status"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "not installed" in captured.out
