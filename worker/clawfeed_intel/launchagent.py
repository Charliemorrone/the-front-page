"""macOS launchd scheduling for the daily brief (Phase 6b).

The architecture doc's Phase 6 calls for a scheduled daily trigger at
06:15 local. The 2026-05-15 amendment specifies macOS ``launchd`` as
the mechanism — OpenClaw cron schedules ``agentTurn`` payloads (LLM
calls), not shell commands, so it was the wrong primitive for firing
a deterministic Python CLI on a clock.

This module is the pure layer + subprocess wrappers. The CLI surface
lives in :mod:`clawfeed_intel.cli` (``clawfeed-intel cron
install/uninstall/status``). Both halves mirror the Phase-6a
``cleanup`` posture: **destructive-default-off** — running ``cron
install`` without ``--install`` prints the rendered plist without
touching disk; ``--install`` writes the file and runs
``launchctl bootstrap``.

Pure helpers (no subprocess, no filesystem mutations):

- :data:`LABEL` — the launchd label; also the plist filename stem.
- :func:`plist_path` — `~/Library/LaunchAgents/<label>.plist`.
- :func:`log_paths` — `data/logs/daily-brief.{out,err}.log`.
- :func:`render_plist` — the deterministic XML body.
- :func:`gui_domain_target` — ``gui/<uid>`` for ``launchctl bootstrap``.

Subprocess wrappers (each takes a ``subprocess.run``-shaped runner so
tests can stub without touching real ``launchctl``):

- :func:`bootstrap_agent` — `launchctl bootstrap gui/<uid> <plist>`.
- :func:`bootout_agent` — `launchctl bootout gui/<uid>/<label>`.
- :func:`print_agent` — `launchctl print gui/<uid>/<label>`.

The launchctl `bootstrap` / `bootout` interface is the modern
replacement for `launchctl load` / `unload`; macOS 10.10+ supports
it and Apple has deprecated the older verbs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .paths import REPO_ROOT

LABEL = "local.clawfeed.daily-brief"
"""launchd job label. Doubles as the plist filename stem.

The ``local.`` prefix is the Apple-recommended convention for
non-org-namespaced personal LaunchAgents — keeps this job out of any
reverse-DNS namespace someone else might claim.
"""


# A `subprocess.run`-shaped callable, narrowed for type-checker hints.
# Tests stub this; production uses :func:`subprocess.run` directly.
SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


def _default_runner(*args, **kwargs) -> subprocess.CompletedProcess[str]:
    """Default subprocess runner — pass-through to :func:`subprocess.run`.

    Wrapping it as a module-level helper (rather than passing
    ``subprocess.run`` directly) keeps the test surface uniform: tests
    monkeypatch ``launchagent._default_runner``, production calls
    everything through it.
    """
    return subprocess.run(*args, **kwargs)


# ── Pure helpers ──────────────────────────────────────────────────────────────


def plist_path(home: Path | None = None) -> Path:
    """Where the LaunchAgent plist lives.

    macOS `~/Library/LaunchAgents/` is the per-user agent directory;
    it loads automatically at login. ``home`` is injectable so tests
    don't write into the operator's real home dir.
    """
    base = home if home is not None else Path.home()
    return base / "Library" / "LaunchAgents" / f"{LABEL}.plist"


@dataclass(frozen=True)
class LogPaths:
    """Absolute paths for the LaunchAgent's stdout + stderr logs."""

    out: Path
    err: Path

    @property
    def directory(self) -> Path:
        """Parent dir that must exist before launchd opens these files."""
        return self.out.parent


def log_paths(repo_root: Path | None = None) -> LogPaths:
    """Resolve the absolute log paths under the repo's ``data/logs/`` dir.

    launchd doesn't create intermediate directories for
    ``StandardOutPath`` / ``StandardErrorPath`` — if the parent dir
    doesn't exist the daily run silently produces no logs (the
    process still runs; you just lose stdout/stderr). The CLI's
    install path ensures the directory before writing the plist.
    """
    root = repo_root if repo_root is not None else REPO_ROOT
    logs_dir = (root / "data" / "logs").resolve()
    return LogPaths(
        out=logs_dir / "daily-brief.out.log",
        err=logs_dir / "daily-brief.err.log",
    )


def gui_domain_target(uid: int | None = None) -> str:
    """`gui/<uid>` — the per-user GUI domain launchctl bootstraps into.

    The `gui/<uid>` domain is the modern per-user GUI session
    container. Loading there means the agent has access to the
    user's keychain / login-session state — important for the daily
    run's Gemini CLI OAuth refresh.
    """
    effective_uid = uid if uid is not None else os.getuid()
    return f"gui/{effective_uid}"


def gui_domain_label(uid: int | None = None) -> str:
    """`gui/<uid>/<label>` — fully-qualified service name for bootout/print."""
    return f"{gui_domain_target(uid)}/{LABEL}"


def render_plist(
    *,
    repo_root: Path | None = None,
    hour: int = 6,
    minute: int = 15,
    uv_path: str | None = None,
) -> str:
    """Build the LaunchAgent plist XML.

    The job fires once per day at ``hour:minute`` local time and runs
    ``uv run clawfeed-intel run daily --window 24h`` with the repo's
    working directory set explicitly. stdout + stderr land in
    ``data/logs/`` so a failed run leaves diagnostic trace.

    ``uv_path`` defaults to the resolved location of ``uv`` on PATH
    (caller responsibility to ensure it's installed) — falling back
    to the homebrew default ``/opt/homebrew/bin/uv`` when ``which``
    returns nothing. Pinning the absolute path avoids depending on
    launchd's minimal PATH environment matching the user's shell PATH.

    Output is deterministic for the same inputs — important for
    idempotent install (re-running ``cron install --install`` should
    produce a byte-identical plist if nothing changed).
    """
    if not 0 <= hour <= 23:
        raise ValueError(f"hour must be 0-23, got {hour}")
    if not 0 <= minute <= 59:
        raise ValueError(f"minute must be 0-59, got {minute}")
    root = (repo_root if repo_root is not None else REPO_ROOT).resolve()
    logs = log_paths(root)
    resolved_uv = uv_path or shutil.which("uv") or "/opt/homebrew/bin/uv"

    # The PATH env var here is the launchd job's environment, not the
    # operator's. It must include the homebrew bin (for `uv`) and the
    # working node we use for the Gemini CLI subprocess (also under
    # /opt/homebrew/bin/). `/usr/bin` + `/bin` cover system tools the
    # worker may invoke.
    path_env = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        f"    <key>Label</key>\n    <string>{LABEL}</string>\n"
        f"    <key>WorkingDirectory</key>\n    <string>{root}</string>\n"
        "    <key>ProgramArguments</key>\n"
        "    <array>\n"
        f"        <string>{resolved_uv}</string>\n"
        "        <string>run</string>\n"
        "        <string>clawfeed-intel</string>\n"
        "        <string>run</string>\n"
        "        <string>daily</string>\n"
        "        <string>--window</string>\n"
        "        <string>24h</string>\n"
        "    </array>\n"
        "    <key>StartCalendarInterval</key>\n"
        "    <dict>\n"
        f"        <key>Hour</key>\n        <integer>{hour}</integer>\n"
        f"        <key>Minute</key>\n        <integer>{minute}</integer>\n"
        "    </dict>\n"
        "    <key>RunAtLoad</key>\n    <false/>\n"
        f"    <key>StandardOutPath</key>\n    <string>{logs.out}</string>\n"
        f"    <key>StandardErrorPath</key>\n    <string>{logs.err}</string>\n"
        "    <key>EnvironmentVariables</key>\n"
        "    <dict>\n"
        f"        <key>PATH</key>\n        <string>{path_env}</string>\n"
        "    </dict>\n"
        "</dict>\n"
        "</plist>\n"
    )


def is_installed(home: Path | None = None) -> bool:
    """True if the plist file exists at the expected path.

    File presence is the cheap check. ``print_agent`` reaches into
    launchctl for live state — use it when the operator asks for
    status rather than every install/uninstall.
    """
    return plist_path(home).is_file()


# ── Subprocess wrappers ──────────────────────────────────────────────────────


def write_plist_atomic(
    text: str,
    *,
    home: Path | None = None,
) -> Path:
    """Write the plist via temp-rename so a partial write can't poison it.

    The Python ``os.replace`` is atomic on the same filesystem. Also
    ensures the LaunchAgents directory exists — it may not on a
    fresh user account.

    Returns the absolute path to the written file.
    """
    target = plist_path(home)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)
    return target


def ensure_log_dir(repo_root: Path | None = None) -> Path:
    """Create the log directory if needed; return its path.

    launchd silently drops stdout/stderr when the parent of
    ``StandardOutPath`` doesn't exist (it doesn't autocreate). Cheap
    insurance run on every install.
    """
    logs = log_paths(repo_root)
    logs.directory.mkdir(parents=True, exist_ok=True)
    return logs.directory


def bootstrap_agent(
    plist_file: Path,
    *,
    uid: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """`launchctl bootstrap gui/<uid> <plist>` — load + enable the job.

    Returns the completed process so callers can inspect ``returncode``
    + ``stderr`` for diagnostics. Doesn't raise on non-zero — the CLI
    layer decides whether to surface the failure to the operator.

    The subprocess invocation goes through the module-level
    :func:`_default_runner` so tests can swap it via
    ``monkeypatch.setattr(launchagent, "_default_runner", stub)``
    without having to thread a runner kwarg through every call site.
    """
    return _default_runner(
        ["launchctl", "bootstrap", gui_domain_target(uid), str(plist_file)],
        capture_output=True,
        text=True,
        check=False,
    )


def bootout_agent(
    *,
    uid: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """`launchctl bootout gui/<uid>/<label>` — unload + disable the job."""
    return _default_runner(
        ["launchctl", "bootout", gui_domain_label(uid)],
        capture_output=True,
        text=True,
        check=False,
    )


def print_agent(
    *,
    uid: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """`launchctl print gui/<uid>/<label>` — query live launchd state.

    Returns exit 0 + a verbose dump when the agent is loaded; exit
    non-zero (typically 113) when it isn't. Callers grep stdout for
    fields they care about (`state`, `next firing`, etc.).
    """
    return _default_runner(
        ["launchctl", "print", gui_domain_label(uid)],
        capture_output=True,
        text=True,
        check=False,
    )


__all__ = (
    "LABEL",
    "LogPaths",
    "SubprocessRunner",
    "bootout_agent",
    "bootstrap_agent",
    "ensure_log_dir",
    "gui_domain_label",
    "gui_domain_target",
    "is_installed",
    "log_paths",
    "plist_path",
    "print_agent",
    "render_plist",
    "write_plist_atomic",
)
