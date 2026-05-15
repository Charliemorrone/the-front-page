"""ClawFeed Intelligence CLI.

Phase 1 + Phase 6 surface:
    clawfeed-intel doctor              health-check vMLX, DB
    clawfeed-intel run daily           run a daily brief (24h window)
    clawfeed-intel cleanup             prune old raw_items + llm_calls
    clawfeed-intel cron install        install the 06:15 launchd job
    clawfeed-intel cron uninstall      remove the launchd job
    clawfeed-intel cron status         report whether the job is loaded

`doctor` is the canonical "is the system runnable" probe. It exits non-zero
if any check fails so a cron job can short-circuit cleanly before kicking
off a daily run. `cleanup` is the retention complement — keeps the DB
from growing unbounded over indefinite daily operation per the
architecture-doc retention policy (raw items: 30-90 days; llm calls:
30 days). `cron` registers the macOS launchd LaunchAgent that fires
the daily brief at 06:15 local time (architecture-doc Phase 6b; the
2026-05-15 amendment moved this from OpenClaw cron to launchd).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import closing

from pathlib import Path

from . import __version__, db, doctor, launchagent
from .llm import load_routing
from .paths import DB_PATH, REPO_ROOT
from .pipeline.orchestrator import run_daily
from .sources import build_source_plan

log = logging.getLogger(__name__)

_INTEL_TABLES: tuple[str, ...] = (
    "intel_runs",
    "intel_jobs",
    "raw_items",
    "run_raw_items",
    "item_clusters",
    "cluster_items",
    "item_summaries",
    "llm_calls",
    "source_fetch_state",
    "source_categories",
)


def _configure_logging() -> None:
    level = os.environ.get("CLAWFEED_LOG", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Probe vMLX + DB. Exit 0 if everything's reachable, 1 otherwise.

    The exit code is the contract: cron uses it to decide whether to run
    the daily brief. A non-zero return means the system isn't shippable —
    fix vMLX, then retry.
    """
    print(f"clawfeed-intel {__version__}")
    print(f"  repo:     {REPO_ROOT}")
    print(f"  db:       {DB_PATH}")
    print(f"            exists={DB_PATH.exists()}")

    db_ok = True
    if DB_PATH.exists():
        try:
            with closing(db.connect()) as conn:
                placeholders = ",".join("?" for _ in _INTEL_TABLES)
                row = conn.execute(
                    f"SELECT COUNT(*) AS n FROM sqlite_master "
                    f"WHERE type = 'table' AND name IN ({placeholders})",
                    _INTEL_TABLES,
                ).fetchone()
                found = row["n"] if row else 0
                marker = "ok" if found == len(_INTEL_TABLES) else "MISSING"
                print(f"            intel tables: {found}/{len(_INTEL_TABLES)} [{marker}]")
                if found != len(_INTEL_TABLES):
                    db_ok = False
        except Exception as exc:
            print(f"            db open failed: {exc}")
            db_ok = False
    else:
        print("            db missing — run migrations before first daily brief")
        db_ok = False

    print("  vmlx:")
    try:
        routing = load_routing()
    except Exception as exc:
        print(f"    [FAIL] routing config: {type(exc).__name__}: {exc}")
        print("  openclaw: not implemented")  # step 11
        return 1

    try:
        results = asyncio.run(doctor.run_doctor_probes(routing))
    except Exception as exc:
        print(f"    [FAIL] probes raised unexpectedly: {type(exc).__name__}: {exc}")
        print("  openclaw: not implemented")
        return 1

    vmlx_ok = True
    for result in results:
        marker = " ok " if result.ok else "FAIL"
        print(f"    [{marker}] {result.name:24s} {result.latency_ms:>5d}ms — {result.detail}")
        if not result.ok:
            vmlx_ok = False

    print("  openclaw: not implemented")  # step 11
    return 0 if (db_ok and vmlx_ok) else 1


def cmd_run_daily(args: argparse.Namespace) -> int:
    # Preflight: vMLX must be reachable + the configured filter model must
    # answer a single tiny chat completion before we touch fetch/cluster.
    # Without this guard, "vMLX was down from the start" silently degrades:
    # fetch + cluster succeed, every filter batch fails into
    # `coverage.failed_filter_batches`, all clusters stay at `status='pending'`,
    # and the run still publishes a stub digest with `kept_clusters=0` — which
    # cron / OpenClaw would read as success. Mid-run hiccups continue to
    # degrade per-batch as before; this only catches the "down from the
    # start" case. Exit 3 distinguishes preflight failure from the existing
    # 2 used for `ValueError` (bad window spec).
    if args.dry_run:
        return cmd_run_daily_dry_run(args)

    try:
        routing = load_routing()
    except Exception as exc:
        print(
            f"preflight: routing config failed to load: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3
    try:
        results = asyncio.run(doctor.run_doctor_probes(routing))
    except Exception as exc:
        print(
            f"preflight: probes raised unexpectedly: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3

    if any(not r.ok for r in results):
        print("preflight: vMLX is not ready — aborting before run", file=sys.stderr)
        for r in results:
            marker = " ok " if r.ok else "FAIL"
            print(
                f"    [{marker}] {r.name:24s} {r.latency_ms:>5d}ms — {r.detail}",
                file=sys.stderr,
            )
        return 3

    try:
        digest_id = run_daily(args.window)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"published digest {digest_id}")
    return 0


def cmd_run_daily_dry_run(args: argparse.Namespace) -> int:
    """Validate the daily-run pipeline without invoking LLM stages or fetchers.

    Each section reports ``[ ok ]`` / ``[FAIL]`` / ``[info]`` / ``[warn]``
    and a short detail line. Exit 0 if every critical check passes;
    exit 3 on the first failure that would prevent the live run from
    succeeding.

    The dry-run is the canonical "would the next 06:15 launchd fire
    actually produce a brief?" check. It runs every preflight the live
    ``run daily`` would run, plus inspection of pieces the live run
    doesn't validate up-front (source plan resolution, log dir
    writability, launchd registration state, Gemini CLI binary
    reachability). Useful right after ``cron install --install`` to
    catch configuration drift before the operator forgets the change.

    Critical (returns 3 on failure): routing parses; vMLX probes pass;
    ``gemini_cli`` provider's executable + script files exist when the
    provider is configured; source plan resolves without raising; log
    dir is creatable.

    Informational (never fails the check): launchd installation status;
    source plan warnings; absence of a configured ``gemini_cli``
    provider (means Tier-1 routes through vMLX, which is a valid but
    unusual deployment after the 2026-05-15 amendment).
    """
    print("=== clawfeed-intel run daily --dry-run ===")
    print(f"window: {args.window}")
    print()

    # 1. Routing config
    print("[routing]")
    try:
        routing = load_routing()
    except Exception as exc:
        print(f"  [FAIL] config failed to load: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(f"  [ ok ] {len(routing.stages)} stages: {', '.join(sorted(routing.stages))}")
    final = routing.stages.get("final_compose")
    if final is not None:
        fb = (
            f" → fallback to {final.fallback.provider}/{final.fallback.model}"
            if final.fallback
            else " (no fallback)"
        )
        print(f"  [ ok ] final_compose: {final.provider}/{final.model}{fb}")
    print()

    # 2. vMLX probes
    print("[vmlx]")
    try:
        probes = asyncio.run(doctor.run_doctor_probes(routing))
    except Exception as exc:
        print(f"  [FAIL] probes raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    for probe in probes:
        marker = " ok " if probe.ok else "FAIL"
        print(f"  [{marker}] {probe.name:24s} {probe.latency_ms:>5d}ms — {probe.detail}")
    if any(not p.ok for p in probes):
        print("  vMLX preflight failed — aborting", file=sys.stderr)
        return 3
    print()

    # 3. Gemini CLI provider reachability
    print("[gemini_cli]")
    gem_cfg = routing.providers.gemini_cli
    if gem_cfg is None:
        print(
            "  [warn] provider not declared — final_compose Tier 1 will use vMLX "
            "(unusual after the 2026-05-15 amendment)"
        )
    else:
        node_ok = gem_cfg.executable_path is None or Path(gem_cfg.executable_path).is_file()
        script_ok = Path(gem_cfg.script_path).is_file()
        node_marker = " ok " if node_ok else "FAIL"
        script_marker = " ok " if script_ok else "FAIL"
        print(f"  [{node_marker}] executable: {gem_cfg.executable_path or '<PATH-resolved>'}")
        print(f"  [{script_marker}] script:     {gem_cfg.script_path}")
        print(
            f"  [info] model: {final.model if final else '?'}, "
            f"idle {gem_cfg.idle_timeout_seconds:.0f}s, "
            f"hard {gem_cfg.hard_timeout_seconds:.0f}s, "
            f"retries {gem_cfg.retries}"
        )
        if not (node_ok and script_ok):
            print(
                "  gemini_cli binaries missing — Tier 1 will fail; brief will fall back to Tier 2",
                file=sys.stderr,
            )
            return 3
    print()

    # 4. Source plan
    print("[source_plan]")
    try:
        with closing(db.connect()) as conn:
            plan = build_source_plan(conn)
    except Exception as exc:
        print(f"  [FAIL] resolver raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    grouped = plan.tasks_by_kind()
    total_tasks = sum(len(v) for v in grouped.values())
    populated_kinds = [k for k, v in grouped.items() if v]
    print(
        f"  [ ok ] {len(plan.categories)} categories, {total_tasks} tasks across {len(populated_kinds)} fetcher kinds"
    )
    for kind in populated_kinds:
        print(f"         {kind}: {len(grouped[kind])} task(s)")
    if plan.warnings:
        for w in plan.warnings:
            cat = f" ({w.category})" if w.category else ""
            print(f"  [warn] {w.origin}{cat}: {w.message}")
    print()

    # 5. Log dir writability
    print("[log_dir]")
    log_paths = launchagent.log_paths()
    try:
        log_paths.directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"  [FAIL] cannot create {log_paths.directory}: {exc}", file=sys.stderr)
        return 3
    print(f"  [ ok ] {log_paths.directory}")
    print(f"  [info] launchd will write to:\n         {log_paths.out}\n         {log_paths.err}")
    print()

    # 6. LaunchAgent registration (informational only — daily run works
    # whether or not launchd is firing it; this just tells the operator
    # what they're looking at).
    print("[launchd]")
    plist = launchagent.plist_path()
    if launchagent.is_installed():
        print(f"  [ ok ] installed at {plist}")
    else:
        print(f"  [info] not installed at {plist}")
        print("         (run `clawfeed-intel cron install --install` to register)")
    print()

    print("preflight passed — `run daily` (without --dry-run) is ready to execute")
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Prune old ``raw_items`` and ``llm_calls`` per the retention policy.

    Default mode is dry-run: print counts of what would be removed, then
    exit 0 without touching the DB. ``--apply`` is the explicit opt-in to
    the destructive operation. This shape suits both interactive use
    ("show me what's stale") and cron use (call with ``--apply`` once a
    week). Exit code is 0 on success; non-zero only on unexpected errors
    (DB unreachable, etc.) so cron can read ``$?`` as a "did cleanup
    finish cleanly" signal.

    The two retention windows are independently tunable so an operator
    can keep more raw_items than llm_calls (the architecture-doc default).
    """
    raw_keep = args.raw_items_keep_days
    llm_keep = args.llm_calls_keep_days
    raw_cutoff = db.cutoff_iso(keep_days=raw_keep)
    llm_cutoff = db.cutoff_iso(keep_days=llm_keep)

    with closing(db.connect()) as conn:
        raw_count = db.count_raw_items_before(conn, raw_cutoff)
        llm_count = db.count_llm_calls_before(conn, llm_cutoff)
        if args.apply:
            raw_removed = db.prune_raw_items_before(conn, raw_cutoff)
            llm_removed = db.prune_llm_calls_before(conn, llm_cutoff)
            print(f"removed {raw_removed} raw_items older than {raw_cutoff}")
            print(f"removed {llm_removed} llm_calls older than {llm_cutoff}")
        else:
            print(f"would remove {raw_count} raw_items older than {raw_cutoff}")
            print(f"would remove {llm_count} llm_calls older than {llm_cutoff}")
            print("(re-run with --apply to actually delete)")
    return 0


def cmd_cron_install(args: argparse.Namespace) -> int:
    """Install (or preview) the launchd LaunchAgent for the daily brief.

    Default mode is dry-run: render the plist, print where it would be
    written, and exit 0 without touching the filesystem or launchd.
    ``--install`` is the explicit opt-in to the destructive operation
    (write the plist + ``launchctl bootstrap``). Mirrors Phase 6a's
    cleanup posture — destructive-default-off keeps an accidental
    invocation from registering a scheduled job.

    ``--hour`` / ``--minute`` let an operator shift the fire time
    without editing source; defaults are 06:15 per the architecture-
    doc's "Daily Schedule" section.

    Exit code 0 on success (install or dry-run); non-zero only on
    unexpected errors (plist write failed, launchctl returned an
    error message worth surfacing).
    """
    plist_text = launchagent.render_plist(hour=args.hour, minute=args.minute)
    target = launchagent.plist_path()

    if not args.install:
        print(f"would write LaunchAgent to: {target}")
        print(f"would invoke: launchctl bootstrap {launchagent.gui_domain_target()} {target}")
        print()
        print("--- plist preview ---")
        print(plist_text)
        print("(re-run with --install to actually register)")
        return 0

    launchagent.ensure_log_dir()
    written = launchagent.write_plist_atomic(plist_text)
    print(f"wrote {written}")

    # Bootout first — `bootstrap` of an already-loaded service is an
    # error on macOS. Treat bootout-failure as a soft "wasn't loaded";
    # only bootstrap-failure is a real problem.
    launchagent.bootout_agent()
    result = launchagent.bootstrap_agent(written)
    if result.returncode != 0:
        print(f"launchctl bootstrap failed (exit {result.returncode})", file=sys.stderr)
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        return 1
    print(f"loaded {launchagent.gui_domain_label()}")
    return 0


def cmd_cron_uninstall(args: argparse.Namespace) -> int:
    """Remove (or preview) the launchd LaunchAgent.

    Default mode is dry-run: print what would be removed. ``--remove``
    actually unloads via ``launchctl bootout`` and deletes the plist
    file. Both operations are idempotent — uninstalling when nothing
    is installed prints a friendly message and returns 0.
    """
    target = launchagent.plist_path()
    installed = target.is_file()

    if not args.remove:
        if installed:
            print(f"would invoke: launchctl bootout {launchagent.gui_domain_label()}")
            print(f"would remove: {target}")
            print("(re-run with --remove to actually uninstall)")
        else:
            print(f"no LaunchAgent installed at {target}")
        return 0

    if not installed:
        print(f"no LaunchAgent installed at {target}")
        return 0

    # bootout is best-effort — the plist may have been hand-removed
    # from launchd already; in that case bootout returns non-zero but
    # we still want to remove the file.
    launchagent.bootout_agent()
    target.unlink()
    print(f"removed {target}")
    return 0


def cmd_cron_status(_args: argparse.Namespace) -> int:
    """Report whether the LaunchAgent is installed + launchctl's view.

    Always returns exit 0; the printed status is the contract.
    """
    target = launchagent.plist_path()
    if not target.is_file():
        print(f"not installed (no plist at {target})")
        return 0
    print(f"installed at {target}")
    result = launchagent.print_agent()
    if result.returncode == 0:
        print(f"launchctl state for {launchagent.gui_domain_label()}:")
        print(result.stdout.rstrip())
    else:
        print(
            f"plist present but launchctl print returned {result.returncode} "
            "(agent may not be loaded — try `cron install --install`)"
        )
        if result.stderr:
            print(result.stderr.rstrip())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clawfeed-intel")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="health-check pipeline dependencies")

    run = sub.add_parser("run", help="execute a brief run")
    run_sub = run.add_subparsers(dest="run_type", required=True)
    daily = run_sub.add_parser("daily", help="run the daily brief")
    daily.add_argument(
        "--window",
        default="24h",
        help="time window covered by this run (default: 24h)",
    )
    daily.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "validate routing + vMLX + Gemini CLI + source plan + log dir + "
            "launchd registration, then exit without invoking fetchers or LLMs "
            "(useful before installing the cron job, after config changes, or "
            "when debugging a 'no brief published' report)"
        ),
    )

    cleanup = sub.add_parser(
        "cleanup",
        help="prune old raw_items and llm_calls per retention policy",
    )
    cleanup.add_argument(
        "--raw-items-keep-days",
        type=int,
        default=90,
        help="delete raw_items older than N days (default: 90)",
    )
    cleanup.add_argument(
        "--llm-calls-keep-days",
        type=int,
        default=30,
        help="delete llm_calls older than N days (default: 30)",
    )
    cleanup.add_argument(
        "--apply",
        action="store_true",
        help="actually delete (default: dry-run, print counts only)",
    )

    cron = sub.add_parser(
        "cron",
        help="register/unregister the macOS launchd daily-brief schedule",
    )
    cron_sub = cron.add_subparsers(dest="cron_action", required=True)

    cron_install = cron_sub.add_parser(
        "install",
        help="install the LaunchAgent (default: dry-run, --install to write)",
    )
    cron_install.add_argument(
        "--install",
        action="store_true",
        help="actually write the plist + launchctl bootstrap (default: dry-run)",
    )
    cron_install.add_argument(
        "--hour", type=int, default=6, help="fire-time hour 0-23 (default: 6)"
    )
    cron_install.add_argument(
        "--minute", type=int, default=15, help="fire-time minute 0-59 (default: 15)"
    )

    cron_uninstall = cron_sub.add_parser(
        "uninstall",
        help="remove the LaunchAgent (default: dry-run, --remove to delete)",
    )
    cron_uninstall.add_argument(
        "--remove",
        action="store_true",
        help="actually launchctl bootout + delete the plist (default: dry-run)",
    )

    cron_sub.add_parser("status", help="report whether the LaunchAgent is loaded")

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "doctor":
        return cmd_doctor(args)
    if args.cmd == "run" and args.run_type == "daily":
        return cmd_run_daily(args)
    if args.cmd == "cleanup":
        return cmd_cleanup(args)
    if args.cmd == "cron":
        if args.cron_action == "install":
            return cmd_cron_install(args)
        if args.cron_action == "uninstall":
            return cmd_cron_uninstall(args)
        if args.cron_action == "status":
            return cmd_cron_status(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
