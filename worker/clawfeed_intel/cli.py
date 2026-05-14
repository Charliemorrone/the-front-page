"""ClawFeed Intelligence CLI.

Phase 1 surface:
    clawfeed-intel doctor             health-check vMLX, OpenClaw, DB
    clawfeed-intel run daily          run a daily brief (24h window)

`doctor` is the canonical "is the system runnable" probe. It exits non-zero
if any check fails so a cron job can short-circuit cleanly before kicking
off a daily run.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import closing

from . import __version__, db, doctor
from .llm import load_routing
from .paths import DB_PATH, REPO_ROOT
from .pipeline.orchestrator import run_daily

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

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "doctor":
        return cmd_doctor(args)
    if args.cmd == "run" and args.run_type == "daily":
        return cmd_run_daily(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
