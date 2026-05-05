"""ClawFeed Intelligence CLI.

Phase 1 surface:
    clawfeed-intel doctor             health-check vMLX, OpenClaw, DB
    clawfeed-intel run daily          run a daily brief (24h window)

Implementation lands incrementally; subcommands print a clear "not implemented"
message until their pipeline stage is wired up.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import closing

from . import __version__, db
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
    print(f"clawfeed-intel {__version__}")
    print(f"  repo:     {REPO_ROOT}")
    print(f"  db:       {DB_PATH}")
    print(f"            exists={DB_PATH.exists()}")

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
        except Exception as exc:
            print(f"            db open failed: {exc}")

    print("  vmlx:     not implemented")
    print("  openclaw: not implemented")
    return 0


def cmd_run_daily(args: argparse.Namespace) -> int:
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
