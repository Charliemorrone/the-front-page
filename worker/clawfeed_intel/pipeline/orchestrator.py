"""End-to-end run orchestration.

The orchestrator drives a run through the lifecycle states declared in
``intel_runs.status``::

    pending → fetching → filtering → summarizing → composing → published

Each state transition is durable in the DB so a partial run is observable from
the dashboard. Stage implementations land incrementally; until they exist this
module produces a skeleton brief whose only purpose is to prove the spine —
run row created, states walked, digest published, ``intel_runs.digest_id``
linked.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3

from .. import db
from ..fetchers import run_fetch_stage
from ..runs import RunMetadata
from ..sources import build_source_plan
from ..timewindow import window_for

log = logging.getLogger(__name__)


_STUB_BRIEF_TEMPLATE = """# Daily Intelligence Brief — {date}

_This brief was produced by the run-lifecycle skeleton. Fetcher and
intelligence stages have not been implemented yet; the skeleton exists to
prove that runs walk through every state and publish a real digest._

## Coverage

- Sources attempted: {sources_attempted}
- Sources succeeded: {sources_succeeded}
- Raw items: {raw_items}
- Event clusters: {clusters}
- Kept clusters: {kept_clusters}
- Failed sources: {failed_sources}

_Run id: {run_id}. Window: {window_start} → {window_end}._
"""


def run_daily(window_spec: str, *, conn: sqlite3.Connection | None = None) -> int:
    """Execute one daily run end-to-end. Returns the published digest id.

    If *conn* is omitted a fresh connection is opened and closed. Tests pass an
    existing connection bound to a temp DB.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = db.connect()
    try:
        return _execute_daily(conn, window_spec)
    finally:
        if owns_conn:
            conn.close()


def _execute_daily(conn: sqlite3.Connection, window_spec: str) -> int:
    window_start, window_end = window_for(window_spec)
    log.info("creating daily run window=%s..%s", window_start, window_end)
    run_id = db.create_run(
        conn,
        run_type="daily",
        window_start=window_start,
        window_end=window_end,
    )
    log.info("run %d: created (pending)", run_id)

    metadata = RunMetadata(
        brief_kind="daily",
        run_id=run_id,
        window_start=window_start,
        window_end=window_end,
    )

    try:
        return _drive_run(conn, run_id, metadata)
    except Exception as exc:
        log.exception("run %d: failed during execution", run_id)
        try:
            db.finish_run(conn, run_id, status="failed", error=str(exc))
        except Exception:
            log.exception("run %d: failed-state update also failed", run_id)
        raise


def _drive_run(conn: sqlite3.Connection, run_id: int, metadata: RunMetadata) -> int:
    db.mark_run_started(conn, run_id)
    log.info("run %d: → fetching", run_id)
    plan = build_source_plan(conn)
    log.info(
        "run %d: source plan resolved (%d categories, %d warnings)",
        run_id,
        len(plan.categories),
        len(plan.warnings),
    )
    asyncio.run(run_fetch_stage(conn, run_id=run_id, plan=plan, coverage=metadata.coverage))

    db.advance_run_status(conn, run_id, "filtering")
    log.info("run %d: → filtering", run_id)
    # dedup() / cluster() / relevance_filter()           # TODO: steps 5/7/9

    db.advance_run_status(conn, run_id, "summarizing")
    log.info("run %d: → summarizing", run_id)
    # cluster_summary()                                  # TODO: step 10

    db.advance_run_status(conn, run_id, "composing")
    log.info("run %d: → composing", run_id)
    markdown = _compose_stub(metadata)

    db.update_run_metadata(conn, run_id, metadata.as_json())
    digest_id = db.create_digest(
        conn,
        digest_type="daily",
        content=markdown,
        metadata=metadata.as_json(),
    )
    db.finish_run(conn, run_id, status="published", digest_id=digest_id)
    log.info("run %d: → published (digest=%d)", run_id, digest_id)
    return digest_id


def _compose_stub(metadata: RunMetadata) -> str:
    coverage = metadata.coverage
    return _STUB_BRIEF_TEMPLATE.format(
        date=metadata.window_end[:10],
        run_id=metadata.run_id,
        window_start=metadata.window_start,
        window_end=metadata.window_end,
        sources_attempted=coverage.sources_attempted,
        sources_succeeded=coverage.sources_succeeded,
        raw_items=coverage.raw_items,
        clusters=coverage.clusters,
        kept_clusters=coverage.kept_clusters,
        failed_sources=", ".join(coverage.failed_sources) or "none",
    )
