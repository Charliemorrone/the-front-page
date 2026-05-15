"""End-to-end run orchestration.

The orchestrator drives a run through the lifecycle states declared in
``intel_runs.status``::

    pending → fetching → filtering → summarizing → composing → published

Each state transition is durable in the DB so a partial run is observable from
the dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3

from .. import db
from ..fetchers import run_fetch_stage
from ..llm import LLMClient, RoutingConfig, load_routing
from ..runs import RunMetadata
from ..sources import build_source_plan
from ..timewindow import window_for
from .cluster import cluster_run
from .compose import compose_brief
from .relevance import filter_clusters
from .summary import summarize_clusters

log = logging.getLogger(__name__)


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
    metadata.coverage.clusters = cluster_run(conn, run_id)

    routing = load_routing()
    llm_client = _build_llm_client(routing, conn=conn, run_id=run_id)
    stage_config = routing.resolve("relevance_filter")
    metadata.coverage.kept_clusters = asyncio.run(
        filter_clusters(
            conn,
            run_id,
            llm_client,
            metadata.coverage,
            categories=list(plan.categories),
            batch_size=stage_config.batch_size or 12,
        )
    )
    metadata.local_models["filter"] = stage_config.model
    log.info(
        "run %d: filtered (%d kept, %d failed batches)",
        run_id,
        metadata.coverage.kept_clusters,
        metadata.coverage.failed_filter_batches,
    )

    db.advance_run_status(conn, run_id, "summarizing")
    log.info("run %d: → summarizing", run_id)
    summary_stage_config = routing.resolve("cluster_summary")
    metadata.coverage.summarized_clusters = asyncio.run(
        summarize_clusters(
            conn,
            run_id,
            llm_client,
            metadata.coverage,
            plan=plan,
            model=summary_stage_config.model,
        )
    )
    metadata.local_models["summary"] = summary_stage_config.model
    log.info(
        "run %d: summarized (%d clusters, %d failed)",
        run_id,
        metadata.coverage.summarized_clusters,
        metadata.coverage.failed_summary_clusters,
    )

    db.advance_run_status(conn, run_id, "composing")
    log.info("run %d: → composing", run_id)
    compose_stage_config = routing.resolve("final_compose")
    compose_result = asyncio.run(
        compose_brief(
            conn,
            run_id,
            llm_client,
            plan=plan,
            coverage=metadata.coverage,
            window_start=metadata.window_start,
            window_end=metadata.window_end,
            model=compose_stage_config.model,
        )
    )
    metadata.composition_provider = compose_result.provider_tag
    metadata.composition_model = compose_result.model
    log.info(
        "run %d: composed (provider=%s, model=%s)",
        run_id,
        compose_result.provider_tag,
        compose_result.model,
    )

    db.update_run_metadata(conn, run_id, metadata.as_json())
    digest_id = db.create_digest(
        conn,
        digest_type="daily",
        content=compose_result.markdown,
        metadata=metadata.as_json(),
    )
    db.finish_run(conn, run_id, status="published", digest_id=digest_id)
    log.info("run %d: → published (digest=%d)", run_id, digest_id)
    return digest_id


def _build_llm_client(
    routing: RoutingConfig,
    *,
    conn: sqlite3.Connection,
    run_id: int,
) -> LLMClient:
    """Construct the per-run LLM client.

    Pulled into a thin helper so tests can monkeypatch this name to
    inject a client backed by :class:`httpx.MockTransport` without
    needing to refactor :func:`_drive_run`'s control flow.
    """
    return LLMClient(routing, conn=conn, run_id=run_id)
