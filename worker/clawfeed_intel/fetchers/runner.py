"""Drive the fetching stage of a run.

Walks the ``SourcePlan`` produced by :func:`sources.build_source_plan`,
dispatches each ``ResolvedTask`` to its registered fetcher, persists the
returned items, and records outcomes against both
:class:`runs.Coverage` and ``source_fetch_state``.

Failure handling mirrors the hard requirement that *failed sources degrade
coverage; they do not fail the run*:

- A fetcher that raises produces a ``failed`` outcome and a coverage
  ``failed_sources`` entry; sibling tasks of the same kind continue.
- A kind with no registered fetcher produces ``skipped`` outcomes for all
  its tasks. This is the honest state during the staged build-out — the
  harness ships before all eight fetchers are written.
- Resolver-side warnings (missing config, malformed YAML entries, unknown
  ``sources.type``) flow into ``Coverage.plan_warnings`` so the brief can
  surface them too.

Concurrency: tasks of the same kind run in parallel via :func:`asyncio.gather`.
HTTP fan-out lives inside the fetchers; the runner only orchestrates.
SQLite writes happen serially because the event loop is single-threaded —
upserts and ``source_fetch_state`` updates can't interleave even though
HTTP fetches do.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time

from .. import db
from ..runs import Coverage
from ..sources import PlanWarning, ResolvedTask, SourcePlan
from .base import FETCHER_REGISTRY, FetchedItem, FetcherCallable, FetchOutcome

log = logging.getLogger(__name__)


async def run_fetch_stage(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    plan: SourcePlan,
    coverage: Coverage,
    fetchers: dict[str, FetcherCallable] | None = None,
) -> list[FetchOutcome]:
    """Execute every resolved task in *plan* and return per-task outcomes.

    Mutates *coverage* in place. Idempotency is provided by
    :func:`db.upsert_raw_item`, so re-running the same plan against a run
    that already has items only links them rather than duplicating them.

    *fetchers* defaults to the module-level :data:`FETCHER_REGISTRY`. Tests
    pass a small dict of fakes; production passes a registry preloaded with
    the imported fetcher modules.
    """
    registry = fetchers if fetchers is not None else FETCHER_REGISTRY

    for warning in plan.warnings:
        coverage.record_plan_warning(_format_plan_warning(warning))

    outcomes: list[FetchOutcome] = []
    for kind, tasks in plan.tasks_by_kind().items():
        if not tasks:
            continue
        fetcher = registry.get(kind)
        if fetcher is None:
            for task in tasks:
                outcomes.append(_record_skipped(conn, run_id, task, coverage))
            continue

        gathered = await asyncio.gather(
            *(_run_one(conn, run_id, task, fetcher, coverage) for task in tasks)
        )
        outcomes.extend(gathered)

    return outcomes


async def _run_one(
    conn: sqlite3.Connection,
    run_id: int,
    task: ResolvedTask,
    fetcher: FetcherCallable,
    coverage: Coverage,
) -> FetchOutcome:
    started = time.monotonic()
    try:
        items = await fetcher(conn, task)
    except Exception as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        log.exception("run %d fetch failed: kind=%s source=%r", run_id, task.kind, task.source_name)
        outcome = FetchOutcome(
            kind=task.kind,
            source_id=task.source_id,
            source_name=task.source_name,
            status="failed",
            latency_ms=latency_ms,
            error=_short_error(exc),
        )
        coverage.record_failure(task.source_name)
        if task.source_id is not None:
            db.record_fetch_failure(
                conn, source_id=task.source_id, fetcher=task.kind, error=outcome.error or ""
            )
        return outcome

    items_new = _persist_items(conn, run_id, task, items)
    latency_ms = int((time.monotonic() - started) * 1000)
    outcome = FetchOutcome(
        kind=task.kind,
        source_id=task.source_id,
        source_name=task.source_name,
        status="succeeded",
        items_seen=len(items),
        items_new=items_new,
        latency_ms=latency_ms,
    )
    coverage.record_success(task.source_name, outcome.items_seen)
    if task.source_id is not None:
        db.record_fetch_success(conn, source_id=task.source_id, fetcher=task.kind)
    return outcome


def _persist_items(
    conn: sqlite3.Connection,
    run_id: int,
    task: ResolvedTask,
    items: list[FetchedItem],
) -> int:
    """Upsert each fetched item; return how many were first sightings.

    Per-item failures are logged and the rest of the batch continues — one
    bad row should not blackhole an otherwise healthy fetch.
    """
    items_new = 0
    for item in items:
        try:
            _, was_new = db.upsert_raw_item(
                conn,
                run_id=run_id,
                source_id=task.source_id,
                source_name=task.source_name,
                **item.upsert_kwargs(),
            )
        except Exception:
            log.exception(
                "run %d: upsert failed for %s/%s", run_id, item.source_type, item.dedup_key
            )
            continue
        if was_new:
            items_new += 1
    return items_new


def _record_skipped(
    conn: sqlite3.Connection,
    run_id: int,
    task: ResolvedTask,
    coverage: Coverage,
) -> FetchOutcome:
    reason = f"no fetcher for kind {task.kind!r}"
    coverage.record_skipped(task.source_name, reason)
    # Don't update source_fetch_state for a skip — we never tried, so
    # bumping consecutive_errors would be misleading. The skip is captured
    # in coverage and in the returned outcome.
    return FetchOutcome(
        kind=task.kind,
        source_id=task.source_id,
        source_name=task.source_name,
        status="skipped",
        error=reason,
    )


def _format_plan_warning(warning: PlanWarning) -> str:
    cat = warning.category or "<config>"
    return f"plan({warning.origin}/{cat}): {warning.message}"


def _short_error(exc: BaseException) -> str:
    """One-line ``Type: message`` form, capped to keep the metadata blob small."""
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 240 else text[:237] + "…"
