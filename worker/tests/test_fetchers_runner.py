"""Tests for the fetcher runner.

Concrete fetcher modules (RSS, arXiv, …) get their own test files in
subsequent steps; here we exercise the runner contract with stub fetchers.
The runner has three responsibilities:

1. Persist fetched items, update Coverage, update source_fetch_state on
   success and failure.
2. Catch fetcher exceptions and produce a 'failed' outcome — never let a
   single bad source kill the run.
3. Honestly record 'skipped' for kinds without a registered fetcher, so a
   partially-built harness produces an honest coverage report.

These behaviours are the load-bearing part of the "failed sources degrade
coverage; they do not fail the run" hard requirement.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing


from clawfeed_intel import db as worker_db
from clawfeed_intel.fetchers import FetchedItem, run_fetch_stage
from clawfeed_intel.runs import Coverage
from clawfeed_intel.sources import (
    CategoryPlan,
    GdeltTask,
    PlanWarning,
    ProfileConfig,
    ResolvedTask,
    RssTask,
    SourcePlan,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_run(conn: sqlite3.Connection) -> int:
    return worker_db.create_run(
        conn,
        run_type="daily",
        window_start="2026-05-04T00:00:00+00:00",
        window_end="2026-05-05T00:00:00+00:00",
    )


def _add_source(conn: sqlite3.Connection, name: str, type_: str = "rss") -> int:
    cur = conn.execute(
        "INSERT INTO sources (name, type, config, is_active) VALUES (?, ?, ?, 1)",
        (name, type_, "{}"),
    )
    return int(cur.lastrowid)


def _yaml_task(*, kind: str = "rss", category: str = "scratch") -> ResolvedTask:
    if kind == "rss":
        task = RssTask(kind="rss", url="https://example.com/feed")
    elif kind == "gdelt":
        task = GdeltTask(kind="gdelt", query="anything")
    else:
        raise ValueError(f"helper does not know how to build {kind!r}")
    return ResolvedTask(
        task=task,
        category=category,
        origin="yaml",
        source_id=None,
        source_name=f"{category}:{kind}",
    )


def _db_task(source_id: int, source_name: str, *, kind: str = "rss") -> ResolvedTask:
    if kind == "rss":
        task = RssTask(kind="rss", url="https://example.com/feed")
    else:
        raise ValueError(f"helper does not know how to build {kind!r}")
    return ResolvedTask(
        task=task,
        category="scratch",
        origin="db",
        source_id=source_id,
        source_name=source_name,
    )


def _plan(
    *,
    categories: list[CategoryPlan] | None = None,
    warnings: list[PlanWarning] | None = None,
) -> SourcePlan:
    return SourcePlan(
        profile=ProfileConfig(),
        categories=categories or [],
        dynamic_search=[],
        warnings=warnings or [],
    )


def _items(*urls: str) -> list[FetchedItem]:
    out: list[FetchedItem] = []
    for url in urls:
        out.append(
            FetchedItem(
                source_type="rss",
                dedup_key=url,
                title=url.rsplit("/", 1)[-1],
                url=url,
                canonical_url=url,
                content=f"body for {url}",
                content_hash="hash-" + url.rsplit("/", 1)[-1],
            )
        )
    return out


# ── happy path ────────────────────────────────────────────────────────────────


async def test_success_persists_items_and_updates_coverage(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        sid = _add_source(conn, "rss-x")

        async def fetch(_conn, _task):
            return _items("https://x.example/a", "https://x.example/b")

        task = _db_task(sid, "rss-x")
        plan = _plan(categories=[CategoryPlan(name="scratch", tasks=[task])])
        coverage = Coverage()

        outcomes = await run_fetch_stage(
            conn, run_id=run_id, plan=plan, coverage=coverage, fetchers={"rss": fetch}
        )

    assert len(outcomes) == 1
    o = outcomes[0]
    assert o.status == "succeeded"
    assert o.items_seen == 2
    assert o.items_new == 2
    assert o.error is None
    assert o.latency_ms >= 0

    assert coverage.sources_attempted == 1
    assert coverage.sources_succeeded == 1
    assert coverage.raw_items == 2
    assert coverage.failed_sources == []
    assert coverage.skipped_sources == []


async def test_success_writes_run_membership_and_fetch_state(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        sid = _add_source(conn, "rss-x")

        async def fetch(_conn, _task):
            return _items("https://x.example/a", "https://x.example/b")

        task = _db_task(sid, "rss-x")
        plan = _plan(categories=[CategoryPlan(name="scratch", tasks=[task])])

        await run_fetch_stage(
            conn,
            run_id=run_id,
            plan=plan,
            coverage=Coverage(),
            fetchers={"rss": fetch},
        )

        # Both items linked to this run.
        rows = conn.execute(
            "SELECT raw_item_id FROM run_raw_items WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        assert len(rows) == 2

        sfs = conn.execute(
            "SELECT * FROM source_fetch_state WHERE source_id = ? AND fetcher = 'rss'",
            (sid,),
        ).fetchone()
        assert sfs is not None
        assert sfs["last_success_at"] is not None
        assert sfs["last_error"] is None
        assert sfs["consecutive_errors"] == 0


async def test_yaml_origin_task_does_not_touch_fetch_state(temp_db):
    """YAML-origin tasks have no source_id, so source_fetch_state has no row
    to address — bumping a row keyed on NULL would be meaningless."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)

        async def fetch(_conn, _task):
            return _items("https://x.example/a")

        task = _yaml_task()
        plan = _plan(categories=[CategoryPlan(name="scratch", tasks=[task])])

        await run_fetch_stage(
            conn,
            run_id=run_id,
            plan=plan,
            coverage=Coverage(),
            fetchers={"rss": fetch},
        )

        sfs = conn.execute("SELECT COUNT(*) AS n FROM source_fetch_state").fetchone()
        assert sfs["n"] == 0


async def test_idempotent_rerun_records_zero_new(temp_db):
    """Same items fetched twice → second run reports items_seen=N, items_new=0."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id_1 = _make_run(conn)
        sid = _add_source(conn, "rss-x")

        async def fetch(_conn, _task):
            return _items("https://x.example/a", "https://x.example/b")

        task = _db_task(sid, "rss-x")
        plan = _plan(categories=[CategoryPlan(name="scratch", tasks=[task])])

        await run_fetch_stage(
            conn,
            run_id=run_id_1,
            plan=plan,
            coverage=Coverage(),
            fetchers={"rss": fetch},
        )

        run_id_2 = _make_run(conn)
        coverage = Coverage()
        outcomes = await run_fetch_stage(
            conn,
            run_id=run_id_2,
            plan=plan,
            coverage=coverage,
            fetchers={"rss": fetch},
        )

        assert outcomes[0].items_seen == 2
        assert outcomes[0].items_new == 0
        assert coverage.raw_items == 2  # coverage counts items_seen, not items_new

        # Run 2 still gets membership rows even though items aren't new.
        rows_run2 = conn.execute(
            "SELECT 1 FROM run_raw_items WHERE run_id = ?", (run_id_2,)
        ).fetchall()
        assert len(rows_run2) == 2


# ── failure path ──────────────────────────────────────────────────────────────


async def test_fetcher_exception_becomes_failed_outcome(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        sid = _add_source(conn, "rss-x")

        async def fetch(_conn, _task):
            raise TimeoutError("upstream feed timed out")

        task = _db_task(sid, "rss-x")
        plan = _plan(categories=[CategoryPlan(name="scratch", tasks=[task])])
        coverage = Coverage()

        outcomes = await run_fetch_stage(
            conn,
            run_id=run_id,
            plan=plan,
            coverage=coverage,
            fetchers={"rss": fetch},
        )

    assert outcomes[0].status == "failed"
    assert "TimeoutError" in (outcomes[0].error or "")
    assert "upstream feed" in (outcomes[0].error or "")
    assert coverage.sources_attempted == 1
    assert coverage.sources_succeeded == 0
    assert coverage.failed_sources == ["rss-x"]


async def test_fetcher_failure_records_fetch_state(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        sid = _add_source(conn, "rss-x")

        async def fetch(_conn, _task):
            raise RuntimeError("boom")

        task = _db_task(sid, "rss-x")
        plan = _plan(categories=[CategoryPlan(name="scratch", tasks=[task])])

        await run_fetch_stage(
            conn,
            run_id=run_id,
            plan=plan,
            coverage=Coverage(),
            fetchers={"rss": fetch},
        )

        row = conn.execute(
            "SELECT * FROM source_fetch_state WHERE source_id = ? AND fetcher = 'rss'",
            (sid,),
        ).fetchone()
        assert row is not None
        assert row["consecutive_errors"] == 1
        assert "boom" in row["last_error"]


async def test_one_failure_does_not_kill_other_tasks(temp_db):
    """Sibling tasks of the same kind must continue when one fetcher raises."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        sid_ok = _add_source(conn, "rss-ok")
        sid_bad = _add_source(conn, "rss-bad")

        async def fetch(_conn, task):
            if task.source_name == "rss-bad":
                raise RuntimeError("boom")
            return _items("https://x.example/a")

        plan = _plan(
            categories=[
                CategoryPlan(
                    name="scratch",
                    tasks=[_db_task(sid_ok, "rss-ok"), _db_task(sid_bad, "rss-bad")],
                )
            ]
        )
        coverage = Coverage()
        outcomes = await run_fetch_stage(
            conn,
            run_id=run_id,
            plan=plan,
            coverage=coverage,
            fetchers={"rss": fetch},
        )

    statuses = sorted((o.source_name, o.status) for o in outcomes)
    assert statuses == [("rss-bad", "failed"), ("rss-ok", "succeeded")]
    assert coverage.sources_succeeded == 1
    assert coverage.failed_sources == ["rss-bad"]
    assert coverage.raw_items == 1


async def test_per_item_upsert_failure_does_not_kill_batch(monkeypatch, temp_db):
    """One bad item should not blackhole an otherwise healthy fetch."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)

        async def fetch(_conn, _task):
            return _items("https://x.example/a", "https://x.example/b", "https://x.example/c")

        task = _yaml_task()
        plan = _plan(categories=[CategoryPlan(name="scratch", tasks=[task])])

        original = worker_db.upsert_raw_item
        calls = {"n": 0}

        def flaky_upsert(conn_, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated upsert failure")
            return original(conn_, **kwargs)

        monkeypatch.setattr("clawfeed_intel.fetchers.runner.db.upsert_raw_item", flaky_upsert)

        coverage = Coverage()
        outcomes = await run_fetch_stage(
            conn,
            run_id=run_id,
            plan=plan,
            coverage=coverage,
            fetchers={"rss": fetch},
        )

        # All three items reported as seen; only two land as new (the second
        # raised inside upsert and was logged + skipped).
        assert outcomes[0].status == "succeeded"
        assert outcomes[0].items_seen == 3
        assert outcomes[0].items_new == 2

        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM raw_items WHERE source_type = 'rss'"
        ).fetchone()
        assert rows["n"] == 2


# ── skipped path ──────────────────────────────────────────────────────────────


async def test_no_fetcher_for_kind_records_skipped(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)

        plan = _plan(
            categories=[
                CategoryPlan(
                    name="scratch",
                    tasks=[_yaml_task(kind="gdelt"), _yaml_task(kind="rss")],
                )
            ]
        )
        coverage = Coverage()

        # Only RSS has a fetcher; gdelt should be skipped.
        async def fetch_rss(_conn, _task):
            return _items("https://x.example/a")

        outcomes = await run_fetch_stage(
            conn,
            run_id=run_id,
            plan=plan,
            coverage=coverage,
            fetchers={"rss": fetch_rss},
        )

    by_kind = {(o.kind, o.status) for o in outcomes}
    assert by_kind == {("rss", "succeeded"), ("gdelt", "skipped")}
    assert any("no fetcher" in s for s in coverage.skipped_sources)
    assert coverage.sources_attempted == 2
    assert coverage.sources_succeeded == 1


async def test_skipped_does_not_touch_fetch_state(temp_db):
    """A 'skipped' outcome means we never tried — bumping consecutive_errors
    would be misleading."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        sid = _add_source(conn, "gdelt-x", type_="rss")  # type doesn't matter here

        plan = _plan(
            categories=[
                CategoryPlan(
                    name="scratch",
                    tasks=[
                        ResolvedTask(
                            task=GdeltTask(kind="gdelt", query="anything"),
                            category="scratch",
                            origin="db",
                            source_id=sid,
                            source_name="gdelt-x",
                        )
                    ],
                )
            ]
        )
        await run_fetch_stage(conn, run_id=run_id, plan=plan, coverage=Coverage(), fetchers={})

        row = conn.execute("SELECT COUNT(*) AS n FROM source_fetch_state").fetchone()
        assert row["n"] == 0


# ── plan warnings ─────────────────────────────────────────────────────────────


async def test_plan_warnings_flow_into_coverage(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        plan = _plan(
            warnings=[
                PlanWarning(origin="config", category=None, message="config not found"),
                PlanWarning(origin="yaml", category="scratch", message="bad source entry"),
            ]
        )
        coverage = Coverage()
        await run_fetch_stage(conn, run_id=run_id, plan=plan, coverage=coverage, fetchers={})

    assert len(coverage.plan_warnings) == 2
    assert any("config not found" in w for w in coverage.plan_warnings)
    assert any("scratch" in w and "bad source entry" in w for w in coverage.plan_warnings)


# ── empty plan ────────────────────────────────────────────────────────────────


async def test_empty_plan_runs_clean(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        coverage = Coverage()
        outcomes = await run_fetch_stage(
            conn, run_id=run_id, plan=_plan(), coverage=coverage, fetchers={}
        )
    assert outcomes == []
    assert coverage.sources_attempted == 0
    assert coverage.failed_sources == []
    assert coverage.skipped_sources == []


# ── concurrent dispatch ───────────────────────────────────────────────────────


async def test_tasks_of_same_kind_run_concurrently(temp_db):
    """Five tasks each sleeping 50ms must finish in under ~150ms — gather
    runs them in parallel rather than serially.
    """
    import asyncio
    import time

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)

        async def slow(_conn, _task):
            await asyncio.sleep(0.05)
            return []

        tasks = [_yaml_task(category=f"c{i}") for i in range(5)]
        plan = _plan(categories=[CategoryPlan(name="scratch", tasks=tasks)])

        started = time.monotonic()
        outcomes = await run_fetch_stage(
            conn,
            run_id=run_id,
            plan=plan,
            coverage=Coverage(),
            fetchers={"rss": slow},
        )
        elapsed = time.monotonic() - started

    assert len(outcomes) == 5
    assert all(o.status == "succeeded" for o in outcomes)
    # Five sequential 50ms sleeps would take ~250ms; concurrent ~50ms plus
    # event-loop overhead. Allow generous headroom for slow CI hosts.
    assert elapsed < 0.20, f"expected concurrent dispatch, took {elapsed:.3f}s"
