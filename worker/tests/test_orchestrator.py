from __future__ import annotations

import json
from contextlib import closing

import pytest

from clawfeed_intel import db as worker_db
from clawfeed_intel.pipeline.orchestrator import run_daily
from clawfeed_intel.sources import build_source_plan


def _isolate_config(monkeypatch, tmp_path, body: str = "categories: {}\n"):
    """Point the resolver at a tmp config so orchestrator tests don't depend
    on the editorial config that ships in the repo."""
    config = tmp_path / "intel-sources.yaml"
    config.write_text(body, encoding="utf-8")
    monkeypatch.setattr("clawfeed_intel.sources.DEFAULT_CONFIG_PATH", config)
    return config


def _empty_fetcher_registry(monkeypatch):
    """Detach the orchestrator from whichever real fetchers happen to be
    registered. Each fetcher step (6.1 RSS, 6.2 arXiv, …) adds its own
    callable at import time, which would otherwise hit live HTTP from
    orchestrator tests that exercise the 'no-fetcher' code path.

    We patch ``runner.FETCHER_REGISTRY`` (the binding the runner actually
    reads) rather than ``base.FETCHER_REGISTRY``; the runner ``from .base
    import`` rebinds it into its own module namespace, and monkeypatch on
    that namespace is what the function lookup sees at call time.
    """
    monkeypatch.setattr("clawfeed_intel.fetchers.runner.FETCHER_REGISTRY", {})


def test_run_daily_publishes_digest_and_links_run(temp_db, monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    with closing(worker_db.connect(temp_db)) as conn:
        digest_id = run_daily("24h", conn=conn)

        digest = conn.execute("SELECT * FROM digests WHERE id = ?", (digest_id,)).fetchone()
        assert digest is not None
        assert digest["type"] == "daily"
        assert "Daily Intelligence Brief" in digest["content"]

        meta = json.loads(digest["metadata"])
        assert meta["brief_kind"] == "daily"
        assert meta["run_id"] > 0
        assert meta["window_start"].endswith("+00:00")
        assert meta["window_end"].endswith("+00:00")
        assert "coverage" in meta
        assert meta["coverage"]["sources_attempted"] == 0
        assert meta["coverage"]["failed_sources"] == []
        assert meta["coverage"]["skipped_sources"] == []
        assert meta["coverage"]["plan_warnings"] == []

        runs = conn.execute("SELECT * FROM intel_runs WHERE digest_id = ?", (digest_id,)).fetchall()
        assert len(runs) == 1
        run = runs[0]
        assert run["status"] == "published"
        assert run["started_at"] is not None
        assert run["finished_at"] is not None
        assert run["error"] is None
        assert run["run_type"] == "daily"


def test_run_daily_skips_resolved_tasks_when_no_fetcher_registered(temp_db, monkeypatch, tmp_path):
    """With the resolver wired in but no fetchers registered, every YAML task
    becomes a 'skipped' coverage entry. The run still publishes; the brief
    just reports an honest empty harness."""
    yaml_body = """
categories:
  scratch:
    sources:
      - kind: rss
        url: https://example.com/feed
      - kind: gdelt
        query: hello
"""
    _isolate_config(monkeypatch, tmp_path, yaml_body)
    _empty_fetcher_registry(monkeypatch)

    with closing(worker_db.connect(temp_db)) as conn:
        digest_id = run_daily("24h", conn=conn)

        plan = build_source_plan(conn)
        expected = sum(len(tasks) for tasks in plan.tasks_by_kind().values())

        meta = json.loads(
            conn.execute("SELECT metadata FROM digests WHERE id = ?", (digest_id,)).fetchone()[
                "metadata"
            ]
        )

    assert expected == 2
    assert meta["coverage"]["sources_attempted"] == 2
    assert meta["coverage"]["sources_succeeded"] == 0
    assert len(meta["coverage"]["skipped_sources"]) == 2
    assert all("no fetcher" in s for s in meta["coverage"]["skipped_sources"])


def test_run_daily_records_resolver_warning_in_coverage(temp_db, monkeypatch, tmp_path):
    """A missing config produces a PlanWarning that surfaces as
    coverage.plan_warnings — the brief should be able to explain why the
    pool was thin."""
    monkeypatch.setattr(
        "clawfeed_intel.sources.DEFAULT_CONFIG_PATH",
        tmp_path / "absent.yaml",
    )
    with closing(worker_db.connect(temp_db)) as conn:
        digest_id = run_daily("24h", conn=conn)
        meta = json.loads(
            conn.execute("SELECT metadata FROM digests WHERE id = ?", (digest_id,)).fetchone()[
                "metadata"
            ]
        )

    assert meta["coverage"]["plan_warnings"], "expected at least one plan warning"
    assert any("not found" in w for w in meta["coverage"]["plan_warnings"])


def test_run_daily_invalid_window_raises_value_error(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(ValueError):
            run_daily("not-a-window", conn=conn)


def test_run_daily_marks_failure_on_db_error(temp_db, monkeypatch):
    """If a stage raises, the run row must end up in 'failed' state, not stuck mid-flow."""
    from clawfeed_intel.pipeline import orchestrator

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated stage failure")

    monkeypatch.setattr(orchestrator.db, "create_digest", boom)

    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(RuntimeError, match="simulated stage failure"):
            run_daily("24h", conn=conn)

        rows = conn.execute("SELECT * FROM intel_runs ORDER BY id DESC LIMIT 1").fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0]["finished_at"] is not None
        assert "simulated stage failure" in (rows[0]["error"] or "")
        assert rows[0]["digest_id"] is None
