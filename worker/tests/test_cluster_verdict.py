"""Tests for ``db.update_cluster_verdict``.

The verdict helper is the persistence half of step 9 — once 9b's
``filter_clusters`` parses an LLM response, this is the call that
promotes the cluster row from ``pending`` to ``kept`` / ``filtered_out``
and records the LLM's judgement fields.

All exercises run against a fresh ``temp_db`` so the SQL CHECK on
``item_clusters.status`` is the same shape we'll see in production.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from clawfeed_intel import db as worker_db


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_run(conn: sqlite3.Connection) -> int:
    return worker_db.create_run(
        conn,
        run_type="daily",
        window_start="2026-05-10T00:00:00+00:00",
        window_end="2026-05-11T00:00:00+00:00",
    )


def _seed_pending_cluster(conn: sqlite3.Connection) -> int:
    run_id = _make_run(conn)
    raw_item_id, _ = worker_db.upsert_raw_item(
        conn,
        run_id=run_id,
        source_type="rss",
        dedup_key="seed-1",
        title="Seed item",
        url="https://example.com/a",
        canonical_url="https://example.com/a",
        content="",
    )
    cluster_id, _ = worker_db.create_cluster(
        conn,
        run_id=run_id,
        cluster_key="https://example.com/a",
        title="Seed item",
        raw_item_ids=[raw_item_id],
    )
    return cluster_id


# ── Happy paths ───────────────────────────────────────────────────────────────


def test_kept_verdict_writes_all_fields(temp_db) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_pending_cluster(conn)

        worker_db.update_cluster_verdict(
            conn,
            cluster_id=cluster_id,
            status="kept",
            relevance_score=0.82,
            category="startup_funding",
            event_type="funding_round",
            filter_reason="Substantive Series B announcement.",
        )

        row = conn.execute(
            "SELECT status, relevance_score, category, event_type, filter_reason "
            "FROM item_clusters WHERE id = ?",
            (cluster_id,),
        ).fetchone()
        assert row["status"] == "kept"
        assert row["relevance_score"] == 0.82
        assert row["category"] == "startup_funding"
        assert row["event_type"] == "funding_round"
        assert row["filter_reason"] == "Substantive Series B announcement."


def test_filtered_out_verdict_writes_all_fields(temp_db) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_pending_cluster(conn)

        worker_db.update_cluster_verdict(
            conn,
            cluster_id=cluster_id,
            status="filtered_out",
            relevance_score=0.15,
            category="ai_research",
            event_type=None,
            filter_reason="Marginal benchmark delta.",
        )

        row = conn.execute(
            "SELECT status, relevance_score, category, event_type, filter_reason "
            "FROM item_clusters WHERE id = ?",
            (cluster_id,),
        ).fetchone()
        assert row["status"] == "filtered_out"
        assert row["relevance_score"] == 0.15
        assert row["event_type"] is None


def test_filtered_out_verdict_accepts_null_category_and_reason(temp_db) -> None:
    """Local models reliably emit ``null`` for category and reason on
    rejected verdicts (caught during the first live-vMLX smoke). The DB
    columns are nullable; the helper must accept ``None`` without raising.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_pending_cluster(conn)

        worker_db.update_cluster_verdict(
            conn,
            cluster_id=cluster_id,
            status="filtered_out",
            relevance_score=0.05,
            category=None,
            event_type=None,
            filter_reason=None,
        )

        row = conn.execute(
            "SELECT status, category, filter_reason FROM item_clusters WHERE id = ?",
            (cluster_id,),
        ).fetchone()
        assert row["status"] == "filtered_out"
        assert row["category"] is None
        assert row["filter_reason"] is None


def test_idempotent_on_repeated_apply(temp_db) -> None:
    """Re-applying the same verdict is a clean no-op modulo timestamps."""
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_pending_cluster(conn)

        for _ in range(2):
            worker_db.update_cluster_verdict(
                conn,
                cluster_id=cluster_id,
                status="kept",
                relevance_score=0.7,
                category="ai_research",
                event_type="model_release",
                filter_reason="Open-weight checkpoint released.",
            )

        row = conn.execute(
            "SELECT status, category, relevance_score FROM item_clusters WHERE id = ?",
            (cluster_id,),
        ).fetchone()
        assert row["status"] == "kept"
        assert row["category"] == "ai_research"
        assert row["relevance_score"] == 0.7


def test_verdict_overwrites_prior_decision(temp_db) -> None:
    """A re-run of the filter must be able to flip a verdict — the
    idempotency contract is about replay, not append-only history.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_pending_cluster(conn)

        worker_db.update_cluster_verdict(
            conn,
            cluster_id=cluster_id,
            status="filtered_out",
            relevance_score=0.2,
            category="ai_research",
            event_type=None,
            filter_reason="Noisy preprint.",
        )
        worker_db.update_cluster_verdict(
            conn,
            cluster_id=cluster_id,
            status="kept",
            relevance_score=0.9,
            category="ai_research",
            event_type="paper",
            filter_reason="Reclassified after re-prompt.",
        )

        row = conn.execute(
            "SELECT status, relevance_score, event_type FROM item_clusters WHERE id = ?",
            (cluster_id,),
        ).fetchone()
        assert row["status"] == "kept"
        assert row["relevance_score"] == 0.9
        assert row["event_type"] == "paper"


# ── Boundary validation ───────────────────────────────────────────────────────


def test_pending_status_rejected_at_boundary(temp_db) -> None:
    """The relevance filter must not push a cluster back to ``pending`` —
    that would invalidate prior verdicts on a re-run.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_pending_cluster(conn)
        with pytest.raises(ValueError, match="invalid verdict status"):
            worker_db.update_cluster_verdict(
                conn,
                cluster_id=cluster_id,
                status="pending",
                relevance_score=0.5,
                category="x",
                event_type=None,
                filter_reason="should not apply",
            )


def test_summarized_status_rejected_at_boundary(temp_db) -> None:
    """``summarized`` is owned by the cluster-summary stage (step 10), not
    the relevance filter.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_pending_cluster(conn)
        with pytest.raises(ValueError, match="invalid verdict status"):
            worker_db.update_cluster_verdict(
                conn,
                cluster_id=cluster_id,
                status="summarized",
                relevance_score=0.5,
                category="x",
                event_type=None,
                filter_reason="should not apply",
            )


def test_typo_status_rejected_at_boundary(temp_db) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_pending_cluster(conn)
        with pytest.raises(ValueError, match="invalid verdict status"):
            worker_db.update_cluster_verdict(
                conn,
                cluster_id=cluster_id,
                status="keep",  # typo for "kept"
                relevance_score=0.5,
                category="x",
                event_type=None,
                filter_reason="x",
            )


def test_missing_cluster_id_raises_lookup_error(temp_db) -> None:
    """A vanished cluster surfaces loudly — silent no-op would mask bugs."""
    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(LookupError, match="row 9999 not found"):
            worker_db.update_cluster_verdict(
                conn,
                cluster_id=9999,
                status="kept",
                relevance_score=0.5,
                category="x",
                event_type=None,
                filter_reason="x",
            )


def test_verdict_does_not_touch_unrelated_cluster(temp_db) -> None:
    """Sibling clusters must not be affected by a verdict applied to one."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        rid_1, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="rss",
            dedup_key="one",
            title="One",
            url="https://example.com/1",
            canonical_url="https://example.com/1",
            content="",
        )
        rid_2, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="rss",
            dedup_key="two",
            title="Two",
            url="https://example.com/2",
            canonical_url="https://example.com/2",
            content="",
        )
        cluster_1, _ = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://example.com/1",
            title="One",
            raw_item_ids=[rid_1],
        )
        cluster_2, _ = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://example.com/2",
            title="Two",
            raw_item_ids=[rid_2],
        )

        worker_db.update_cluster_verdict(
            conn,
            cluster_id=cluster_1,
            status="kept",
            relevance_score=0.9,
            category="ai_research",
            event_type=None,
            filter_reason="Reason A.",
        )

        row = conn.execute(
            "SELECT status, relevance_score FROM item_clusters WHERE id = ?",
            (cluster_2,),
        ).fetchone()
        assert row["status"] == "pending"
        assert row["relevance_score"] is None
