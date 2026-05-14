"""Tests for ``db.create_item_summary``.

The helper is the persistence half of step 10 — once 10b's
``summarize_clusters`` parses an LLM response into a
:class:`ClusterSummaryPayload`, this is the call that writes the row
the final composer (step 11) will read.

The narrative list fields (``entities`` / ``key_facts`` / ``caveats`` /
``source_urls``) land in TEXT columns as JSON strings; the round-trip
through ``json.dumps``/``json.loads`` is load-bearing for the composer
to read them back as lists.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest

from clawfeed_intel import db as worker_db
from clawfeed_intel.llm.schemas import ClusterSummaryPayload


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_run(conn: sqlite3.Connection) -> int:
    return worker_db.create_run(
        conn,
        run_type="daily",
        window_start="2026-05-12T00:00:00+00:00",
        window_end="2026-05-13T00:00:00+00:00",
    )


def _seed_kept_cluster(conn: sqlite3.Connection) -> int:
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
    worker_db.update_cluster_verdict(
        conn,
        cluster_id=cluster_id,
        status="kept",
        relevance_score=0.8,
        category="startup_funding",
        event_type="funding_round",
        filter_reason="Substantive announcement.",
    )
    return cluster_id


def _full_payload() -> ClusterSummaryPayload:
    return ClusterSummaryPayload(
        headline="Anthropic closes $500M Series E",
        summary=(
            "Anthropic announced a $500M Series E led by GeneralCo. "
            "Proceeds fund continued model training."
        ),
        why_it_matters="Largest AI-lab financing of the week.",
        entities=["Anthropic", "GeneralCo"],
        key_facts=["$500M raise", "Series E", "Led by GeneralCo"],
        caveats=["Valuation not disclosed."],
        source_urls=[
            "https://techcrunch.com/anthropic-series-e",
            "https://example.com/sec-form-d",
        ],
        confidence=0.85,
    )


def _minimal_payload() -> ClusterSummaryPayload:
    return ClusterSummaryPayload(
        headline="Headline",
        summary="One factual sentence.",
    )


# ── Happy paths ───────────────────────────────────────────────────────────────


def test_full_payload_round_trips(temp_db) -> None:
    """Every field on the payload maps to its column with the right type."""
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_kept_cluster(conn)
        payload = _full_payload()

        summary_id = worker_db.create_item_summary(
            conn,
            cluster_id=cluster_id,
            model="mlx-community/Qwen3.5-27B-4bit",
            prompt_version="summary.v1",
            payload=payload,
        )

        assert summary_id > 0
        row = conn.execute("SELECT * FROM item_summaries WHERE id = ?", (summary_id,)).fetchone()
        assert row["cluster_id"] == cluster_id
        assert row["model"] == "mlx-community/Qwen3.5-27B-4bit"
        assert row["prompt_version"] == "summary.v1"
        assert row["headline"].startswith("Anthropic")
        assert row["summary"].startswith("Anthropic announced")
        assert row["why_it_matters"] == "Largest AI-lab financing of the week."
        assert row["confidence"] == 0.85
        assert json.loads(row["entities"]) == ["Anthropic", "GeneralCo"]
        assert json.loads(row["key_facts"]) == [
            "$500M raise",
            "Series E",
            "Led by GeneralCo",
        ]
        assert json.loads(row["caveats"]) == ["Valuation not disclosed."]
        assert json.loads(row["source_urls"]) == [
            "https://techcrunch.com/anthropic-series-e",
            "https://example.com/sec-form-d",
        ]


def test_minimal_payload_serializes_defaults(temp_db) -> None:
    """The 9c-lesson path: a payload with only ``headline`` and
    ``summary`` writes empty-string / empty-array / null defaults
    rather than blowing up on missing fields.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_kept_cluster(conn)

        worker_db.create_item_summary(
            conn,
            cluster_id=cluster_id,
            model="mlx-community/Qwen3.5-27B-4bit",
            prompt_version="summary.v1",
            payload=_minimal_payload(),
        )

        row = conn.execute(
            "SELECT * FROM item_summaries WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()
        assert row["why_it_matters"] == ""
        assert row["confidence"] is None
        assert json.loads(row["entities"]) == []
        assert json.loads(row["key_facts"]) == []
        assert json.loads(row["caveats"]) == []
        assert json.loads(row["source_urls"]) == []


def test_multiple_summaries_per_cluster_append(temp_db) -> None:
    """The helper is append-only — re-running step 10 against the same
    cluster (e.g. after a prompt version bump) adds a new row rather
    than upserting. Production-side, the orchestration filters by
    ``status='kept'`` so a cluster summarized once and advanced to
    ``'summarized'`` won't be re-processed; this guards the helper's
    contract independent of that filter.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_kept_cluster(conn)

        id_a = worker_db.create_item_summary(
            conn,
            cluster_id=cluster_id,
            model="mlx-community/Qwen3.5-27B-4bit",
            prompt_version="summary.v1",
            payload=_minimal_payload(),
        )
        id_b = worker_db.create_item_summary(
            conn,
            cluster_id=cluster_id,
            model="mlx-community/Qwen3.5-122B-A10B-4bit",
            prompt_version="summary.v2",
            payload=_minimal_payload(),
        )

        assert id_b > id_a
        rows = conn.execute(
            "SELECT id, model, prompt_version FROM item_summaries WHERE cluster_id = ? ORDER BY id",
            (cluster_id,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["model"] == "mlx-community/Qwen3.5-27B-4bit"
        assert rows[0]["prompt_version"] == "summary.v1"
        assert rows[1]["model"] == "mlx-community/Qwen3.5-122B-A10B-4bit"
        assert rows[1]["prompt_version"] == "summary.v2"


def test_unicode_round_trips_through_json_columns(temp_db) -> None:
    """Non-ASCII entity names (Asian company names, accented authors)
    must round-trip without corruption. ``json.dumps`` with the default
    ``ensure_ascii=True`` escapes them, but ``json.loads`` restores the
    original code points — verified end-to-end so a future
    ``ensure_ascii=False`` change is a deliberate decision, not silent.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_kept_cluster(conn)
        payload = ClusterSummaryPayload(
            headline="DeepSeek-V4 details emerge",
            summary="DeepSeek (深度求索) released DeepSeek-V4 weights.",
            entities=["深度求索", "DeepSeek-V4"],
            source_urls=["https://example.com/post"],
        )

        worker_db.create_item_summary(
            conn,
            cluster_id=cluster_id,
            model="mlx-community/Qwen3.5-27B-4bit",
            prompt_version="summary.v1",
            payload=payload,
        )

        row = conn.execute(
            "SELECT entities, summary FROM item_summaries WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
        assert json.loads(row["entities"]) == ["深度求索", "DeepSeek-V4"]
        assert "深度求索" in row["summary"]


# ── Validation ────────────────────────────────────────────────────────────────


def test_blank_model_rejected(temp_db) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_kept_cluster(conn)
        with pytest.raises(ValueError, match="model is required"):
            worker_db.create_item_summary(
                conn,
                cluster_id=cluster_id,
                model="",
                prompt_version="summary.v1",
                payload=_minimal_payload(),
            )


def test_blank_prompt_version_rejected(temp_db) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_kept_cluster(conn)
        with pytest.raises(ValueError, match="prompt_version is required"):
            worker_db.create_item_summary(
                conn,
                cluster_id=cluster_id,
                model="mlx-community/Qwen3.5-27B-4bit",
                prompt_version="",
                payload=_minimal_payload(),
            )


def test_missing_cluster_id_raises_integrity_error(temp_db) -> None:
    """The FK from ``item_summaries.cluster_id → item_clusters.id``
    catches the dangling-id case so a future caller can't silently
    land an orphan row.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            worker_db.create_item_summary(
                conn,
                cluster_id=99_999,
                model="mlx-community/Qwen3.5-27B-4bit",
                prompt_version="summary.v1",
                payload=_minimal_payload(),
            )


def test_cluster_cascade_deletes_summaries(temp_db) -> None:
    """``ON DELETE CASCADE`` on ``cluster_id`` (migration 010 line 113)
    means a cluster's summaries vanish with it — load-bearing for the
    retention path so deleting an old run's data doesn't leave orphan
    summaries behind.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        cluster_id = _seed_kept_cluster(conn)
        worker_db.create_item_summary(
            conn,
            cluster_id=cluster_id,
            model="mlx-community/Qwen3.5-27B-4bit",
            prompt_version="summary.v1",
            payload=_minimal_payload(),
        )
        before = conn.execute(
            "SELECT COUNT(*) AS n FROM item_summaries WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()["n"]
        assert before == 1

        conn.execute("DELETE FROM item_clusters WHERE id = ?", (cluster_id,))
        conn.commit()

        after = conn.execute(
            "SELECT COUNT(*) AS n FROM item_summaries WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()["n"]
        assert after == 0
