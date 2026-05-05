from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from clawfeed_intel import db as worker_db

W_START = "2026-05-03T06:15:00+00:00"
W_END = "2026-05-04T06:15:00+00:00"


def _new_run(conn) -> int:
    return worker_db.create_run(conn, run_type="daily", window_start=W_START, window_end=W_END)


def _item_args(**overrides) -> dict:
    """Default raw-item kwargs; overrides merge in last."""
    base = dict(
        source_type="rss",
        dedup_key="https://example.com/article-1",
        title="Hello",
        url="https://example.com/article-1?utm_source=feed",
        canonical_url="https://example.com/article-1",
        content="Body of the article.",
    )
    base.update(overrides)
    return base


# ── upsert_raw_item ───────────────────────────────────────────────────────────


def test_first_upsert_returns_was_new_true(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        raw_id, was_new = worker_db.upsert_raw_item(conn, run_id=run_id, **_item_args())
        assert raw_id > 0
        assert was_new is True


def test_repeat_upsert_returns_same_id_was_new_false(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        a_id, a_new = worker_db.upsert_raw_item(conn, run_id=run_id, **_item_args())
        b_id, b_new = worker_db.upsert_raw_item(conn, run_id=run_id, **_item_args())
        assert a_id == b_id
        assert a_new is True
        assert b_new is False


def test_conflict_scoped_to_source_type(temp_db):
    """Same dedup_key under a different source_type yields a separate row."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        rss_id, _ = worker_db.upsert_raw_item(
            conn, run_id=run_id, **_item_args(source_type="rss", dedup_key="abc")
        )
        gdelt_id, was_new = worker_db.upsert_raw_item(
            conn, run_id=run_id, **_item_args(source_type="gdelt", dedup_key="abc")
        )
        assert rss_id != gdelt_id
        assert was_new is True


def test_upsert_links_to_run_raw_items(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        raw_id, _ = worker_db.upsert_raw_item(conn, run_id=run_id, **_item_args())
        rows = conn.execute(
            "SELECT * FROM run_raw_items WHERE run_id = ? AND raw_item_id = ?",
            (run_id, raw_id),
        ).fetchall()
        assert len(rows) == 1


def test_upsert_idempotent_link_within_same_run(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        raw_id, _ = worker_db.upsert_raw_item(conn, run_id=run_id, **_item_args())
        worker_db.upsert_raw_item(conn, run_id=run_id, **_item_args())
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM run_raw_items WHERE run_id = ? AND raw_item_id = ?",
            (run_id, raw_id),
        ).fetchone()["n"]
        assert count == 1


def test_upsert_across_runs_keeps_first_run_id_but_links_both(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        first_run = _new_run(conn)
        raw_id, _ = worker_db.upsert_raw_item(conn, run_id=first_run, **_item_args())

        second_run = _new_run(conn)
        same_id, was_new = worker_db.upsert_raw_item(conn, run_id=second_run, **_item_args())
        assert same_id == raw_id
        assert was_new is False

        # raw_items.run_id (first-sight) preserved.
        first_sight = conn.execute(
            "SELECT run_id FROM raw_items WHERE id = ?", (raw_id,)
        ).fetchone()["run_id"]
        assert first_sight == first_run

        # Both runs are linked through run_raw_items.
        linked_runs = sorted(
            row["run_id"]
            for row in conn.execute(
                "SELECT run_id FROM run_raw_items WHERE raw_item_id = ?", (raw_id,)
            ).fetchall()
        )
        assert linked_runs == sorted([first_run, second_run])


def test_upsert_persists_canonical_url_and_content_hash(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        raw_id, _ = worker_db.upsert_raw_item(
            conn, run_id=run_id, **_item_args(content_hash_value="deadbeef" * 8)
        )
        row = conn.execute(
            "SELECT canonical_url, content_hash, url, metadata, raw_payload "
            "FROM raw_items WHERE id = ?",
            (raw_id,),
        ).fetchone()
        assert row["canonical_url"] == "https://example.com/article-1"
        assert row["url"] == "https://example.com/article-1?utm_source=feed"
        assert row["content_hash"] == "deadbeef" * 8
        assert row["metadata"] == "{}"
        assert row["raw_payload"] == "{}"


def test_upsert_persists_optional_fields(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        raw_id, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="hn",
            dedup_key="42135123",
            title="Show HN: foo",
            url="https://news.ycombinator.com/item?id=42135123",
            canonical_url="https://news.ycombinator.com/item?id=42135123",
            content="discussion text",
            source_name="Hacker News (Top)",
            author="alice",
            excerpt="Quick excerpt",
            published_at="2026-05-04T06:00:00+00:00",
            metadata={"points": 247, "comments": 88},
            raw_payload={"raw": "fragment"},
        )
        row = conn.execute(
            "SELECT source_type, source_name, author, excerpt, published_at, "
            "metadata, raw_payload FROM raw_items WHERE id = ?",
            (raw_id,),
        ).fetchone()
        assert row["source_type"] == "hn"
        assert row["source_name"] == "Hacker News (Top)"
        assert row["author"] == "alice"
        assert row["excerpt"] == "Quick excerpt"
        assert row["published_at"] == "2026-05-04T06:00:00+00:00"
        assert '"points":247' in row["metadata"]
        assert '"comments":88' in row["metadata"]
        assert '"raw":"fragment"' in row["raw_payload"]


def test_upsert_rejects_blank_required_fields(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        with pytest.raises(ValueError):
            worker_db.upsert_raw_item(conn, run_id=run_id, **_item_args(source_type=""))
        with pytest.raises(ValueError):
            worker_db.upsert_raw_item(conn, run_id=run_id, **_item_args(dedup_key=""))


def test_upsert_rejects_unknown_run_id(temp_db):
    """run_id FK violation propagates as IntegrityError."""
    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            worker_db.upsert_raw_item(conn, run_id=999_999, **_item_args())


# ── link_raw_item_to_run ──────────────────────────────────────────────────────


def test_link_returns_false_when_already_linked(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        raw_id, _ = worker_db.upsert_raw_item(conn, run_id=run_id, **_item_args())
        # upsert_raw_item already linked this pair, so the explicit link is a no-op.
        assert worker_db.link_raw_item_to_run(conn, run_id=run_id, raw_item_id=raw_id) is False


def test_link_returns_true_for_new_run(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        first_run = _new_run(conn)
        raw_id, _ = worker_db.upsert_raw_item(conn, run_id=first_run, **_item_args())
        other_run = _new_run(conn)
        assert worker_db.link_raw_item_to_run(conn, run_id=other_run, raw_item_id=raw_id) is True


def test_link_to_unknown_raw_item_raises(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        with pytest.raises(sqlite3.IntegrityError):
            worker_db.link_raw_item_to_run(conn, run_id=run_id, raw_item_id=999_999)


def test_link_to_unknown_run_raises(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        raw_id, _ = worker_db.upsert_raw_item(conn, run_id=run_id, **_item_args())
        with pytest.raises(sqlite3.IntegrityError):
            worker_db.link_raw_item_to_run(conn, run_id=999_999, raw_item_id=raw_id)


# ── cascade behavior verifies our schema choices ──────────────────────────────


def test_run_delete_cascades_run_raw_items_but_keeps_raw_items(temp_db):
    """Deleting a run must remove its run_raw_items rows (CASCADE) but
    leave the raw_items rows intact (raw_items.run_id is SET NULL).
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        raw_id, _ = worker_db.upsert_raw_item(conn, run_id=run_id, **_item_args())

        with worker_db.transaction(conn):
            conn.execute("DELETE FROM intel_runs WHERE id = ?", (run_id,))

        item = conn.execute("SELECT id, run_id FROM raw_items WHERE id = ?", (raw_id,)).fetchone()
        assert item is not None, "raw_items row must survive run deletion"
        assert item["run_id"] is None, "raw_items.run_id must be set NULL"

        links = conn.execute(
            "SELECT 1 FROM run_raw_items WHERE raw_item_id = ?", (raw_id,)
        ).fetchall()
        assert links == []
