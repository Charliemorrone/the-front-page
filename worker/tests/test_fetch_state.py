"""Tests for source_fetch_state success/failure helpers.

These two helpers are the only path the runner has to record cursor /
last-success / consecutive-error state on the row level. The contract:

- ``record_fetch_success`` resets ``consecutive_errors`` to 0 and clears
  ``last_error`` whether the row exists or not.
- ``record_fetch_failure`` bumps ``consecutive_errors`` by 1 each call
  and never touches ``last_success_at`` or ``cursor`` — a stretch of
  failures keeps the last-known-good cursor reachable.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest

from clawfeed_intel import db as worker_db


def _add_source(conn: sqlite3.Connection, name: str = "rss-x", type_: str = "rss") -> int:
    cur = conn.execute(
        "INSERT INTO sources (name, type, config, is_active) VALUES (?, ?, ?, 1)",
        (name, type_, "{}"),
    )
    return int(cur.lastrowid)


def test_record_fetch_success_creates_row(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        sid = _add_source(conn)
        worker_db.record_fetch_success(conn, source_id=sid, fetcher="rss")
        row = conn.execute(
            "SELECT * FROM source_fetch_state WHERE source_id = ? AND fetcher = ?",
            (sid, "rss"),
        ).fetchone()
        assert row is not None
        assert row["last_success_at"] is not None
        assert row["last_attempt_at"] is not None
        assert row["last_error"] is None
        assert row["consecutive_errors"] == 0
        assert row["cursor"] is None
        assert row["metadata"] == "{}"


def test_record_fetch_failure_creates_row_and_increments(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        sid = _add_source(conn)
        worker_db.record_fetch_failure(conn, source_id=sid, fetcher="rss", error="timeout")
        worker_db.record_fetch_failure(conn, source_id=sid, fetcher="rss", error="timeout")
        worker_db.record_fetch_failure(conn, source_id=sid, fetcher="rss", error="500 server")

        row = conn.execute(
            "SELECT * FROM source_fetch_state WHERE source_id = ? AND fetcher = ?",
            (sid, "rss"),
        ).fetchone()
        assert row["consecutive_errors"] == 3
        assert row["last_error"] == "500 server"
        assert row["last_success_at"] is None


def test_success_resets_consecutive_errors(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        sid = _add_source(conn)
        for _ in range(3):
            worker_db.record_fetch_failure(conn, source_id=sid, fetcher="rss", error="boom")

        worker_db.record_fetch_success(conn, source_id=sid, fetcher="rss")

        row = conn.execute(
            "SELECT * FROM source_fetch_state WHERE source_id = ? AND fetcher = ?",
            (sid, "rss"),
        ).fetchone()
        assert row["consecutive_errors"] == 0
        assert row["last_error"] is None
        assert row["last_success_at"] is not None


def test_failure_preserves_cursor_and_last_success(temp_db):
    """Cursor and last_success_at survive subsequent failures so we don't
    lose the last-known-good resumption point during a flaky stretch."""
    with closing(worker_db.connect(temp_db)) as conn:
        sid = _add_source(conn)
        worker_db.record_fetch_success(
            conn, source_id=sid, fetcher="rss", cursor="2026-05-04T06:00:00Z"
        )
        first_success = conn.execute(
            "SELECT last_success_at FROM source_fetch_state WHERE source_id = ?",
            (sid,),
        ).fetchone()["last_success_at"]

        worker_db.record_fetch_failure(conn, source_id=sid, fetcher="rss", error="boom")

        row = conn.execute(
            "SELECT * FROM source_fetch_state WHERE source_id = ?",
            (sid,),
        ).fetchone()
        assert row["cursor"] == "2026-05-04T06:00:00Z"
        assert row["last_success_at"] == first_success
        assert row["consecutive_errors"] == 1


def test_success_preserves_existing_cursor_when_omitted(temp_db):
    """Calling record_fetch_success without a cursor must not wipe the cursor
    a previous success wrote — useful for fetchers that only set cursor on
    some calls (e.g. when there were new items)."""
    with closing(worker_db.connect(temp_db)) as conn:
        sid = _add_source(conn)
        worker_db.record_fetch_success(conn, source_id=sid, fetcher="rss", cursor="cursor-1")
        worker_db.record_fetch_success(conn, source_id=sid, fetcher="rss")

        row = conn.execute(
            "SELECT cursor FROM source_fetch_state WHERE source_id = ?",
            (sid,),
        ).fetchone()
        assert row["cursor"] == "cursor-1"


def test_success_replaces_metadata_only_when_supplied(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        sid = _add_source(conn)
        worker_db.record_fetch_success(conn, source_id=sid, fetcher="rss", metadata={"etag": "v1"})
        worker_db.record_fetch_success(conn, source_id=sid, fetcher="rss")
        row = conn.execute(
            "SELECT metadata FROM source_fetch_state WHERE source_id = ?",
            (sid,),
        ).fetchone()
        assert json.loads(row["metadata"]) == {"etag": "v1"}

        worker_db.record_fetch_success(conn, source_id=sid, fetcher="rss", metadata={"etag": "v2"})
        row = conn.execute(
            "SELECT metadata FROM source_fetch_state WHERE source_id = ?",
            (sid,),
        ).fetchone()
        assert json.loads(row["metadata"]) == {"etag": "v2"}


def test_state_is_per_fetcher(temp_db):
    """source_fetch_state PK is (source_id, fetcher) — different fetchers on
    the same source must not collide."""
    with closing(worker_db.connect(temp_db)) as conn:
        sid = _add_source(conn)
        worker_db.record_fetch_success(conn, source_id=sid, fetcher="rss")
        worker_db.record_fetch_failure(conn, source_id=sid, fetcher="website", error="boom")
        rows = conn.execute(
            "SELECT fetcher, consecutive_errors, last_error FROM source_fetch_state "
            "WHERE source_id = ? ORDER BY fetcher",
            (sid,),
        ).fetchall()
        assert len(rows) == 2
        by_fetcher = {r["fetcher"]: r for r in rows}
        assert by_fetcher["rss"]["consecutive_errors"] == 0
        assert by_fetcher["rss"]["last_error"] is None
        assert by_fetcher["website"]["consecutive_errors"] == 1
        assert by_fetcher["website"]["last_error"] == "boom"


def test_unknown_source_id_raises_foreign_key(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            worker_db.record_fetch_success(conn, source_id=999999, fetcher="rss")
