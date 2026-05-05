from __future__ import annotations

import json
from contextlib import closing

import pytest

from clawfeed_intel import db as worker_db

W_START = "2026-05-03T06:15:00+00:00"
W_END = "2026-05-04T06:15:00+00:00"


def _new_run(conn, **overrides) -> int:
    kwargs = dict(run_type="daily", window_start=W_START, window_end=W_END)
    kwargs.update(overrides)
    return worker_db.create_run(conn, **kwargs)


def test_create_run_defaults_to_pending(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        row = worker_db.get_run(conn, run_id)
        assert row is not None
        assert row["status"] == "pending"
        assert row["run_type"] == "daily"
        assert row["window_start"] == W_START
        assert row["window_end"] == W_END
        assert row["started_at"] is None
        assert row["finished_at"] is None
        assert row["digest_id"] is None
        assert json.loads(row["metadata"]) == {}


def test_invalid_run_type_rejected(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(ValueError):
            worker_db.create_run(
                conn,
                run_type="weekly",
                window_start=W_START,
                window_end=W_END,
            )


def test_lifecycle_happy_path(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        worker_db.mark_run_started(conn, run_id)

        row = worker_db.get_run(conn, run_id)
        assert row["status"] == "fetching"
        assert row["started_at"] is not None

        worker_db.advance_run_status(conn, run_id, "filtering")
        worker_db.advance_run_status(conn, run_id, "summarizing")
        worker_db.advance_run_status(conn, run_id, "composing")

        digest_id = worker_db.create_digest(
            conn,
            digest_type="daily",
            content="# stub",
            metadata={"k": 1},
        )
        worker_db.finish_run(conn, run_id, status="published", digest_id=digest_id)

        row = worker_db.get_run(conn, run_id)
        assert row["status"] == "published"
        assert row["digest_id"] == digest_id
        assert row["finished_at"] is not None
        assert row["error"] is None


def test_double_start_raises_run_state_error(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        worker_db.mark_run_started(conn, run_id)
        with pytest.raises(worker_db.RunStateError):
            worker_db.mark_run_started(conn, run_id)


def test_advance_from_pending_rejected(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        with pytest.raises(worker_db.RunStateError):
            worker_db.advance_run_status(conn, run_id, "filtering")


def test_advance_from_terminal_rejected(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        worker_db.mark_run_started(conn, run_id)
        worker_db.finish_run(conn, run_id, status="failed", error="boom")
        with pytest.raises(worker_db.RunStateError):
            worker_db.advance_run_status(conn, run_id, "filtering")


def test_advance_to_invalid_status_raises_value_error(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        worker_db.mark_run_started(conn, run_id)
        with pytest.raises(ValueError):
            worker_db.advance_run_status(conn, run_id, "published")


def test_finish_with_non_terminal_status_rejected(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        worker_db.mark_run_started(conn, run_id)
        with pytest.raises(ValueError):
            worker_db.finish_run(conn, run_id, status="filtering")


def test_finish_unknown_run_raises(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(worker_db.RunStateError):
            worker_db.finish_run(conn, run_id=9999, status="failed", error="nope")


def test_create_digest_rejects_unknown_type(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(ValueError):
            worker_db.create_digest(conn, digest_type="topic", content="x", metadata={})


def test_create_digest_persists_metadata_json(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        digest_id = worker_db.create_digest(
            conn,
            digest_type="daily",
            content="# hello",
            metadata={"brief_kind": "daily", "coverage": {"raw_items": 0}},
        )
        row = conn.execute("SELECT * FROM digests WHERE id = ?", (digest_id,)).fetchone()
        assert row["type"] == "daily"
        assert row["content"] == "# hello"
        meta = json.loads(row["metadata"])
        assert meta["brief_kind"] == "daily"
        assert meta["coverage"]["raw_items"] == 0


def test_update_run_metadata_replaces_blob(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn, metadata={"original": True})
        worker_db.update_run_metadata(conn, run_id, {"replaced": True})
        row = worker_db.get_run(conn, run_id)
        assert json.loads(row["metadata"]) == {"replaced": True}


def test_transaction_rolls_back_on_error(temp_db):
    """A raised exception inside transaction() must not leave a partial write."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _new_run(conn)
        before = worker_db.get_run(conn, run_id)["status"]
        with pytest.raises(RuntimeError):
            with worker_db.transaction(conn):
                conn.execute(
                    "UPDATE intel_runs SET status = 'fetching' WHERE id = ?",
                    (run_id,),
                )
                raise RuntimeError("simulate stage failure")
        after = worker_db.get_run(conn, run_id)["status"]
        assert before == after == "pending"
