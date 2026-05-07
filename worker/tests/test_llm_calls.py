"""Tests for the ``llm_calls`` audit-log helper."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from clawfeed_intel import db


@pytest.fixture
def conn(temp_db: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection on the migrated temp DB and clean up after each test."""
    with closing(db.connect(temp_db)) as c:
        yield c


def _create_run(conn: sqlite3.Connection) -> int:
    return db.create_run(
        conn,
        run_type="daily",
        window_start="2026-05-06T00:00:00+00:00",
        window_end="2026-05-07T00:00:00+00:00",
    )


def _row(conn: sqlite3.Connection, call_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM llm_calls WHERE id = ?", (call_id,)).fetchone()
    assert row is not None
    return row


# ── Happy path ────────────────────────────────────────────────────────────


def test_record_succeeded_call_full_fields(conn: sqlite3.Connection) -> None:
    run_id = _create_run(conn)
    call_id = db.record_llm_call(
        conn,
        stage="relevance_filter",
        provider="vmlx",
        model="mlx-community/Qwen3.5-27B-4bit",
        status="succeeded",
        latency_ms=1234,
        run_id=run_id,
        prompt_version="rel-v1",
        input_hash="a" * 64,
        output_hash="b" * 64,
        prompt_tokens=120,
        completion_tokens=80,
    )

    row = _row(conn, call_id)
    assert row["status"] == "succeeded"
    assert row["stage"] == "relevance_filter"
    assert row["provider"] == "vmlx"
    assert row["model"] == "mlx-community/Qwen3.5-27B-4bit"
    assert row["run_id"] == run_id
    assert row["prompt_version"] == "rel-v1"
    assert row["input_hash"] == "a" * 64
    assert row["output_hash"] == "b" * 64
    assert row["latency_ms"] == 1234
    assert row["prompt_tokens"] == 120
    assert row["completion_tokens"] == 80
    assert row["error"] is None


def test_record_failed_call_with_error(conn: sqlite3.Connection) -> None:
    run_id = _create_run(conn)
    call_id = db.record_llm_call(
        conn,
        stage="source_planning",
        provider="vmlx",
        model="mlx-community/Qwen3-8B-4bit",
        status="failed",
        latency_ms=500,
        run_id=run_id,
        input_hash="c" * 64,
        error="HTTPStatusError: 500 boom",
    )
    row = _row(conn, call_id)
    assert row["status"] == "failed"
    assert row["error"] == "HTTPStatusError: 500 boom"
    assert row["output_hash"] is None
    assert row["prompt_tokens"] == 0
    assert row["completion_tokens"] == 0


def test_run_id_optional(conn: sqlite3.Connection) -> None:
    """Background calls (e.g. doctor probe) may have no associated run."""
    call_id = db.record_llm_call(
        conn,
        stage="doctor_probe",
        provider="vmlx",
        model="m",
        status="succeeded",
        latency_ms=10,
    )
    row = _row(conn, call_id)
    assert row["run_id"] is None


def test_appends_one_row_per_call(conn: sqlite3.Connection) -> None:
    """Each call produces a fresh row — the audit log is append-only."""
    run_id = _create_run(conn)
    for i in range(3):
        db.record_llm_call(
            conn,
            stage="cluster_summary",
            provider="vmlx",
            model="m",
            status="succeeded",
            latency_ms=i * 100,
            run_id=run_id,
        )
    count = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    assert count == 3


# ── Validation surface ────────────────────────────────────────────────────


def test_invalid_status_rejected_at_python_boundary(
    conn: sqlite3.Connection,
) -> None:
    """Catch typos in Python before round-tripping to SQLite's CHECK."""
    with pytest.raises(ValueError, match="invalid llm_calls status"):
        db.record_llm_call(
            conn,
            stage="x",
            provider="vmlx",
            model="m",
            status="success",  # typo: should be 'succeeded'
            latency_ms=0,
        )


def test_invalid_status_caught_by_sql_check(conn: sqlite3.Connection) -> None:
    """If the Python guard is bypassed, SQL CHECK enforces it."""
    # Direct SQL bypassing the helper to verify the constraint exists.
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute(
                "INSERT INTO llm_calls (stage, provider, model, latency_ms, status) "
                "VALUES (?, ?, ?, ?, ?)",
                ("x", "vmlx", "m", 0, "made-up"),
            )


@pytest.mark.parametrize(
    "field,value",
    [
        ("stage", ""),
        ("provider", ""),
        ("model", ""),
    ],
)
def test_required_string_fields_rejected_when_blank(
    conn: sqlite3.Connection, field: str, value: str
) -> None:
    kwargs = {
        "stage": "x",
        "provider": "vmlx",
        "model": "m",
        "status": "succeeded",
        "latency_ms": 0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        db.record_llm_call(conn, **kwargs)


def test_run_id_fk_set_to_null_on_run_delete(conn: sqlite3.Connection) -> None:
    """Schema declares ``ON DELETE SET NULL`` so deleting a run preserves
    the audit row but unlinks it. Verify the constraint behaves."""
    run_id = _create_run(conn)
    call_id = db.record_llm_call(
        conn,
        stage="x",
        provider="vmlx",
        model="m",
        status="succeeded",
        latency_ms=0,
        run_id=run_id,
    )
    with conn:
        conn.execute("DELETE FROM intel_runs WHERE id = ?", (run_id,))
    row = _row(conn, call_id)
    assert row["run_id"] is None
