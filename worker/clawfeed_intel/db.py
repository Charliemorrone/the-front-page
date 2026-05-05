"""SQLite access layer for the worker.

The same database file is shared with the ClawFeed Node server. Both runtimes
open it in WAL mode, so concurrent readers and a single writer are safe under
brief contention.

All write operations use ``BEGIN IMMEDIATE`` inside :func:`transaction` to
acquire the write lock up-front; SQLite's per-connection ``busy_timeout`` (set
in :func:`connect`) handles transient contention without busy-looping in
Python. If the timeout elapses, ``OperationalError`` propagates and the caller
decides the retry policy.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import DB_PATH

RUN_TYPES: frozenset[str] = frozenset({"daily", "topic"})

RUN_STATUSES: frozenset[str] = frozenset(
    {
        "pending",
        "fetching",
        "filtering",
        "summarizing",
        "composing",
        "published",
        "failed",
        "cancelled",
    }
)

INTERMEDIATE_RUN_STATUSES: frozenset[str] = frozenset({"filtering", "summarizing", "composing"})

TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"published", "failed", "cancelled"})

# digests.type is constrained by the ClawFeed schema (migration 001).
# Topic briefs piggyback on 'daily' until the constraint is relaxed in a later phase.
DIGEST_TYPES: frozenset[str] = frozenset({"4h", "daily", "weekly", "monthly"})

# How long SQLite waits for the write lock before returning SQLITE_BUSY.
# 5s is comfortable for a single-machine personal system with at most one
# concurrent writer (the Node server) holding the lock for a few ms at a time.
_BUSY_TIMEOUT_MS = 5000


class RunStateError(RuntimeError):
    """Raised when a run-state transition is invalid."""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open a worker SQLite connection.

    The connection is in autocommit mode so :func:`transaction` can manage
    ``BEGIN IMMEDIATE`` / ``COMMIT`` / ``ROLLBACK`` explicitly. Foreign keys
    are enabled (SQLite defaults to off). Rows are returned as
    :class:`sqlite3.Row` for dict-style access by column name.
    """
    db_path = Path(path) if path is not None else DB_PATH
    conn = sqlite3.connect(
        db_path,
        isolation_level=None,
        timeout=_BUSY_TIMEOUT_MS / 1000.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a write transaction with ``BEGIN IMMEDIATE``.

    Commits on clean exit, rolls back on any exception (including
    :class:`KeyboardInterrupt`). Keep transactions short so the Node server is
    not blocked on writes for long.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _utc_now_iso() -> str:
    """Timezone-aware UTC timestamp matching the ``YYYY-MM-DDTHH:MM:SS+00:00`` shape."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dump_metadata(metadata: dict[str, Any] | None) -> str:
    return json.dumps(metadata or {}, separators=(",", ":"), sort_keys=True)


# ── intel_runs ────────────────────────────────────────────────────────────────


def create_run(
    conn: sqlite3.Connection,
    *,
    run_type: str,
    window_start: str,
    window_end: str,
    query: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Insert a new ``intel_runs`` row in ``pending`` state. Returns the run id."""
    if run_type not in RUN_TYPES:
        raise ValueError(f"invalid run_type {run_type!r}; must be one of {sorted(RUN_TYPES)}")
    metadata_json = _dump_metadata(metadata)
    with transaction(conn):
        cur = conn.execute(
            """
            INSERT INTO intel_runs (run_type, query, window_start, window_end, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_type, query, window_start, window_end, metadata_json),
        )
        return int(cur.lastrowid)


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM intel_runs WHERE id = ?", (run_id,)).fetchone()


def mark_run_started(conn: sqlite3.Connection, run_id: int) -> None:
    """Transition ``pending`` → ``fetching`` and stamp ``started_at``."""
    with transaction(conn):
        cur = conn.execute(
            """
            UPDATE intel_runs
               SET status = 'fetching',
                   started_at = ?
             WHERE id = ? AND status = 'pending'
            """,
            (_utc_now_iso(), run_id),
        )
        if cur.rowcount == 0:
            raise RunStateError(f"run {run_id}: cannot start (not found or not in 'pending' state)")


def advance_run_status(
    conn: sqlite3.Connection,
    run_id: int,
    new_status: str,
) -> None:
    """Move an in-flight run forward through a non-terminal status.

    Allowed targets are ``filtering``, ``summarizing``, ``composing``. The
    update will not fire if the run is in ``pending`` or any terminal state,
    which catches misordered orchestrator calls early.
    """
    if new_status not in INTERMEDIATE_RUN_STATUSES:
        raise ValueError(
            f"advance_run_status: {new_status!r} is not an intermediate status; "
            f"expected one of {sorted(INTERMEDIATE_RUN_STATUSES)}"
        )
    with transaction(conn):
        cur = conn.execute(
            """
            UPDATE intel_runs
               SET status = ?
             WHERE id = ?
               AND status NOT IN ('pending', 'published', 'failed', 'cancelled')
            """,
            (new_status, run_id),
        )
        if cur.rowcount == 0:
            raise RunStateError(
                f"run {run_id}: cannot advance to {new_status!r} "
                f"(run not found, still pending, or already terminal)"
            )


def update_run_metadata(
    conn: sqlite3.Connection,
    run_id: int,
    metadata: dict[str, Any],
) -> None:
    """Replace the metadata blob on a run. Caller is responsible for the merge."""
    metadata_json = _dump_metadata(metadata)
    with transaction(conn):
        cur = conn.execute(
            "UPDATE intel_runs SET metadata = ? WHERE id = ?",
            (metadata_json, run_id),
        )
        if cur.rowcount == 0:
            raise RunStateError(f"run {run_id}: not found (update_run_metadata)")


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: str,
    digest_id: int | None = None,
    error: str | None = None,
) -> None:
    """Move a run to a terminal status and stamp ``finished_at``."""
    if status not in TERMINAL_RUN_STATUSES:
        raise ValueError(
            f"finish_run requires a terminal status, got {status!r}; "
            f"expected one of {sorted(TERMINAL_RUN_STATUSES)}"
        )
    with transaction(conn):
        cur = conn.execute(
            """
            UPDATE intel_runs
               SET status = ?,
                   finished_at = ?,
                   digest_id = COALESCE(?, digest_id),
                   error = ?
             WHERE id = ?
            """,
            (status, _utc_now_iso(), digest_id, error, run_id),
        )
        if cur.rowcount == 0:
            raise RunStateError(f"run {run_id}: not found (finish_run)")


# ── digests ───────────────────────────────────────────────────────────────────


def create_digest(
    conn: sqlite3.Connection,
    *,
    digest_type: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Insert a finished brief into ClawFeed's ``digests`` table."""
    if digest_type not in DIGEST_TYPES:
        raise ValueError(
            f"invalid digest type {digest_type!r}; must be one of {sorted(DIGEST_TYPES)}"
        )
    metadata_json = _dump_metadata(metadata)
    with transaction(conn):
        cur = conn.execute(
            "INSERT INTO digests (type, content, metadata) VALUES (?, ?, ?)",
            (digest_type, content, metadata_json),
        )
        return int(cur.lastrowid)


# ── raw_items ─────────────────────────────────────────────────────────────────


def upsert_raw_item(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    source_type: str,
    dedup_key: str,
    title: str,
    url: str,
    canonical_url: str,
    content: str,
    source_id: int | None = None,
    source_name: str | None = None,
    author: str = "",
    excerpt: str = "",
    published_at: str | None = None,
    content_hash_value: str | None = None,
    metadata: dict[str, Any] | None = None,
    raw_payload: dict[str, Any] | None = None,
) -> tuple[int, bool]:
    """Insert a raw item idempotently and link it to *run_id* atomically.

    On a ``(source_type, dedup_key)`` conflict the existing row is left
    untouched — fetchers must not silently overwrite a richer earlier capture.
    The ``(run_id, raw_item_id)`` pair is inserted into ``run_raw_items``
    regardless, so the current run sees the item even when discovered earlier
    (which matters for topical search reusing the daily run's cache).

    Both writes happen inside a single ``BEGIN IMMEDIATE`` transaction.

    Returns:
        ``(raw_item_id, was_new)`` where ``was_new`` is ``True`` iff this was
        the first sighting of the ``(source_type, dedup_key)`` pair.

    Raises:
        ValueError: if ``source_type`` or ``dedup_key`` is empty.
        sqlite3.IntegrityError: if a referenced ``run_id`` does not exist.
    """
    if not source_type:
        raise ValueError("source_type is required")
    if not dedup_key:
        raise ValueError("dedup_key is required")

    metadata_json = _dump_metadata(metadata)
    raw_payload_json = _dump_metadata(raw_payload)

    with transaction(conn):
        cur = conn.execute(
            """
            INSERT INTO raw_items (
                source_id, run_id, source_type, source_name,
                title, url, canonical_url, author, content, excerpt,
                published_at, dedup_key, content_hash, metadata, raw_payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_type, dedup_key) DO NOTHING
            RETURNING id
            """,
            (
                source_id,
                run_id,
                source_type,
                source_name,
                title,
                url,
                canonical_url,
                author,
                content,
                excerpt,
                published_at,
                dedup_key,
                content_hash_value,
                metadata_json,
                raw_payload_json,
            ),
        )
        inserted = cur.fetchone()
        if inserted is not None:
            raw_item_id = int(inserted["id"])
            was_new = True
        else:
            existing = conn.execute(
                "SELECT id FROM raw_items WHERE source_type = ? AND dedup_key = ?",
                (source_type, dedup_key),
            ).fetchone()
            if existing is None:
                # ON CONFLICT DO NOTHING fired but the conflicting row vanished:
                # would only happen under concurrent DELETE, which we don't do.
                raise RuntimeError(
                    f"raw_items upsert lost row for ({source_type!r}, {dedup_key!r})"
                )
            raw_item_id = int(existing["id"])
            was_new = False

        conn.execute(
            "INSERT OR IGNORE INTO run_raw_items (run_id, raw_item_id) VALUES (?, ?)",
            (run_id, raw_item_id),
        )

    return raw_item_id, was_new


def link_raw_item_to_run(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    raw_item_id: int,
) -> bool:
    """Idempotently associate an existing raw item with a run.

    Useful when a topical-search run reuses an item discovered by an earlier
    daily run — :func:`upsert_raw_item` already covers the same-run case.

    Returns:
        ``True`` if a new ``run_raw_items`` row was inserted, ``False`` if the
        link already existed.

    Raises:
        sqlite3.IntegrityError: if either ``run_id`` or ``raw_item_id`` does
            not reference an existing row.
    """
    with transaction(conn):
        cur = conn.execute(
            "INSERT OR IGNORE INTO run_raw_items (run_id, raw_item_id) VALUES (?, ?)",
            (run_id, raw_item_id),
        )
        return cur.rowcount == 1


# ── source_fetch_state ────────────────────────────────────────────────────────


def record_fetch_success(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    fetcher: str,
    cursor: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Stamp a successful fetch on ``source_fetch_state``.

    Resets ``consecutive_errors`` to 0 and clears ``last_error``. ``cursor``
    and ``metadata`` are preserved if not supplied — fetchers that don't use
    cursoring (e.g. RSS) should leave them ``None``.
    """
    now = _utc_now_iso()
    metadata_json = None if metadata is None else _dump_metadata(metadata)
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO source_fetch_state
                (source_id, fetcher, last_success_at, last_attempt_at,
                 last_error, consecutive_errors, cursor, metadata)
            VALUES (?, ?, ?, ?, NULL, 0, ?, COALESCE(?, '{}'))
            ON CONFLICT (source_id, fetcher) DO UPDATE SET
                last_success_at = excluded.last_success_at,
                last_attempt_at = excluded.last_attempt_at,
                last_error = NULL,
                consecutive_errors = 0,
                cursor = COALESCE(excluded.cursor, source_fetch_state.cursor),
                metadata = COALESCE(?, source_fetch_state.metadata)
            """,
            (source_id, fetcher, now, now, cursor, metadata_json, metadata_json),
        )


def record_fetch_failure(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    fetcher: str,
    error: str,
) -> None:
    """Stamp a failed fetch on ``source_fetch_state``.

    Bumps ``consecutive_errors`` by 1 and overwrites ``last_error``;
    ``last_success_at`` and ``cursor`` are intentionally untouched so a
    stretch of failures doesn't lose the last-known-good cursor.
    """
    now = _utc_now_iso()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO source_fetch_state
                (source_id, fetcher, last_attempt_at, last_error,
                 consecutive_errors)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT (source_id, fetcher) DO UPDATE SET
                last_attempt_at = excluded.last_attempt_at,
                last_error = excluded.last_error,
                consecutive_errors = source_fetch_state.consecutive_errors + 1
            """,
            (source_id, fetcher, now, error),
        )
