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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .paths import DB_PATH

if TYPE_CHECKING:
    from .llm.schemas import ClusterSummaryPayload

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


# ── github_repo_observations ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RepoVelocity:
    """Star/fork delta for a GitHub repository over a recent observation window.

    Computed from ``github_repo_observations`` rows whose ``observed_at`` falls
    within the requested window. ``earliest`` and ``latest`` are by *time of
    observation*, not by star count — for "gaining traction" the time-ordered
    delta is what matters; using min/max of stars would conflate temporary
    unstar dips with the trend.
    """

    full_name: str
    star_delta: int
    fork_delta: int | None
    days_observed: float
    earliest_stars: int
    latest_stars: int
    earliest_at: str
    latest_at: str
    observation_count: int


def record_repo_observation(
    conn: sqlite3.Connection,
    *,
    full_name: str,
    stars: int,
    discovered_via: Literal["trending", "search"],
    forks: int | None = None,
    watchers: int | None = None,
    open_issues: int | None = None,
    language: str | None = None,
    topics: list[str] | None = None,
    pushed_at: str | None = None,
    observed_at: str | None = None,
) -> int:
    """Record one observation of a GitHub repository's state.

    Each call appends a new row — observations accumulate so velocity can be
    computed across runs. The fetcher passes the same ``observed_at`` for
    every repo in a single fetch so a daily run produces a coherent snapshot.
    Returns the inserted row id.
    """
    if not full_name or not full_name.strip():
        raise ValueError("full_name is required")
    if stars < 0:
        raise ValueError("stars must be non-negative")

    when = observed_at or _utc_now_iso()
    topics_json = json.dumps(list(topics or []), separators=(",", ":"), sort_keys=False)

    with transaction(conn):
        cur = conn.execute(
            """
            INSERT INTO github_repo_observations
                (full_name, observed_at, stars, forks, watchers, open_issues,
                 language, topics, pushed_at, discovered_via)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                full_name.strip(),
                when,
                int(stars),
                forks,
                watchers,
                open_issues,
                language,
                topics_json,
                pushed_at,
                discovered_via,
            ),
        )
        return int(cur.lastrowid or 0)


def get_repo_velocity(
    conn: sqlite3.Connection,
    *,
    full_name: str,
    window_days: int = 7,
    reference_at: str | None = None,
) -> RepoVelocity | None:
    """Compute the star/fork delta for a repo over the recent observation window.

    Returns ``None`` when fewer than two observations exist in the window —
    Day-1 has no velocity by definition (architecture doc explicitly accepts
    this). Pulling observations from before the window would conflate
    historical levels with current trend, so we don't.

    ``reference_at`` (UTC ISO) lets tests pin "now" without needing freezegun;
    in production callers omit it and the helper uses wall-clock time.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if not full_name or not full_name.strip():
        raise ValueError("full_name is required")

    ref = (
        datetime.fromisoformat(reference_at)
        if reference_at is not None
        else datetime.now(timezone.utc)
    )
    if ref.tzinfo is None:
        raise ValueError("reference_at must be timezone-aware")
    threshold = (ref - timedelta(days=window_days)).isoformat(timespec="seconds")

    rows = conn.execute(
        """
        SELECT observed_at, stars, forks
          FROM github_repo_observations
         WHERE full_name = ?
           AND observed_at >= ?
         ORDER BY observed_at ASC
        """,
        (full_name.strip(), threshold),
    ).fetchall()

    if len(rows) < 2:
        return None

    earliest = rows[0]
    latest = rows[-1]
    earliest_at = datetime.fromisoformat(earliest["observed_at"])
    latest_at = datetime.fromisoformat(latest["observed_at"])
    days = (latest_at - earliest_at).total_seconds() / 86400.0

    fork_delta: int | None
    if earliest["forks"] is not None and latest["forks"] is not None:
        fork_delta = int(latest["forks"]) - int(earliest["forks"])
    else:
        fork_delta = None

    return RepoVelocity(
        full_name=full_name.strip(),
        star_delta=int(latest["stars"]) - int(earliest["stars"]),
        fork_delta=fork_delta,
        days_observed=days,
        earliest_stars=int(earliest["stars"]),
        latest_stars=int(latest["stars"]),
        earliest_at=earliest["observed_at"],
        latest_at=latest["observed_at"],
        observation_count=len(rows),
    )


# ── item_clusters / cluster_items ─────────────────────────────────────────────


CLUSTER_STATUSES: frozenset[str] = frozenset({"pending", "filtered_out", "kept", "summarized"})


def iter_run_raw_items(
    conn: sqlite3.Connection,
    run_id: int,
) -> Iterator[sqlite3.Row]:
    """Yield every raw item linked to *run_id* via ``run_raw_items``.

    Ordered by ``raw_items.id ASC`` so clustering picks a deterministic
    representative when several items share a cluster key. Includes items
    first sighted in earlier runs that are reused by this run — the
    ``run_raw_items`` join table is the run-scoped pool.
    """
    rows = conn.execute(
        """
        SELECT ri.id, ri.source_type, ri.title, ri.url, ri.canonical_url,
               ri.content_hash, ri.published_at
          FROM run_raw_items rri
          JOIN raw_items ri ON ri.id = rri.raw_item_id
         WHERE rri.run_id = ?
         ORDER BY ri.id ASC
        """,
        (run_id,),
    ).fetchall()
    yield from rows


def create_cluster(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    cluster_key: str,
    title: str,
    raw_item_ids: Iterable[int],
    status: str = "pending",
    metadata: dict[str, Any] | None = None,
) -> tuple[int, bool]:
    """Insert a cluster and attach its members idempotently in one transaction.

    On a ``(run_id, cluster_key)`` conflict the existing cluster is left
    untouched: ``title``, ``status``, ``metadata`` are not overwritten because
    the relevance filter (a later stage) mutates ``status`` and we must not
    clobber its decisions on a clustering re-run. New ``raw_item_ids`` are
    still appended via ``INSERT OR IGNORE INTO cluster_items`` so a re-run
    that discovers additional members of an existing cluster still records
    the linkage.

    Returns:
        ``(cluster_id, was_new)`` where ``was_new`` is ``True`` iff this call
        inserted the ``item_clusters`` row.

    Raises:
        ValueError: if ``cluster_key`` is empty, ``status`` is not a valid
            cluster status, or ``raw_item_ids`` is empty (a cluster of zero
            members is meaningless and would only mask an upstream bug).
        sqlite3.IntegrityError: if ``run_id`` does not reference an existing
            run, or any ``raw_item_id`` does not exist.
    """
    if not cluster_key:
        raise ValueError("cluster_key is required")
    if status not in CLUSTER_STATUSES:
        raise ValueError(
            f"invalid cluster status {status!r}; must be one of {sorted(CLUSTER_STATUSES)}"
        )
    ids = [int(rid) for rid in raw_item_ids]
    if not ids:
        raise ValueError("raw_item_ids must contain at least one id")

    metadata_json = _dump_metadata(metadata)

    with transaction(conn):
        cur = conn.execute(
            """
            INSERT INTO item_clusters (run_id, cluster_key, title, status, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (run_id, cluster_key) DO NOTHING
            RETURNING id
            """,
            (run_id, cluster_key, title, status, metadata_json),
        )
        inserted = cur.fetchone()
        if inserted is not None:
            cluster_id = int(inserted["id"])
            was_new = True
        else:
            existing = conn.execute(
                "SELECT id FROM item_clusters WHERE run_id = ? AND cluster_key = ?",
                (run_id, cluster_key),
            ).fetchone()
            if existing is None:
                raise RuntimeError(f"item_clusters upsert lost row for ({run_id}, {cluster_key!r})")
            cluster_id = int(existing["id"])
            was_new = False

        conn.executemany(
            "INSERT OR IGNORE INTO cluster_items (cluster_id, raw_item_id) VALUES (?, ?)",
            [(cluster_id, rid) for rid in ids],
        )

    return cluster_id, was_new


VERDICT_STATUSES: frozenset[str] = frozenset({"kept", "filtered_out"})


def iter_pending_clusters_with_members(
    conn: sqlite3.Connection,
    run_id: int,
) -> Iterator[tuple[int, str, list[sqlite3.Row]]]:
    """Yield ``(cluster_id, title, members)`` for each pending cluster.

    Members are :class:`sqlite3.Row` objects exposing ``canonical_url``
    and ``excerpt``, sorted by ``raw_items.id ASC`` — the first member
    is the cluster's representative, matching the clustering layer's
    smallest-id tie-break rule. Clusters with no surviving members
    (would shouldn't happen, since ``create_cluster`` rejects empty
    member lists) are skipped silently rather than yielding an
    obviously-broken row.

    The query is one round-trip; grouping happens in Python. ~700
    clusters with ~1-2 members each is well within the size at which
    that's the right shape.
    """
    rows = conn.execute(
        """
        SELECT ic.id        AS cluster_id,
               ic.title     AS title,
               ri.id        AS raw_item_id,
               ri.canonical_url AS canonical_url,
               ri.excerpt   AS excerpt
          FROM item_clusters ic
          JOIN cluster_items ci ON ci.cluster_id = ic.id
          JOIN raw_items     ri ON ri.id = ci.raw_item_id
         WHERE ic.run_id = ?
           AND ic.status = 'pending'
         ORDER BY ic.id ASC, ri.id ASC
        """,
        (run_id,),
    ).fetchall()

    if not rows:
        return

    current_id = rows[0]["cluster_id"]
    current_title = rows[0]["title"] or ""
    current_members: list[sqlite3.Row] = []
    for row in rows:
        if row["cluster_id"] != current_id:
            yield current_id, current_title, current_members
            current_id = row["cluster_id"]
            current_title = row["title"] or ""
            current_members = []
        current_members.append(row)
    yield current_id, current_title, current_members


def update_cluster_verdict(
    conn: sqlite3.Connection,
    *,
    cluster_id: int,
    status: str,
    relevance_score: float,
    category: str | None,
    event_type: str | None,
    filter_reason: str | None,
) -> None:
    """Apply one relevance verdict to an existing cluster row.

    Promotes ``status`` from ``'pending'`` to ``'kept'`` or
    ``'filtered_out'`` and records the LLM's judgement fields. Called
    by the relevance filter (step 9b) once per cluster in a batch. The
    write is idempotent — re-applying the same verdict is a clean
    no-op modulo the timestamps in adjacent rows.

    Restricting ``status`` to the verdict set here (not the broader
    ``CLUSTER_STATUSES``) is load-bearing: the relevance filter must
    not push a cluster back to ``'pending'`` (would invalidate prior
    verdicts on a re-run) or directly to ``'summarized'`` (that
    transition is owned by the cluster-summary stage).

    ``category`` and ``filter_reason`` accept ``None`` because local
    models reliably emit ``null`` for both fields on rejected
    verdicts — the schema is permissive there and so is this helper.
    The corresponding columns are nullable in the migration.

    Raises:
        ValueError: ``status`` not in :data:`VERDICT_STATUSES`.
        LookupError: ``cluster_id`` does not reference an existing
            cluster row. Surfaces lost rows loudly rather than masking
            them as a silent no-op.
    """
    if status not in VERDICT_STATUSES:
        raise ValueError(
            f"invalid verdict status {status!r}; must be one of {sorted(VERDICT_STATUSES)}"
        )

    with transaction(conn):
        cur = conn.execute(
            """
            UPDATE item_clusters
               SET status = ?,
                   relevance_score = ?,
                   category = ?,
                   event_type = ?,
                   filter_reason = ?
             WHERE id = ?
            """,
            (status, relevance_score, category, event_type, filter_reason, cluster_id),
        )
        if cur.rowcount == 0:
            raise LookupError(f"item_clusters row {cluster_id} not found")


# ── item_summaries ────────────────────────────────────────────────────────────


def create_item_summary(
    conn: sqlite3.Connection,
    *,
    cluster_id: int,
    model: str,
    prompt_version: str,
    payload: "ClusterSummaryPayload",
) -> int:
    """Append one row to ``item_summaries`` and return its id.

    Called by the cluster-summary stage (step 10b) once per kept
    cluster. The narrative list fields (``entities`` / ``key_facts`` /
    ``caveats`` / ``source_urls``) round-trip through the TEXT columns
    as JSON strings — the migration declares those columns ``DEFAULT
    '[]'`` so an empty array survives canonical serialization. The
    column is read back by the final composer (step 11) via
    ``json.loads``.

    ``model`` and ``prompt_version`` are validated non-empty at the
    boundary so a future caller wiring up the helper directly can't
    silently land a row without provenance. The cluster's status
    advancement to ``'summarized'`` happens at the orchestration layer
    in step 10b, not here — keeping the helper focused on a single
    table lets the orchestration choose its own transaction boundary.
    """
    if not model:
        raise ValueError("model is required")
    if not prompt_version:
        raise ValueError("prompt_version is required")

    with transaction(conn):
        cur = conn.execute(
            """
            INSERT INTO item_summaries (
                cluster_id, model, prompt_version,
                headline, summary, why_it_matters,
                entities, key_facts, caveats,
                confidence, source_urls
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cluster_id,
                model,
                prompt_version,
                payload.headline,
                payload.summary,
                payload.why_it_matters,
                json.dumps(payload.entities),
                json.dumps(payload.key_facts),
                json.dumps(payload.caveats),
                payload.confidence,
                json.dumps(payload.source_urls),
            ),
        )
        return int(cur.lastrowid or 0)


# ── llm_calls ─────────────────────────────────────────────────────────────────


LLM_CALL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed"})


def record_llm_call(
    conn: sqlite3.Connection,
    *,
    stage: str,
    provider: str,
    model: str,
    status: str,
    latency_ms: int,
    run_id: int | None = None,
    prompt_version: str | None = None,
    input_hash: str | None = None,
    output_hash: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    error: str | None = None,
) -> int:
    """Append one row to ``llm_calls`` and return its id.

    ``input_hash`` / ``output_hash`` are SHA-256 hex digests over the canonical
    message list and the response content, respectively. Full prompt and
    response text are intentionally NOT stored: the row carries enough to
    reconstruct context (prompt_version + run metadata) and storing the
    rest would bloat the DB with sensitive content recoverable elsewhere.

    On ``status='failed'``, ``error`` should be a short reason; ``output_hash``
    and token counts may be zero/None. The schema's CHECK constraint enforces
    valid statuses; we validate at the boundary too so a typo raises in
    Python before round-tripping to SQLite.
    """
    if status not in LLM_CALL_STATUSES:
        raise ValueError(
            f"invalid llm_calls status {status!r}; must be one of {sorted(LLM_CALL_STATUSES)}"
        )
    if not stage:
        raise ValueError("stage is required")
    if not provider:
        raise ValueError("provider is required")
    if not model:
        raise ValueError("model is required")

    with transaction(conn):
        cur = conn.execute(
            """
            INSERT INTO llm_calls (
                run_id, stage, provider, model, prompt_version,
                input_hash, output_hash, latency_ms,
                prompt_tokens, completion_tokens, status, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                stage,
                provider,
                model,
                prompt_version,
                input_hash,
                output_hash,
                int(latency_ms),
                int(prompt_tokens),
                int(completion_tokens),
                status,
                error,
            ),
        )
        return int(cur.lastrowid or 0)
