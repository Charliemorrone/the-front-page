"""Raw-cache search fetcher for topical search (Phase 7c).

Surfaces items the daily runs already collected — instead of re-fetching
the open web for a topic query, look first in the local ``raw_items``
table. The cache is often the highest-signal evidence pool for any
topic the operator has been daily-briefed on: those items have already
been normalized, deduped, content-extracted, and (for past kept
clusters) summarized.

Match semantics: each :class:`RawCacheTask` carries 3-8 ``query_variants``
(from the :class:`SearchPlan`). A raw_item is surfaced if **any**
variant case-insensitively appears in its title, canonical URL, or
content. SQL ``DISTINCT`` handles the natural multi-variant overlap
(e.g. "Khosla Ventures" and "Vinod Khosla" both matching the same
article surfaces it once).

**No re-fetching, no re-normalization**. The emitted :class:`FetchedItem`
carries each matched raw_item's original ``source_type`` + ``dedup_key``
verbatim. When the runner calls :func:`db.upsert_raw_item`, the
``ON CONFLICT (source_type, dedup_key) DO NOTHING`` clause no-ops the
row insert, and ``INSERT OR IGNORE INTO run_raw_items`` adds the
topic-run linkage. The cache fetcher is essentially "link existing
rows to this run"; the existing harness handles it because the upsert
contract was always designed for cross-run sharing.

Window scoping: ``window_start`` (ISO UTC) bounds matched items by
``published_at`` (falling back to ``fetched_at`` when published_at is
null). Without a window, the topic search for "Khosla Ventures" would
surface every Khosla mention back to the first daily run — useful for
some queries, noisy for most. The orchestrator (Phase 7e) passes the
run's window start; passing ``None`` means "search all-time".

Performance: with a personal-scale ``raw_items`` table (~thousands of
rows after retention), three OR'd LIKE patterns on title / URL /
content scan fast enough that no FTS index is needed for Phase 7c.
SQLite's query planner uses ``idx_raw_items_published`` for the
window filter, then linear-scans matching rows. FTS5 is the future
optimization (architecture-doc Phase 9).

Two-layer pattern:

- :func:`build_raw_cache_query` is pure — variants + window →
  ``(sql, params)``. Fixture-testable.
- :func:`fetch_raw_cache` does the SQL execution + row hydration into
  :class:`FetchedItem`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from ..sources import RawCacheTask, ResolvedTask
from .base import FETCHER_REGISTRY, FetchedItem

log = logging.getLogger(__name__)

KIND = "raw_cache"


async def fetch_raw_cache(conn: sqlite3.Connection, task: ResolvedTask) -> list[FetchedItem]:
    """Execute the cache search; return matched raw_items as :class:`FetchedItem`s.

    The emitted items carry each match's original ``source_type`` /
    ``dedup_key`` so the runner's upsert no-ops the row and adds the
    run linkage via ``run_raw_items``.
    """
    if not isinstance(task.task, RawCacheTask):
        raise TypeError(f"fetch_raw_cache expected RawCacheTask, got {type(task.task).__name__}")

    sql, params = build_raw_cache_query(
        variants=task.task.query_variants,
        window_start=task.task.window_start,
        limit=task.task.limit,
    )

    rows = conn.execute(sql, params).fetchall()
    out: list[FetchedItem] = []
    for row in rows:
        try:
            item = _row_to_item(row)
        except Exception:
            log.exception("raw_cache: failed to hydrate row id=%r", _row_id(row))
            continue
        if item is not None:
            out.append(item)
    return out


# ── Query construction (pure) ────────────────────────────────────────────────


_SELECT_COLUMNS = (
    "id",
    "source_type",
    "source_name",
    "title",
    "url",
    "canonical_url",
    "author",
    "content",
    "excerpt",
    "published_at",
    "fetched_at",
    "dedup_key",
    "content_hash",
    "metadata",
    "raw_payload",
)


def build_raw_cache_query(
    *,
    variants: list[str],
    window_start: str | None,
    limit: int,
) -> tuple[str, list[Any]]:
    """Compose the SELECT for one cache search. Pure for fixture testing.

    Variants are OR'd across three text columns each (title, canonical_url,
    content) — 3 × N LIKE clauses. Empty / whitespace-only variants are
    skipped silently so a planner that emits ``["Khosla", ""]`` doesn't
    produce a runaway "match all rows" clause via the empty pattern.

    ``window_start`` is compared against ``COALESCE(published_at,
    fetched_at)`` so items the fetcher captured but couldn't date
    still respect the window via their ingestion time. Items earlier
    than ``window_start`` (strict ``<``) are excluded.

    Raises:
        ValueError: empty ``variants`` (caller error — the
            :class:`RawCacheTask` schema enforces ``min_length=1`` but
            the helper double-checks before constructing nonsense SQL).
    """
    cleaned = [v.strip() for v in variants if v and v.strip()]
    if not cleaned:
        raise ValueError("variants must contain at least one non-empty string")

    columns_sql = ", ".join(_SELECT_COLUMNS)
    where_clauses: list[str] = []
    params: list[Any] = []

    variant_or_clauses: list[str] = []
    for variant in cleaned:
        like_pattern = f"%{variant}%"
        # LIKE is case-insensitive for ASCII by default in SQLite — fine
        # for English-language matching. Non-ASCII matching would need
        # a Unicode-aware FTS5 index; tracked as a future optimization.
        variant_or_clauses.append("(title LIKE ? OR canonical_url LIKE ? OR content LIKE ?)")
        params.extend([like_pattern, like_pattern, like_pattern])
    where_clauses.append("(" + " OR ".join(variant_or_clauses) + ")")

    if window_start is not None:
        where_clauses.append("COALESCE(published_at, fetched_at) >= ?")
        params.append(window_start)

    where_sql = " AND ".join(where_clauses)
    # DISTINCT collapses the natural overlap when a single raw_item
    # matches multiple variants. ORDER BY freshness-first so the most
    # recent matches lead — downstream relevance ordering can re-rank,
    # but a freshness-first prior gives a sensible default.
    sql = (
        f"SELECT DISTINCT {columns_sql} "
        f"FROM raw_items "
        f"WHERE {where_sql} "
        f"ORDER BY COALESCE(published_at, fetched_at) DESC, id DESC "
        f"LIMIT ?"
    )
    params.append(limit)
    return sql, params


# ── Row hydration ────────────────────────────────────────────────────────────


def _row_to_item(row: sqlite3.Row) -> FetchedItem | None:
    """Convert a ``raw_items`` row into a :class:`FetchedItem`.

    Preserves the source's original ``source_type`` + ``dedup_key`` so
    the runner's upsert correctly identifies this as the same row.
    ``metadata`` is augmented with ``discovered_via='raw_cache'`` so the
    downstream stages can tell cache surfacing apart from fresh fetches
    when reading the raw_payload back.
    """
    if row is None:
        return None

    metadata = _parse_json_dict(row["metadata"])
    raw_payload = _parse_json_dict(row["raw_payload"])

    # Augment without clobbering — if the cached row already had a
    # discovered_via tag from its original fetcher, preserve that and
    # add a separate signal that this run surfaced it from the cache.
    augmented_metadata = dict(metadata)
    augmented_metadata.setdefault("original_discovered_via", metadata.get("discovered_via"))
    augmented_metadata["discovered_via"] = "raw_cache"
    augmented_metadata["raw_item_id"] = int(row["id"])

    return FetchedItem(
        source_type=row["source_type"],
        dedup_key=row["dedup_key"],
        title=row["title"] or "",
        url=row["url"] or "",
        canonical_url=row["canonical_url"] or "",
        content=row["content"] or "",
        excerpt=row["excerpt"] or "",
        author=row["author"] or "",
        published_at=row["published_at"],
        content_hash=row["content_hash"],
        metadata=augmented_metadata,
        raw_payload=raw_payload,
    )


def _parse_json_dict(raw: str | None) -> dict[str, Any]:
    """Best-effort parse of a TEXT-stored JSON-object column.

    Defensive: a malformed JSON value (e.g. legacy row from before the
    upsert tightened serialization) is treated as empty rather than
    raising. Matches the posture of ``pipeline/compose._load_json_list``.
    """
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("raw_cache: invalid JSON column %r; treating as empty", raw[:60])
        return {}
    if not isinstance(decoded, dict):
        return {}
    return decoded


def _row_id(row: Any) -> Any:
    try:
        return row["id"] if row is not None else None
    except (KeyError, IndexError, TypeError):
        return None


# Register on import.
FETCHER_REGISTRY[KIND] = fetch_raw_cache
