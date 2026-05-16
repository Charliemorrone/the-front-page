"""Tests for the raw-cache search fetcher (Phase 7c topical-search source).

Three test surfaces:

1. ``build_raw_cache_query`` is pure — the composed SQL + params
   shape under each input combination (single variant, multiple
   variants, window-scoped, no window, limit pinning).
2. Row hydration — round-trips a real ``raw_items`` row through the
   fetcher and back, asserting source_type + dedup_key are preserved
   so the runner's upsert no-ops correctly.
3. ``fetch_raw_cache`` end-to-end against the ``temp_db`` fixture —
   seeded rows + variant matching + window scoping + DISTINCT
   collapse + freshness ordering.
"""

from __future__ import annotations

import json
from contextlib import closing

import pytest

from clawfeed_intel import db as worker_db
from clawfeed_intel.fetchers import FETCHER_REGISTRY
from clawfeed_intel.fetchers.raw_cache import (
    KIND,
    _parse_json_dict,
    _row_to_item,
    build_raw_cache_query,
    fetch_raw_cache,
)
from clawfeed_intel.sources import RawCacheTask, ResolvedTask


# ── helpers ───────────────────────────────────────────────────────────────────


def _task(
    *,
    query_variants: list[str] | None = None,
    window_start: str | None = None,
    limit: int = 200,
) -> ResolvedTask:
    return ResolvedTask(
        task=RawCacheTask(
            kind="raw_cache",
            query_variants=query_variants or ["Khosla"],
            window_start=window_start,
            limit=limit,
        ),
        category="topic",
        origin="yaml",
        source_id=None,
        source_name="topic:khosla",
    )


def _seed_raw_item(
    conn,
    *,
    source_type: str = "rss",
    dedup_key: str,
    title: str = "Test item",
    url: str = "https://example.com/x",
    canonical_url: str | None = None,
    content: str = "",
    published_at: str | None = None,
    fetched_at: str | None = None,
    metadata: dict | None = None,
) -> int:
    """Insert directly via SQL so the test owns the row shape exactly.

    Using db.upsert_raw_item would couple this fixture to the runner's
    upsert behavior; we want to test the cache fetcher's reading shape
    in isolation.
    """
    canonical = canonical_url if canonical_url is not None else url
    cur = conn.execute(
        """
        INSERT INTO raw_items (
            source_type, dedup_key, title, url, canonical_url,
            content, excerpt, published_at, fetched_at,
            metadata, raw_payload
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, datetime('now')), ?, '{}')
        """,
        (
            source_type,
            dedup_key,
            title,
            url,
            canonical,
            content,
            content[:320] if content else "",
            published_at,
            fetched_at,
            json.dumps(metadata or {}),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


# ── build_raw_cache_query (pure) ─────────────────────────────────────────────


def test_build_query_single_variant_no_window():
    sql, params = build_raw_cache_query(
        variants=["Khosla"],
        window_start=None,
        limit=200,
    )
    # Three LIKE clauses (title / canonical_url / content), all with %Khosla%.
    assert sql.count("LIKE ?") == 3
    assert params[:3] == ["%Khosla%", "%Khosla%", "%Khosla%"]
    # WHERE-clause window check omitted (COALESCE still appears in the
    # ORDER BY because that's the freshness rule — present unconditionally).
    assert "WHERE ((title LIKE" in sql
    assert "COALESCE(published_at, fetched_at) >=" not in sql
    # Limit always present.
    assert sql.rstrip().endswith("LIMIT ?")
    assert params[-1] == 200


def test_build_query_multiple_variants_or_joined():
    """N variants → 3N LIKE clauses, joined by OR within the outer
    AND clause. Each variant adds three params (title/url/content).
    """
    sql, params = build_raw_cache_query(
        variants=["Khosla Ventures", "Vinod Khosla", "Khosla led round"],
        window_start=None,
        limit=200,
    )
    assert sql.count("LIKE ?") == 9
    # Each variant appears three times in params (once per LIKE column).
    for variant in ["Khosla Ventures", "Vinod Khosla", "Khosla led round"]:
        assert params.count(f"%{variant}%") == 3
    # OR-joined inside the outer WHERE clause.
    assert " OR " in sql


def test_build_query_window_start_appended():
    sql, params = build_raw_cache_query(
        variants=["x"],
        window_start="2026-04-15T00:00:00+00:00",
        limit=200,
    )
    assert "COALESCE(published_at, fetched_at) >= ?" in sql
    assert "2026-04-15T00:00:00+00:00" in params


def test_build_query_strips_empty_variants():
    """Whitespace-only variants must be skipped — otherwise the LIKE
    pattern '%%' matches every row in the table."""
    sql, params = build_raw_cache_query(
        variants=["Khosla", "", "   ", "Vinod"],
        window_start=None,
        limit=10,
    )
    # 2 non-empty variants × 3 LIKE columns = 6.
    assert sql.count("LIKE ?") == 6
    # No empty LIKE patterns leaked through.
    assert "%%" not in str(params)


def test_build_query_rejects_all_empty_variants():
    """If every variant is blank, the query would match every row in
    the table — fail loudly at the boundary."""
    with pytest.raises(ValueError, match="variants must contain at least one"):
        build_raw_cache_query(variants=["", "   "], window_start=None, limit=10)


def test_build_query_includes_distinct():
    """The natural multi-variant overlap (one article matched by two
    variants) must collapse via SQL DISTINCT — otherwise the topic
    run would surface the same item twice."""
    sql, _ = build_raw_cache_query(
        variants=["a", "b"],
        window_start=None,
        limit=10,
    )
    assert "SELECT DISTINCT" in sql


def test_build_query_orders_by_freshness():
    """Freshness-first ordering means the most recent matches lead.
    Tied freshness sorts by id DESC for deterministic test output."""
    sql, _ = build_raw_cache_query(variants=["x"], window_start=None, limit=10)
    assert "ORDER BY COALESCE(published_at, fetched_at) DESC, id DESC" in sql


# ── _parse_json_dict ─────────────────────────────────────────────────────────


def test_parse_json_dict_happy_path():
    assert _parse_json_dict('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parse_json_dict_empty_or_null_input_returns_empty():
    assert _parse_json_dict(None) == {}
    assert _parse_json_dict("") == {}


def test_parse_json_dict_malformed_returns_empty_with_log():
    """Defensive: a legacy row with malformed JSON shouldn't abort
    the fetcher — treat as empty and continue."""
    assert _parse_json_dict("not json") == {}


def test_parse_json_dict_non_dict_returns_empty():
    """A JSON array or scalar value isn't useful as a metadata dict."""
    assert _parse_json_dict("[1, 2, 3]") == {}
    assert _parse_json_dict('"a string"') == {}


# ── fetch_raw_cache end-to-end (via temp_db) ─────────────────────────────────


def test_fetch_returns_empty_when_no_matches(temp_db):
    """Empty cache → empty result list."""
    import asyncio

    with closing(worker_db.connect(temp_db)) as conn:
        items = asyncio.run(fetch_raw_cache(conn, _task(query_variants=["nothing"])))
    assert items == []


def test_fetch_matches_in_title(temp_db):
    """Variant in title — case-insensitive LIKE."""
    import asyncio

    with closing(worker_db.connect(temp_db)) as conn:
        _seed_raw_item(
            conn,
            dedup_key="https://example.com/khosla-funding",
            title="Khosla Ventures backs new startup",
        )
        _seed_raw_item(
            conn,
            dedup_key="https://example.com/unrelated",
            title="Acme Corp ships product",
        )
        items = asyncio.run(fetch_raw_cache(conn, _task(query_variants=["Khosla"])))

    assert len(items) == 1
    assert "Khosla Ventures" in items[0].title


def test_fetch_matches_in_content(temp_db):
    """An article whose title doesn't mention the topic but whose
    body does — should still surface. This is the load-bearing
    raison d'être for the content-column LIKE clause."""
    import asyncio

    with closing(worker_db.connect(temp_db)) as conn:
        _seed_raw_item(
            conn,
            dedup_key="https://news.example.com/big-fund",
            title="Series A round closes",
            content="The round was led by Khosla Ventures with participation from...",
        )
        items = asyncio.run(fetch_raw_cache(conn, _task(query_variants=["Khosla"])))

    assert len(items) == 1


def test_fetch_matches_in_canonical_url(temp_db):
    """A URL-shaped match (e.g. searching for 'github' surfaces every
    github.com link)."""
    import asyncio

    with closing(worker_db.connect(temp_db)) as conn:
        _seed_raw_item(
            conn,
            dedup_key="https://github.com/acme/repo",
            title="Repo description",
            canonical_url="https://github.com/acme/repo",
        )
        items = asyncio.run(fetch_raw_cache(conn, _task(query_variants=["github.com"])))

    assert len(items) == 1


def test_fetch_dedupes_same_row_across_multiple_variants(temp_db):
    """Critical: an article matching two variants ("Khosla Ventures"
    AND "Vinod Khosla") surfaces ONCE, not twice. Without DISTINCT
    the topic brief would carry duplicates."""
    import asyncio

    with closing(worker_db.connect(temp_db)) as conn:
        _seed_raw_item(
            conn,
            dedup_key="https://example.com/article-1",
            title="Vinod Khosla speaks at Khosla Ventures summit",
        )
        items = asyncio.run(
            fetch_raw_cache(
                conn,
                _task(query_variants=["Khosla Ventures", "Vinod Khosla"]),
            )
        )
    assert len(items) == 1


def test_fetch_respects_window_start(temp_db):
    """Items older than window_start are excluded; items newer or at
    the cutoff are kept. The window applies against COALESCE(
    published_at, fetched_at) so undated items use their ingestion
    time."""
    import asyncio

    with closing(worker_db.connect(temp_db)) as conn:
        _seed_raw_item(
            conn,
            dedup_key="https://example.com/old",
            title="Old Khosla story",
            published_at="2025-01-01T00:00:00+00:00",
        )
        _seed_raw_item(
            conn,
            dedup_key="https://example.com/recent",
            title="Recent Khosla story",
            published_at="2026-05-01T00:00:00+00:00",
        )
        items = asyncio.run(
            fetch_raw_cache(
                conn,
                _task(
                    query_variants=["Khosla"],
                    window_start="2026-04-01T00:00:00+00:00",
                ),
            )
        )
    assert len(items) == 1
    assert items[0].title == "Recent Khosla story"


def test_fetch_window_uses_fetched_at_when_published_at_null(temp_db):
    """An undated raw_item respects the window via its fetched_at
    timestamp — otherwise topic search would silently surface every
    null-dated cached row regardless of recency."""
    import asyncio

    with closing(worker_db.connect(temp_db)) as conn:
        _seed_raw_item(
            conn,
            dedup_key="https://example.com/old-undated",
            title="Old Khosla story (no date)",
            published_at=None,
            fetched_at="2025-01-01T00:00:00+00:00",
        )
        items = asyncio.run(
            fetch_raw_cache(
                conn,
                _task(
                    query_variants=["Khosla"],
                    window_start="2026-04-01T00:00:00+00:00",
                ),
            )
        )
    assert items == []


def test_fetch_orders_by_freshness_desc(temp_db):
    """Most recent matches lead so downstream relevance ordering
    starts from a sensible prior."""
    import asyncio

    with closing(worker_db.connect(temp_db)) as conn:
        _seed_raw_item(
            conn,
            dedup_key="https://example.com/older",
            title="Khosla older",
            published_at="2026-04-01T00:00:00+00:00",
        )
        _seed_raw_item(
            conn,
            dedup_key="https://example.com/newer",
            title="Khosla newer",
            published_at="2026-05-01T00:00:00+00:00",
        )
        items = asyncio.run(fetch_raw_cache(conn, _task(query_variants=["Khosla"])))
    assert [i.title for i in items] == ["Khosla newer", "Khosla older"]


def test_fetch_respects_limit(temp_db):
    """Limit prevents a noisy query from surfacing thousands of items."""
    import asyncio

    with closing(worker_db.connect(temp_db)) as conn:
        for i in range(10):
            _seed_raw_item(
                conn,
                dedup_key=f"https://example.com/item-{i}",
                title=f"Khosla item {i}",
            )
        items = asyncio.run(fetch_raw_cache(conn, _task(query_variants=["Khosla"], limit=3)))
    assert len(items) == 3


def test_fetch_preserves_original_source_type_and_dedup_key(temp_db):
    """Load-bearing: the emitted FetchedItem MUST carry each row's
    original source_type + dedup_key so the runner's upsert
    (ON CONFLICT source_type+dedup_key DO NOTHING) no-ops the row
    insert and adds the topic-run linkage via run_raw_items. If we
    changed source_type to "raw_cache" or generated a new dedup_key,
    the topic run would create a duplicate raw_items row.
    """
    import asyncio

    with closing(worker_db.connect(temp_db)) as conn:
        _seed_raw_item(
            conn,
            source_type="rss",
            dedup_key="https://news.example.com/khosla-story",
            title="Khosla story from RSS",
        )
        _seed_raw_item(
            conn,
            source_type="gdelt",
            dedup_key="https://other.example.com/khosla-piece",
            title="Khosla piece from GDELT",
        )
        items = asyncio.run(fetch_raw_cache(conn, _task(query_variants=["Khosla"])))

    # Each emitted FetchedItem preserves the original source_type +
    # dedup_key. Without this the upsert path produces duplicates.
    assert {(i.source_type, i.dedup_key) for i in items} == {
        ("rss", "https://news.example.com/khosla-story"),
        ("gdelt", "https://other.example.com/khosla-piece"),
    }


def test_fetch_augments_metadata_with_raw_cache_marker(temp_db):
    """The metadata must mark the topic run's surfacing as cache-
    sourced so the dashboard can distinguish cache hits from fresh
    fetches. Original discovered_via is preserved as
    original_discovered_via."""
    import asyncio

    with closing(worker_db.connect(temp_db)) as conn:
        _seed_raw_item(
            conn,
            dedup_key="https://example.com/x",
            title="Khosla story",
            metadata={"discovered_via": "rss", "rss_feed_id": 42},
        )
        items = asyncio.run(fetch_raw_cache(conn, _task(query_variants=["Khosla"])))

    assert len(items) == 1
    assert items[0].metadata["discovered_via"] == "raw_cache"
    assert items[0].metadata["original_discovered_via"] == "rss"
    assert items[0].metadata["raw_item_id"] >= 1


def test_fetch_handles_per_row_hydration_failure(temp_db, monkeypatch):
    """Defensive: a malformed row that breaks `_row_to_item` must
    not abort the fetcher — the rest of the matches still surface."""
    import asyncio

    call_count = {"n": 0}
    original = _row_to_item

    def flaky(row):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated row corruption")
        return original(row)

    monkeypatch.setattr("clawfeed_intel.fetchers.raw_cache._row_to_item", flaky)

    with closing(worker_db.connect(temp_db)) as conn:
        _seed_raw_item(conn, dedup_key="a", title="Khosla one")
        _seed_raw_item(conn, dedup_key="b", title="Khosla two")
        _seed_raw_item(conn, dedup_key="c", title="Khosla three")
        items = asyncio.run(fetch_raw_cache(conn, _task(query_variants=["Khosla"])))

    # One row was poisoned; the other two survive.
    assert len(items) == 2


def test_fetch_rejects_non_raw_cache_task():
    """Boundary type-check defends against an orchestrator bug routing
    a non-RawCacheTask through this fetcher."""
    import asyncio
    import sqlite3

    from clawfeed_intel.sources import HnTask

    wrong = ResolvedTask(
        task=HnTask(kind="hn", list="top"),
        category="ai_coding_tools",
        origin="yaml",
        source_id=None,
        source_name="x",
    )
    with pytest.raises(TypeError, match="expected RawCacheTask"):
        asyncio.run(fetch_raw_cache(sqlite3.connect(":memory:"), wrong))


# ── Registration ─────────────────────────────────────────────────────────────


def test_kind_registered():
    assert KIND == "raw_cache"
    assert FETCHER_REGISTRY[KIND] is fetch_raw_cache


def test_raw_cache_task_min_length_enforced_at_schema():
    """Schema enforces at least one variant — boundary check."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RawCacheTask(kind="raw_cache", query_variants=[])


def test_raw_cache_task_limit_bounds_enforced_at_schema():
    """``limit`` is bounded [1, 2000] to prevent a misconfigured task
    from surfacing the entire raw_items table or returning zero rows."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RawCacheTask(kind="raw_cache", query_variants=["x"], limit=0)
    with pytest.raises(ValidationError):
        RawCacheTask(kind="raw_cache", query_variants=["x"], limit=2001)
