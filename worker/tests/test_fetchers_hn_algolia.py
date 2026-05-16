"""Tests for the HN Algolia search fetcher (Phase 7c topical-search source).

Three test surfaces:

1. ``parse_algolia_hit`` + ``parse_algolia_response`` are pure —
   exercised with hand-built dicts matching the documented Algolia
   shape: story with external URL, Ask HN (story_text body), missing
   objectID, non-story tags, malformed payloads.
2. ``_build_params`` (pure) — assert the query string the fetcher
   composes for the live Algolia endpoint, including the time-window
   numericFilters clause.
3. ``fetch_hn_algolia`` uses ``httpx.MockTransport`` — assert the
   right URL + params are hit, results normalize correctly,
   HTTP errors propagate to let the runner mark the task ``failed``.

Critical dedup invariant pinned by a dedicated test:
``source_type="hn"`` (NOT ``"hn_algolia"``) and the same
``hn_dedup_key`` as the Firebase fetcher → the same HN item discovered
via either path collapses on ``UNIQUE(source_type, dedup_key)``.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from clawfeed_intel import normalize
from clawfeed_intel.fetchers import FETCHER_REGISTRY
from clawfeed_intel.fetchers.hn_algolia import (
    API_URL,
    KIND,
    SOURCE_TYPE,
    _build_params,
    fetch_hn_algolia,
    parse_algolia_hit,
    parse_algolia_response,
)
from clawfeed_intel.sources import HnAlgoliaTask, ResolvedTask


# ── helpers ───────────────────────────────────────────────────────────────────


def _task(
    *,
    query: str = "Khosla Ventures",
    tags: str = "story",
    window_start_epoch: int | None = None,
    hits_per_page: int = 50,
    source_name: str = "topic:khosla-ventures",
) -> ResolvedTask:
    return ResolvedTask(
        task=HnAlgoliaTask(
            kind="hn_algolia",
            query=query,
            tags=tags,
            window_start_epoch=window_start_epoch,
            hits_per_page=hits_per_page,
        ),
        category="topic",
        origin="yaml",
        source_id=None,
        source_name=source_name,
    )


def _hit(
    item_id: int,
    *,
    title: str | None = "Sample Story",
    story_title: str | None = None,
    url: str | None = "https://example.com/article",
    story_url: str | None = None,
    points: int | None = 100,
    num_comments: int | None = 25,
    author: str = "alice",
    created_at_i: int = 1715000000,
    story_text: str | None = None,
    tags: list[str] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One Algolia hit dict mirroring the documented response shape."""
    payload: dict[str, Any] = {
        "objectID": str(item_id),
        "_tags": tags if tags is not None else ["story", f"author_{author}", f"story_{item_id}"],
        "author": author,
        "created_at_i": created_at_i,
        "created_at": "2024-05-06T00:00:00.000Z",
    }
    if title is not None:
        payload["title"] = title
    if story_title is not None:
        payload["story_title"] = story_title
    if url is not None:
        payload["url"] = url
    if story_url is not None:
        payload["story_url"] = story_url
    if points is not None:
        payload["points"] = points
    if num_comments is not None:
        payload["num_comments"] = num_comments
    if story_text is not None:
        payload["story_text"] = story_text
    if extras:
        payload.update(extras)
    return payload


@pytest.fixture
def patch_client(monkeypatch):
    """Replace ``hn_algolia.build_client`` with a MockTransport-backed client.

    Same pattern as ``test_fetchers_hn.py::patch_client``. The handler
    receives the request directly so it can assert against the URL +
    params the fetcher composed.
    """

    def _patch(handler):
        transport = httpx.MockTransport(handler)

        @asynccontextmanager
        async def fake_build_client(*, follow_redirects: bool = True):
            from clawfeed_intel.fetchers.http import DEFAULT_TIMEOUT, default_headers

            async with httpx.AsyncClient(
                transport=transport,
                timeout=DEFAULT_TIMEOUT,
                headers=default_headers(),
                follow_redirects=follow_redirects,
            ) as client:
                yield client

        monkeypatch.setattr("clawfeed_intel.fetchers.hn_algolia.build_client", fake_build_client)

    return _patch


def _conn() -> sqlite3.Connection:
    """In-memory throwaway connection; HN Algolia doesn't touch the DB."""
    return sqlite3.connect(":memory:")


# ── parse_algolia_hit (pure) ──────────────────────────────────────────────────


def test_parse_hit_story_with_external_url():
    hit = _hit(8863, title="Build a thing", url="https://example.com/post")
    item = parse_algolia_hit(hit, query="thing")
    assert item is not None
    assert item.source_type == "hn"  # load-bearing: matches Firebase fetcher
    assert item.dedup_key == normalize.hn_dedup_key(8863)
    assert item.title == "Build a thing"
    assert item.url == "https://example.com/post"
    assert item.canonical_url == normalize.canonicalize_url("https://example.com/post")
    assert item.author == "alice"
    assert item.metadata["hn_id"] == 8863
    assert item.metadata["discovered_via"] == "algolia"
    assert item.metadata["query"] == "thing"
    assert item.metadata["score"] == 100
    assert item.metadata["descendants"] == 25
    assert item.metadata["external_url"] == "https://example.com/post"
    assert item.metadata["discussion_url"] == "https://news.ycombinator.com/item?id=8863"
    assert item.published_at and item.published_at.endswith("+00:00")


def test_parse_hit_ask_hn_with_story_text():
    """Ask HN hits have ``story_text`` and no external URL; the
    canonical URL falls back to the discussion URL."""
    hit = _hit(
        9000,
        title="Ask HN: What's your favorite editor?",
        url=None,
        story_text="<p>I'm <b>curious</b> what folks here use day-to-day.</p>",
    )
    item = parse_algolia_hit(hit, query="editor")
    assert item is not None
    assert item.url == "https://news.ycombinator.com/item?id=9000"
    assert "external_url" not in item.metadata
    # HTML stripped, whitespace collapsed.
    assert "<p>" not in item.content
    assert "curious" in item.content


def test_parse_hit_prefers_title_over_story_title():
    """When both ``title`` and ``story_title`` are set the canonical
    ``title`` wins (Algolia uses ``story_title`` for comment hits to
    carry the parent story's title — we filter comments out, but the
    field-priority rule must hold for clarity)."""
    hit = _hit(1, title="Real title", story_title="Different story title")
    item = parse_algolia_hit(hit, query="x")
    assert item is not None
    assert item.title == "Real title"


def test_parse_hit_falls_back_to_story_title_when_title_missing():
    hit = _hit(1, title=None, story_title="Story title from comment hit")
    item = parse_algolia_hit(hit, query="x")
    assert item is not None
    assert item.title == "Story title from comment hit"


def test_parse_hit_falls_back_to_story_url_when_url_missing():
    """Comment hits use ``story_url`` for the parent story link."""
    hit = _hit(1, url=None, story_url="https://news.example.com/article")
    item = parse_algolia_hit(hit, query="x")
    assert item is not None
    assert item.url == "https://news.example.com/article"


def test_parse_hit_skips_when_no_object_id():
    hit = _hit(1)
    hit.pop("objectID")
    assert parse_algolia_hit(hit, query="x") is None


def test_parse_hit_skips_when_object_id_not_numeric():
    """HN object IDs are integer strings; anything else is unexpected
    and unsafe to assume an int from."""
    hit = _hit(1)
    hit["objectID"] = "abc123"
    assert parse_algolia_hit(hit, query="x") is None


def test_parse_hit_skips_when_no_usable_title():
    hit = _hit(1, title=None, story_title=None)
    assert parse_algolia_hit(hit, query="x") is None


def test_parse_hit_skips_comment_tag():
    """Algolia should respect the tags=story filter, but the defensive
    check catches the case where it doesn't."""
    hit = _hit(1, tags=["comment", "author_alice", "story_42"])
    assert parse_algolia_hit(hit, query="x") is None


def test_parse_hit_accepts_when_tags_missing_entirely():
    """API drift defense — a missing _tags array shouldn't drop the hit."""
    hit = _hit(1)
    hit.pop("_tags")
    item = parse_algolia_hit(hit, query="x")
    assert item is not None


def test_parse_hit_handles_missing_points_and_num_comments():
    hit = _hit(1, points=None, num_comments=None)
    item = parse_algolia_hit(hit, query="x")
    assert item is not None
    assert item.metadata["score"] == 0
    assert item.metadata["descendants"] == 0


def test_parse_hit_returns_none_for_non_dict():
    assert parse_algolia_hit("not a dict", query="x") is None  # type: ignore[arg-type]
    assert parse_algolia_hit(None, query="x") is None  # type: ignore[arg-type]


def test_parse_hit_copies_tags_into_metadata():
    hit = _hit(1, tags=["story", "author_alice", "ask_hn", "front_page"])
    item = parse_algolia_hit(hit, query="x")
    assert item is not None
    assert item.metadata["tags"] == ["story", "author_alice", "ask_hn", "front_page"]


def test_parse_hit_strips_highlight_metadata_from_raw_payload():
    """``_highlightResult`` / ``_snippetResult`` bloat raw_payload by
    3-5x with no downstream value — they must not be persisted."""
    hit = _hit(
        1,
        extras={
            "_highlightResult": {"title": {"value": "<em>x</em>", "matchLevel": "full"}},
            "_snippetResult": {"story_text": {"value": "snippet"}},
        },
    )
    item = parse_algolia_hit(hit, query="x")
    assert item is not None
    assert "_highlightResult" not in item.raw_payload
    assert "_snippetResult" not in item.raw_payload


def test_parse_hit_ignores_invalid_created_at():
    """A non-numeric ``created_at_i`` shouldn't blow up the item — date
    becomes None, the rest is still usable for the brief."""
    hit = _hit(1)
    hit["created_at_i"] = "not a number"
    item = parse_algolia_hit(hit, query="x")
    assert item is not None
    assert item.published_at is None


# ── Cross-API dedup invariant ────────────────────────────────────────────────


def test_dedup_key_matches_firebase_fetcher():
    """Load-bearing: same HN item, either API → same dedup_key.

    Without this, the topic search would surface a freshly-collected
    duplicate of every HN item the daily Firebase fetcher already
    captured. The runner's ``UNIQUE(source_type, dedup_key)`` collapse
    depends on both source_type AND dedup_key matching the daily
    fetcher's output exactly.
    """
    from clawfeed_intel.fetchers.hn import KIND as HN_KIND

    algolia_item = parse_algolia_hit(_hit(12345), query="x")
    assert algolia_item is not None
    # The daily HN Firebase fetcher uses ``source_type = KIND = "hn"``
    # (no separate SOURCE_TYPE constant — KIND doubles as both). We
    # set ``source_type = "hn"`` explicitly for cross-API dedup.
    assert algolia_item.source_type == "hn"
    assert algolia_item.source_type == HN_KIND
    assert algolia_item.dedup_key == normalize.hn_dedup_key(12345)


# ── parse_algolia_response ───────────────────────────────────────────────────


def test_parse_response_happy_path():
    payload = {
        "hits": [_hit(1), _hit(2), _hit(3)],
        "nbHits": 3,
        "page": 0,
        "nbPages": 1,
    }
    items = parse_algolia_response(payload, query="x")
    assert len(items) == 3
    assert [i.metadata["hn_id"] for i in items] == [1, 2, 3]


def test_parse_response_skips_filtered_hits():
    """Non-story tags get filtered out at the hit-parse layer; the
    response-level helper just collects the survivors."""
    payload = {
        "hits": [
            _hit(1),
            _hit(2, tags=["comment", "author_alice"]),  # filtered
            _hit(3),
        ]
    }
    items = parse_algolia_response(payload, query="x")
    assert [i.metadata["hn_id"] for i in items] == [1, 3]


def test_parse_response_non_dict_payload_returns_empty():
    assert parse_algolia_response("not a dict", query="x") == []  # type: ignore[arg-type]
    assert parse_algolia_response(None, query="x") == []  # type: ignore[arg-type]
    assert parse_algolia_response([{"objectID": "1"}], query="x") == []


def test_parse_response_missing_hits_key_returns_empty():
    """API drift defense."""
    assert parse_algolia_response({"nbHits": 0}, query="x") == []


def test_parse_response_non_list_hits_returns_empty():
    assert parse_algolia_response({"hits": "huh"}, query="x") == []


def test_parse_response_per_hit_exception_does_not_abort():
    """One bad hit must not poison the rest of the batch."""

    class _ExplodingHit(dict):
        def get(self, key, default=None):
            if key == "objectID":
                raise RuntimeError("boom")
            return super().get(key, default)

    payload = {
        "hits": [_hit(1), _ExplodingHit(_hit(2)), _hit(3)],
    }
    items = parse_algolia_response(payload, query="x")
    assert [i.metadata["hn_id"] for i in items] == [1, 3]


# ── _build_params (pure) ─────────────────────────────────────────────────────


def test_build_params_defaults():
    task = HnAlgoliaTask(kind="hn_algolia", query="Khosla Ventures")
    params = _build_params(task)
    assert params == {
        "query": "Khosla Ventures",
        "tags": "story",
        "hitsPerPage": 50,
    }
    # numericFilters omitted when no window — must NOT appear as a stub.
    assert "numericFilters" not in params


def test_build_params_composes_numeric_filter_when_window_set():
    task = HnAlgoliaTask(
        kind="hn_algolia",
        query="Khosla Ventures",
        window_start_epoch=1700000000,
    )
    params = _build_params(task)
    assert params["numericFilters"] == "created_at_i>1700000000"


def test_build_params_forwards_custom_tags():
    task = HnAlgoliaTask(
        kind="hn_algolia",
        query="Khosla Ventures",
        tags="(story,comment)",
        hits_per_page=25,
    )
    params = _build_params(task)
    assert params["tags"] == "(story,comment)"
    assert params["hitsPerPage"] == 25


def test_hits_per_page_bounds_enforced_at_schema_layer():
    """Schema rejects out-of-range hitsPerPage at construction time
    rather than risking a runtime API error from Algolia."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HnAlgoliaTask(kind="hn_algolia", query="x", hits_per_page=0)
    with pytest.raises(ValidationError):
        HnAlgoliaTask(kind="hn_algolia", query="x", hits_per_page=1001)


# ── fetch_hn_algolia (HTTP via MockTransport) ────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_happy_path(patch_client):
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"hits": [_hit(1), _hit(2)], "nbHits": 2},
        )

    patch_client(handler)
    items = await fetch_hn_algolia(_conn(), _task(query="Khosla"))
    assert len(items) == 2
    assert captured["url"].startswith(API_URL)
    assert "query=Khosla" in captured["url"]
    assert "tags=story" in captured["url"]
    assert "hitsPerPage=50" in captured["url"]


@pytest.mark.asyncio
async def test_fetch_composes_numeric_filter_in_request(patch_client):
    """End-to-end pin: window_start_epoch on the task → numericFilters
    in the actual Algolia request URL."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"hits": []})

    patch_client(handler)
    task = _task(query="x", window_start_epoch=1700000000)
    await fetch_hn_algolia(_conn(), task)
    assert "numericFilters=created_at_i" in captured["url"]
    # urlencoded `>` is `%3E`
    assert "%3E1700000000" in captured["url"]


@pytest.mark.asyncio
async def test_fetch_empty_hits_returns_empty(patch_client):
    patch_client(lambda req: httpx.Response(200, json={"hits": [], "nbHits": 0}))
    items = await fetch_hn_algolia(_conn(), _task())
    assert items == []


@pytest.mark.asyncio
async def test_fetch_propagates_5xx_as_http_status_error(patch_client):
    patch_client(lambda req: httpx.Response(503, text="upstream lit on fire"))
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_hn_algolia(_conn(), _task())


@pytest.mark.asyncio
async def test_fetch_propagates_4xx_as_http_status_error(patch_client):
    """A 400 from Algolia (bad query syntax, malformed numericFilters,
    etc.) should reach the runner's failed-task path, not silently
    return zero hits."""
    patch_client(lambda req: httpx.Response(400, json={"error": "bad query"}))
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_hn_algolia(_conn(), _task())


@pytest.mark.asyncio
async def test_fetch_rejects_non_hn_algolia_task():
    from clawfeed_intel.sources import HnTask

    wrong_task = ResolvedTask(
        task=HnTask(kind="hn", list="top"),
        category="ai_coding_tools",
        origin="yaml",
        source_id=None,
        source_name="x",
    )
    with pytest.raises(TypeError, match="expected HnAlgoliaTask"):
        await fetch_hn_algolia(_conn(), wrong_task)


# ── Registration ─────────────────────────────────────────────────────────────


def test_kind_registered():
    assert KIND == "hn_algolia"
    assert FETCHER_REGISTRY[KIND] is fetch_hn_algolia


def test_source_type_matches_firebase_for_cross_api_dedup():
    """Pinned at the constant level — accidentally changing
    SOURCE_TYPE to "hn_algolia" would silently break cross-API dedup."""
    assert SOURCE_TYPE == "hn"
