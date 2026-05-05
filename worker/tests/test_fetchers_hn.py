"""Tests for the Hacker News fetcher.

Two test surfaces:

1. ``parse_hn_item`` is pure — exercised with hand-built dicts mirroring the
   shapes documented at https://github.com/HackerNews/API: regular story,
   Ask HN (text body), Show HN with external URL, deleted / dead / null
   items, comments, and edge cases (missing title, missing time).
2. ``fetch_hn`` uses ``httpx.MockTransport`` to assert the right list
   endpoint is hit per ``HnTask.list``, that ``min_score`` filters
   correctly, that ``limit`` truncates, and that one bad item-fetch does
   not abort the batch (the load-bearing failure-mode requirement).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from clawfeed_intel.fetchers import FETCHER_REGISTRY
from clawfeed_intel.fetchers.hn import (
    API_BASE,
    DISCUSSION_URL_TMPL,
    KIND,
    fetch_hn,
    parse_hn_item,
)
from clawfeed_intel.sources import HnTask, ResolvedTask


# ── helpers ───────────────────────────────────────────────────────────────────


def _task(
    *,
    list_name: str = "top",
    min_score: int | None = None,
    limit: int | None = None,
    source_name: str = "hn:top",
) -> ResolvedTask:
    return ResolvedTask(
        task=HnTask(kind="hn", list=list_name, min_score=min_score, limit=limit),
        category="ai_coding_tools",
        origin="yaml",
        source_id=None,
        source_name=source_name,
    )


def _story(
    item_id: int,
    *,
    title: str = "Sample Story",
    url: str | None = "https://example.com/article",
    score: int = 100,
    descendants: int = 25,
    by: str = "alice",
    time: int = 1715000000,
    text: str | None = None,
    item_type: str = "story",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": item_id,
        "type": item_type,
        "title": title,
        "score": score,
        "descendants": descendants,
        "by": by,
        "time": time,
    }
    if url is not None:
        payload["url"] = url
    if text is not None:
        payload["text"] = text
    if extras:
        payload.update(extras)
    return payload


@pytest.fixture
def patch_client(monkeypatch):
    """Replace ``hn.build_client`` with a MockTransport-backed client.

    Pass a routes dict (URL → ``httpx.Response`` or callable taking the
    request and returning a Response). Anything not in the dict yields 404.
    """

    def _patch(routes: dict[str, Any] | Any):
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if callable(routes):
                return routes(request)
            entry = routes.get(url)
            if entry is None:
                return httpx.Response(404, json={"error": "no route"})
            if callable(entry):
                return entry(request)
            return entry

        transport = httpx.MockTransport(handler)

        @asynccontextmanager
        async def fake_build_client(*, follow_redirects: bool = True):
            from clawfeed_intel.fetchers.http import (
                DEFAULT_TIMEOUT,
                default_headers,
            )

            async with httpx.AsyncClient(
                transport=transport,
                timeout=DEFAULT_TIMEOUT,
                headers=default_headers(),
                follow_redirects=follow_redirects,
            ) as client:
                yield client

        monkeypatch.setattr("clawfeed_intel.fetchers.hn.build_client", fake_build_client)

    return _patch


# ── parse_hn_item (pure) ──────────────────────────────────────────────────────


def test_parse_story_with_external_url():
    raw = _story(8863, title="Build a thing", url="https://example.com/post?utm_source=hn")
    item = parse_hn_item(raw, list_name="top")

    assert item is not None
    assert item.source_type == "hn"
    assert item.dedup_key == "8863"
    assert item.title == "Build a thing"
    assert item.url == "https://example.com/post?utm_source=hn"
    # canonicalize_url drops UTM-prefixed tracking params
    assert item.canonical_url == "https://example.com/post"
    assert item.author == "alice"
    assert item.metadata["hn_id"] == 8863
    assert item.metadata["list"] == "top"
    assert item.metadata["type"] == "story"
    assert item.metadata["score"] == 100
    assert item.metadata["descendants"] == 25
    assert item.metadata["discussion_url"] == "https://news.ycombinator.com/item?id=8863"
    assert item.metadata["external_url"] == "https://example.com/post?utm_source=hn"
    assert item.published_at == "2024-05-06T12:53:20+00:00"
    assert item.content_hash is not None


def test_parse_ask_hn_uses_text_as_content_with_html_stripped():
    """Ask HN posts have no url; the question body is in `text` (HTML-encoded)."""
    raw = _story(
        9001,
        title="Ask HN: best editor?",
        url=None,
        text="<p>What&#x27;s your favorite editor in 2026?</p>",
        item_type="story",
    )
    item = parse_hn_item(raw, list_name="ask")
    assert item is not None
    # No external URL → discussion URL becomes the primary
    assert item.url == DISCUSSION_URL_TMPL.format(item_id=9001)
    assert item.canonical_url == DISCUSSION_URL_TMPL.format(item_id=9001)
    # HTML stripped, entities decoded
    assert "What's your favorite editor" in item.content
    assert "<p>" not in item.content
    assert "external_url" not in item.metadata


def test_parse_show_hn_with_external_url():
    raw = _story(
        9100,
        title="Show HN: my project",
        url="https://github.com/me/project",
        text="<p>I built this thing.</p>",
    )
    item = parse_hn_item(raw, list_name="show")
    assert item is not None
    assert item.url == "https://github.com/me/project"
    assert item.canonical_url == "https://github.com/me/project"
    assert "I built this thing" in item.content


def test_parse_skips_deleted_item():
    assert parse_hn_item({"id": 1, "deleted": True}, list_name="top") is None


def test_parse_skips_dead_item():
    assert parse_hn_item({"id": 1, "dead": True, "title": "x"}, list_name="top") is None


def test_parse_skips_comment_type():
    """List endpoints don't return comments, but be defensive."""
    raw = {"id": 1, "type": "comment", "by": "x", "text": "a reply"}
    assert parse_hn_item(raw, list_name="top") is None


def test_parse_skips_untitled():
    raw = _story(1, title="")
    assert parse_hn_item(raw, list_name="top") is None


def test_parse_handles_missing_score_and_descendants():
    raw = _story(1, title="fresh", score=0, descendants=0)
    raw.pop("score")
    raw.pop("descendants")
    item = parse_hn_item(raw, list_name="new")
    assert item is not None
    assert item.metadata["score"] == 0
    assert item.metadata["descendants"] == 0


def test_parse_returns_none_for_non_dict():
    assert parse_hn_item(None, list_name="top") is None  # type: ignore[arg-type]
    assert parse_hn_item("nope", list_name="top") is None  # type: ignore[arg-type]


def test_parse_strips_kids_from_raw_payload():
    """Comment trees are large and irrelevant evidence for the brief — they
    must not bloat raw_payload. The descendant count is preserved in metadata."""
    raw = _story(1, title="x", extras={"kids": list(range(1000))})
    item = parse_hn_item(raw, list_name="top")
    assert item is not None
    assert "kids" not in item.raw_payload
    assert item.metadata["descendants"] == 25


def test_parse_ignores_invalid_time():
    raw = _story(1, title="x", time="not-a-number")  # type: ignore[arg-type]
    item = parse_hn_item(raw, list_name="top")
    assert item is not None
    assert item.published_at is None


# ── fetch_hn with MockTransport ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "list_name,expected_path",
    [
        ("top", "topstories.json"),
        ("best", "beststories.json"),
        ("new", "newstories.json"),
        ("show", "showstories.json"),
        ("ask", "askstories.json"),
    ],
)
async def test_fetch_routes_to_correct_list_endpoint(patch_client, list_name, expected_path):
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        url = str(request.url)
        if url.endswith(expected_path):
            return httpx.Response(200, json=[])
        # any unexpected url → 404 (but we should not see any)
        return httpx.Response(404, json={"unexpected": url})

    patch_client(handler)
    items = await fetch_hn(_task(list_name=list_name))

    assert items == []
    assert any(expected_path in u for u in captured)


async def test_fetch_returns_normalized_items(patch_client):
    routes = {
        f"{API_BASE}/topstories.json": httpx.Response(200, json=[1, 2]),
        f"{API_BASE}/item/1.json": httpx.Response(
            200, json=_story(1, title="One", url="https://x.example/1")
        ),
        f"{API_BASE}/item/2.json": httpx.Response(
            200, json=_story(2, title="Two", url="https://x.example/2")
        ),
    }
    patch_client(routes)
    items = await fetch_hn(_task())

    assert {i.dedup_key for i in items} == {"1", "2"}
    assert {i.title for i in items} == {"One", "Two"}


async def test_fetch_truncates_to_limit(patch_client):
    routes: dict[str, Any] = {
        f"{API_BASE}/topstories.json": httpx.Response(200, json=list(range(1, 11))),
    }
    for n in range(1, 11):
        routes[f"{API_BASE}/item/{n}.json"] = httpx.Response(200, json=_story(n))

    patch_client(routes)
    items = await fetch_hn(_task(limit=3))

    # Only items 1, 2, 3 should be fetched and returned
    assert {i.dedup_key for i in items} == {"1", "2", "3"}


async def test_fetch_applies_min_score_filter(patch_client):
    routes = {
        f"{API_BASE}/topstories.json": httpx.Response(200, json=[1, 2, 3]),
        f"{API_BASE}/item/1.json": httpx.Response(200, json=_story(1, score=50)),
        f"{API_BASE}/item/2.json": httpx.Response(200, json=_story(2, score=150)),
        f"{API_BASE}/item/3.json": httpx.Response(200, json=_story(3, score=100)),
    }
    patch_client(routes)
    items = await fetch_hn(_task(min_score=100))

    # Only items with score >= 100 survive
    assert {i.dedup_key for i in items} == {"2", "3"}


async def test_fetch_skips_null_item_response(patch_client):
    """Deleted HN items return JSON `null`. Skip without aborting the batch."""
    routes = {
        f"{API_BASE}/topstories.json": httpx.Response(200, json=[1, 2]),
        f"{API_BASE}/item/1.json": httpx.Response(200, json=None),
        f"{API_BASE}/item/2.json": httpx.Response(200, json=_story(2, title="alive")),
    }
    patch_client(routes)
    items = await fetch_hn(_task())

    assert {i.dedup_key for i in items} == {"2"}


async def test_per_item_failure_does_not_abort_batch(patch_client):
    """Load-bearing: one 5xx item must not poison the rest of the task."""
    routes = {
        f"{API_BASE}/topstories.json": httpx.Response(200, json=[1, 2, 3]),
        f"{API_BASE}/item/1.json": httpx.Response(200, json=_story(1, title="A")),
        f"{API_BASE}/item/2.json": httpx.Response(503, text="boom"),
        f"{API_BASE}/item/3.json": httpx.Response(200, json=_story(3, title="C")),
    }
    patch_client(routes)
    items = await fetch_hn(_task())

    assert {i.dedup_key for i in items} == {"1", "3"}


async def test_list_endpoint_failure_propagates(patch_client):
    """Failing to fetch the list itself is a task-level failure — runner records it."""

    def handler(_request):
        return httpx.Response(503, text="upstream unavailable")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_hn(_task())


async def test_list_endpoint_returns_non_array_yields_empty(patch_client):
    """If HN ever returns something unexpected, we degrade to empty rather
    than raise — the run is more useful with zero HN items than a failed run."""
    routes = {
        f"{API_BASE}/topstories.json": httpx.Response(200, json={"unexpected": "shape"}),
    }
    patch_client(routes)
    assert await fetch_hn(_task()) == []


async def test_fetch_records_ua_with_contact(patch_client):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["ua"] = request.headers.get("User-Agent")
        return httpx.Response(200, json=[])

    patch_client(handler)
    await fetch_hn(_task())
    assert "ClawFeed-Intel" in (captured["ua"] or "")
    assert "+contact:" in (captured["ua"] or "")


async def test_fetch_rejects_non_hn_task():
    from clawfeed_intel.sources import RssTask

    bad = ResolvedTask(
        task=RssTask(kind="rss", url="https://x.example/feed"),
        category="scratch",
        origin="yaml",
        source_id=None,
        source_name="x",
    )
    with pytest.raises(TypeError, match="expected HnTask"):
        await fetch_hn(bad)


# ── registration ──────────────────────────────────────────────────────────────


def test_hn_fetcher_is_registered():
    assert FETCHER_REGISTRY[KIND] is fetch_hn


def test_kind_constant_matches_source_task_discriminator():
    assert KIND == "hn"
    HnTask(kind=KIND, list="top")
