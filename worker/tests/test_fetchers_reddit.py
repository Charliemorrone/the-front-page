"""Tests for the Reddit fetcher.

Two test surfaces:

1. ``parse_reddit_listing`` is pure — exercised with hand-built ``Listing``
   JSON shapes covering link posts, self posts, removed posts, stickied
   posts, comment-type entries (defensive), missing fields, malformed JSON,
   and the load-bearing assertion that a self-post submitted to two
   different subreddits produces two distinct dedup keys (the
   cross-subreddit attention-signal property).
2. ``fetch_reddit`` uses ``httpx.MockTransport`` to assert per-sort path
   routing, ``limit`` query-param construction, 4xx / 5xx / 429
   propagation, contact-bearing UA, and non-:class:`RedditTask` rejection.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from clawfeed_intel.fetchers import FETCHER_REGISTRY
from clawfeed_intel.fetchers.reddit import (
    API_HOST,
    DEFAULT_LIMIT,
    KIND,
    fetch_reddit,
    parse_reddit_listing,
)
from clawfeed_intel.sources import RedditTask, ResolvedTask


# ── helpers ───────────────────────────────────────────────────────────────────


def _task(
    *,
    subreddit: str = "MachineLearning",
    sort: str = "hot",
    limit: int | None = None,
    source_name: str = "ai_research:r/MachineLearning",
) -> ResolvedTask:
    return ResolvedTask(
        task=RedditTask(kind="reddit", subreddit=subreddit, sort=sort, limit=limit),
        category="ai_research",
        origin="yaml",
        source_id=None,
        source_name=source_name,
    )


def _post(
    post_id: str = "abc123",
    *,
    title: str = "New paper on agentic LLMs",
    url: str | None = "https://example.com/paper",
    permalink: str = "/r/MachineLearning/comments/abc123/new_paper/",
    is_self: bool = False,
    selftext: str = "",
    author: str = "researcher42",
    score: int = 250,
    num_comments: int = 18,
    created_utc: float = 1715000000.0,
    domain: str = "example.com",
    flair: str | None = "Research",
    stickied: bool = False,
    removed_by_category: str | None = None,
    over_18: bool = False,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": post_id,
        "name": f"t3_{post_id}",
        "title": title,
        "permalink": permalink,
        "is_self": is_self,
        "selftext": selftext,
        "author": author,
        "score": score,
        "num_comments": num_comments,
        "created_utc": created_utc,
        "domain": domain,
        "stickied": stickied,
        "removed_by_category": removed_by_category,
        "over_18": over_18,
    }
    if url is not None:
        data["url"] = url
    if flair is not None:
        data["link_flair_text"] = flair
    if extras:
        data.update(extras)
    return {"kind": "t3", "data": data}


def _listing(*posts: dict[str, Any], after: str | None = None) -> dict[str, Any]:
    return {
        "kind": "Listing",
        "data": {"children": list(posts), "after": after, "before": None},
    }


@pytest.fixture
def patch_client(monkeypatch):
    """Replace ``reddit.build_client`` with a MockTransport-backed client."""

    def _patch(routes: dict[str, Any] | Any):
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if callable(routes):
                return routes(request)
            entry = routes.get(url)
            if entry is None:
                return httpx.Response(404, json={"error": "no route", "url": url})
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

        monkeypatch.setattr("clawfeed_intel.fetchers.reddit.build_client", fake_build_client)

    return _patch


# ── parse_reddit_listing (pure) ───────────────────────────────────────────────


def test_parse_link_post_with_external_url():
    listing = _listing(_post("abc123", url="https://example.com/paper?utm_source=reddit"))
    items = parse_reddit_listing(
        json.dumps(listing),
        source_name="r/MachineLearning",
        subreddit="MachineLearning",
        sort="hot",
    )
    assert len(items) == 1
    item = items[0]
    assert item.source_type == "reddit"
    assert item.dedup_key == "t3_abc123"
    assert item.title == "New paper on agentic LLMs"
    assert item.url == "https://example.com/paper?utm_source=reddit"
    assert item.canonical_url == "https://example.com/paper"
    assert item.author == "researcher42"
    assert item.metadata["subreddit"] == "MachineLearning"
    assert item.metadata["sort"] == "hot"
    assert item.metadata["post_id"] == "abc123"
    assert item.metadata["fullname"] == "t3_abc123"
    assert item.metadata["score"] == 250
    assert item.metadata["num_comments"] == 18
    assert (
        item.metadata["discussion_url"]
        == f"{API_HOST}/r/MachineLearning/comments/abc123/new_paper/"
    )
    assert item.metadata["is_self"] is False
    assert item.metadata["domain"] == "example.com"
    assert item.metadata["flair"] == "Research"
    assert item.metadata["external_url"] == "https://example.com/paper?utm_source=reddit"
    assert item.published_at == "2024-05-06T12:53:20+00:00"
    # Link post → no body text
    assert item.content == ""
    assert item.excerpt == "New paper on agentic LLMs"


def test_parse_self_post_uses_selftext_as_content_with_entities_decoded():
    """Reddit HTML-encodes selftext (``&amp;``, ``&gt;``, ``&#x200B;``).
    Decode so the LLM sees clean text. Markdown formatting stays intact."""
    listing = _listing(
        _post(
            "self01",
            title="Discussion: scaling laws",
            url=None,
            permalink="/r/MachineLearning/comments/self01/discussion/",
            is_self=True,
            selftext="What are your thoughts on scaling laws &amp; their limits?\n\n&gt; Quote here\n\n*emphasis*",
            domain="self.MachineLearning",
        )
    )
    items = parse_reddit_listing(
        listing,  # dict input also accepted
        source_name="r/MachineLearning",
        subreddit="MachineLearning",
        sort="hot",
    )
    assert len(items) == 1
    item = items[0]
    assert item.metadata["is_self"] is True
    assert "scaling laws & their limits" in item.content  # &amp; decoded
    assert "> Quote here" in item.content  # &gt; decoded
    assert "*emphasis*" in item.content  # markdown left intact
    # Self post → discussion URL is the canonical reference, no external_url
    assert item.url == f"{API_HOST}/r/MachineLearning/comments/self01/discussion/"
    assert "external_url" not in item.metadata


def test_parse_skips_removed_post():
    listing = _listing(
        _post("rm01", removed_by_category="moderator"),
        _post("ok01", title="kept"),
    )
    items = parse_reddit_listing(listing, source_name="x", subreddit="x", sort="hot")
    assert [i.title for i in items] == ["kept"]


def test_parse_skips_stickied_post():
    """Stickied posts are usually subreddit rules / weekly-thread headers,
    not news signals."""
    listing = _listing(
        _post("st01", title="Weekly Thread", stickied=True),
        _post("ok01", title="real post"),
    )
    items = parse_reddit_listing(listing, source_name="x", subreddit="x", sort="hot")
    assert [i.title for i in items] == ["real post"]


def test_parse_skips_untitled_post():
    listing = _listing(_post("ut01", title=""), _post("ok01", title="kept"))
    items = parse_reddit_listing(listing, source_name="x", subreddit="x", sort="hot")
    assert [i.title for i in items] == ["kept"]


def test_parse_skips_comment_kind():
    """Listing endpoints shouldn't return comments, but be defensive."""
    comment = {"kind": "t1", "data": {"id": "c01", "body": "a reply"}}
    listing = {"kind": "Listing", "data": {"children": [comment]}}
    items = parse_reddit_listing(listing, source_name="x", subreddit="x", sort="hot")
    assert items == []


def test_parse_cross_subreddit_yields_distinct_dedup_keys():
    """Same external article submitted to two subs → two distinct
    attention signals. Cross-source folding happens at the clustering layer
    via content_hash, not here."""
    a = _listing(_post("aaa", title="Article", url="https://example.com/article"))
    b = _listing(_post("bbb", title="Article", url="https://example.com/article"))
    items_a = parse_reddit_listing(a, source_name="x", subreddit="MachineLearning", sort="hot")
    items_b = parse_reddit_listing(b, source_name="y", subreddit="LocalLLaMA", sort="hot")
    assert len(items_a) == 1
    assert len(items_b) == 1
    assert items_a[0].dedup_key != items_b[0].dedup_key
    # Same canonical URL though, which is what content_hash will fold on
    assert items_a[0].canonical_url == items_b[0].canonical_url
    # Each gets its own subreddit metadata
    assert items_a[0].metadata["subreddit"] == "MachineLearning"
    assert items_b[0].metadata["subreddit"] == "LocalLLaMA"


def test_parse_handles_missing_optional_fields():
    """Reddit occasionally omits flair, score, num_comments, domain, etc."""
    minimal = {
        "kind": "t3",
        "data": {
            "id": "min01",
            "name": "t3_min01",
            "title": "minimal post",
            "permalink": "/r/x/comments/min01/_/",
            "url": "https://x.example/a",
            "is_self": False,
            "created_utc": 1715000000,
            "removed_by_category": None,
            "stickied": False,
        },
    }
    listing = {"kind": "Listing", "data": {"children": [minimal]}}
    items = parse_reddit_listing(listing, source_name="x", subreddit="x", sort="hot")
    assert len(items) == 1
    item = items[0]
    assert item.metadata["score"] == 0
    assert item.metadata["num_comments"] == 0
    assert "flair" not in item.metadata


def test_parse_falls_back_to_constructed_fullname():
    """If `name` is missing but `id` is present, build the fullname."""
    raw = {
        "kind": "t3",
        "data": {
            "id": "fb01",
            # no `name` key
            "title": "fallback post",
            "permalink": "/r/x/comments/fb01/_/",
            "url": "https://x.example/a",
            "is_self": False,
            "removed_by_category": None,
            "stickied": False,
        },
    }
    listing = {"kind": "Listing", "data": {"children": [raw]}}
    items = parse_reddit_listing(listing, source_name="x", subreddit="x", sort="hot")
    assert len(items) == 1
    assert items[0].dedup_key == "t3_fb01"


def test_parse_returns_empty_on_non_listing_kind():
    """Reddit error responses can return ``{"error": 403, ...}`` with no
    ``kind`` field — degrade quietly."""
    items = parse_reddit_listing({"error": 403}, source_name="x", subreddit="x", sort="hot")
    assert items == []


def test_parse_returns_empty_on_missing_children():
    payload = {"kind": "Listing", "data": {}}
    items = parse_reddit_listing(payload, source_name="x", subreddit="x", sort="hot")
    assert items == []


def test_parse_returns_empty_on_malformed_json():
    items = parse_reddit_listing("{not valid", source_name="x", subreddit="x", sort="hot")
    assert items == []


def test_parse_returns_empty_on_empty_body():
    assert parse_reddit_listing("", source_name="x", subreddit="x", sort="hot") == []
    assert parse_reddit_listing("   ", source_name="x", subreddit="x", sort="hot") == []


def test_parse_strips_selftext_html_from_raw_payload():
    """selftext_html is 5-10× the size of selftext (which we keep in content)
    — must not bloat raw_payload."""
    listing = _listing(
        _post(
            "big01",
            is_self=True,
            url=None,
            selftext="hello",
            extras={"selftext_html": "<div>" + ("<p>filler</p>" * 1000) + "</div>"},
        )
    )
    items = parse_reddit_listing(listing, source_name="x", subreddit="x", sort="hot")
    assert "selftext_html" not in items[0].raw_payload
    assert "selftext" in items[0].raw_payload  # plain text kept


def test_parse_drops_skipped_post_with_no_url_and_no_permalink():
    raw = {
        "kind": "t3",
        "data": {
            "id": "nu01",
            "name": "t3_nu01",
            "title": "no urls anywhere",
            "permalink": "",
            "is_self": False,
            "removed_by_category": None,
            "stickied": False,
        },
    }
    listing = {"kind": "Listing", "data": {"children": [raw]}}
    items = parse_reddit_listing(listing, source_name="x", subreddit="x", sort="hot")
    assert items == []


def test_parse_handles_invalid_created_utc():
    listing = _listing(_post("t01", created_utc="not-a-number"))  # type: ignore[arg-type]
    items = parse_reddit_listing(listing, source_name="x", subreddit="x", sort="hot")
    assert len(items) == 1
    assert items[0].published_at is None


def test_parse_records_over_18_flag():
    listing = _listing(_post("nsfw01", over_18=True), _post("sfw01", over_18=False))
    items = parse_reddit_listing(listing, source_name="x", subreddit="x", sort="hot")
    assert len(items) == 2
    nsfw, sfw = items
    assert nsfw.metadata["over_18"] is True
    assert sfw.metadata["over_18"] is False


# ── fetch_reddit with MockTransport ───────────────────────────────────────────


@pytest.mark.parametrize("sort", ["hot", "new", "top", "rising"])
async def test_fetch_routes_to_correct_sort_path(patch_client, sort, conn):
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        return httpx.Response(200, json=_listing())

    patch_client(handler)
    await fetch_reddit(conn, _task(sort=sort))

    assert len(captured) == 1
    assert f"/r/MachineLearning/{sort}.json" in captured[0]


async def test_fetch_passes_default_limit_when_unset(patch_client, conn):
    captured: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url)
        return httpx.Response(200, json=_listing())

    patch_client(handler)
    await fetch_reddit(conn, _task())  # limit=None
    assert dict(captured[0].params)["limit"] == str(DEFAULT_LIMIT)


async def test_fetch_passes_custom_limit_when_set(patch_client, conn):
    captured: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url)
        return httpx.Response(200, json=_listing())

    patch_client(handler)
    await fetch_reddit(conn, _task(limit=25))
    assert dict(captured[0].params)["limit"] == "25"


async def test_fetch_url_encodes_subreddit_name(patch_client, conn):
    """Cheap insurance against unusual chars in config typos."""
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        return httpx.Response(200, json=_listing())

    patch_client(handler)
    await fetch_reddit(conn, _task(subreddit="weird name"))
    assert "weird%20name" in captured[0]


async def test_fetch_returns_normalized_items(patch_client, conn):
    body = _listing(
        _post("a01", title="Alpha", url="https://x.example/a"),
        _post("b02", title="Beta", url="https://y.example/b"),
    )

    def handler(_request):
        return httpx.Response(200, json=body)

    patch_client(handler)
    items = await fetch_reddit(conn, _task())
    assert {i.title for i in items} == {"Alpha", "Beta"}
    # The fetcher injects the resolved subreddit + sort into per-item metadata
    assert all(i.metadata["subreddit"] == "MachineLearning" for i in items)
    assert all(i.metadata["sort"] == "hot" for i in items)


async def test_fetch_5xx_propagates_as_http_status_error(patch_client, conn):
    def handler(_request):
        return httpx.Response(503, text="upstream unavailable")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_reddit(conn, _task())


async def test_fetch_429_rate_limit_propagates(patch_client, conn):
    """Reddit returns 429 when over quota. Daily-cadence shouldn't trip it,
    but if it does, we want the runner to record `failed` (not silently
    drop the subreddit). Backoff is a future concern."""

    def handler(_request):
        return httpx.Response(429, text="too many requests")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_reddit(conn, _task())


async def test_fetch_403_propagates(patch_client, conn):
    """Private/banned subs return 403 — surface as failed so coverage is honest."""

    def handler(_request):
        return httpx.Response(403, text="forbidden")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_reddit(conn, _task(subreddit="privatesub"))


async def test_fetch_records_ua_with_contact(patch_client, conn):
    """Reddit's API guidance asks for a UA with contact info."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["ua"] = request.headers.get("User-Agent")
        return httpx.Response(200, json=_listing())

    patch_client(handler)
    await fetch_reddit(conn, _task())
    assert "ClawFeed-Intel" in (captured["ua"] or "")
    assert "+contact:" in (captured["ua"] or "")


async def test_fetch_rejects_non_reddit_task(conn):
    from clawfeed_intel.sources import RssTask

    bad = ResolvedTask(
        task=RssTask(kind="rss", url="https://x.example/feed"),
        category="scratch",
        origin="yaml",
        source_id=None,
        source_name="x",
    )
    with pytest.raises(TypeError, match="expected RedditTask"):
        await fetch_reddit(conn, bad)


# ── registration ──────────────────────────────────────────────────────────────


def test_reddit_fetcher_is_registered():
    assert FETCHER_REGISTRY[KIND] is fetch_reddit


def test_kind_constant_matches_source_task_discriminator():
    assert KIND == "reddit"
    RedditTask(kind=KIND, subreddit="MachineLearning")
