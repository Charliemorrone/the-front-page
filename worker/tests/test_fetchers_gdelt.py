"""Tests for the GDELT DOC 2.0 fetcher.

Two test surfaces:

1. ``parse_gdelt_response`` is pure — exercised with hand-built JSON and
   dict fixtures: regular articles, missing optional fields, malformed
   ``seendate`` values, syndicated UTM-tagged URLs collapsing to one
   dedup_key, missing ``articles`` key, malformed JSON body, and entries
   with control characters in titles (a real-world GDELT quirk).
2. ``fetch_gdelt`` uses ``httpx.MockTransport`` to assert the query URL is
   constructed correctly (every required GDELT param present), 5xx
   propagates as :class:`httpx.HTTPStatusError`, the contact-bearing UA is
   sent, and a non-:class:`GdeltTask` raises :class:`TypeError`.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from clawfeed_intel.fetchers import FETCHER_REGISTRY
from clawfeed_intel.fetchers.gdelt import (
    API_URL,
    DEFAULT_TIMESPAN,
    KIND,
    MAX_RECORDS,
    fetch_gdelt,
    parse_gdelt_response,
)
from clawfeed_intel.sources import GdeltTask, ResolvedTask


# ── helpers ───────────────────────────────────────────────────────────────────


def _task(query: str = "AI funding", source_name: str = "startup_funding:gdelt") -> ResolvedTask:
    return ResolvedTask(
        task=GdeltTask(kind="gdelt", query=query),
        category="startup_funding",
        origin="yaml",
        source_id=None,
        source_name=source_name,
    )


def _article(
    *,
    url: str = "https://example.com/article-1",
    title: str = "AI startup raises $100M Series B",
    seendate: str | None = "20260504T123456Z",
    domain: str = "example.com",
    language: str = "English",
    sourcecountry: str = "United States",
    socialimage: str | None = "https://example.com/og-image.jpg",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "url": url,
        "title": title,
        "domain": domain,
        "language": language,
        "sourcecountry": sourcecountry,
    }
    if seendate is not None:
        payload["seendate"] = seendate
    if socialimage is not None:
        payload["socialimage"] = socialimage
    if extras:
        payload.update(extras)
    return payload


@pytest.fixture
def patch_client(monkeypatch):
    """Replace ``gdelt.build_client`` with a MockTransport-backed client."""

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

        monkeypatch.setattr("clawfeed_intel.fetchers.gdelt.build_client", fake_build_client)

    return _patch


# ── parse_gdelt_response (pure) ───────────────────────────────────────────────


def test_parse_single_article_full_fields():
    payload = {"articles": [_article()]}
    items = parse_gdelt_response(
        json.dumps(payload),
        source_name="startup_funding:gdelt",
        query="AI funding",
        query_url="https://api.example/q",
    )

    assert len(items) == 1
    item = items[0]
    assert item.source_type == "gdelt"
    assert item.title == "AI startup raises $100M Series B"
    assert item.url == "https://example.com/article-1"
    assert item.canonical_url == "https://example.com/article-1"
    assert item.dedup_key == item.canonical_url
    assert item.content == ""  # GDELT carries no body
    assert item.excerpt == "AI startup raises $100M Series B"
    assert item.author == ""
    assert item.published_at == "2026-05-04T12:34:56+00:00"
    assert item.content_hash is not None
    assert item.metadata["domain"] == "example.com"
    assert item.metadata["language"] == "English"
    assert item.metadata["source_country"] == "United States"
    assert item.metadata["social_image"] == "https://example.com/og-image.jpg"
    assert item.metadata["query"] == "AI funding"
    assert item.metadata["query_url"] == "https://api.example/q"
    # raw_payload preserves the original article verbatim
    assert item.raw_payload["url"] == "https://example.com/article-1"


def test_parse_accepts_dict_input():
    """Tests prefer dicts; fetcher in production hands strings. Both must work."""
    payload = {"articles": [_article()]}
    items = parse_gdelt_response(payload, source_name="x")
    assert len(items) == 1
    assert items[0].title == "AI startup raises $100M Series B"


def test_parse_skips_article_without_url():
    payload = {"articles": [_article(url=""), _article(url="https://x.example/keep", title="kept")]}
    items = parse_gdelt_response(payload, source_name="x")
    assert [i.title for i in items] == ["kept"]


def test_parse_skips_article_without_title():
    """Some upstream syndication blanks the title; without it, the article is
    useless for the brief — skip silently."""
    payload = {"articles": [_article(title=""), _article(url="https://x.example/keep", title="ok")]}
    items = parse_gdelt_response(payload, source_name="x")
    assert [i.title for i in items] == ["ok"]


def test_parse_dedup_key_collapses_utm_syndication():
    """Same article syndicated to two trackers shows up as one dedup_key.
    This is the load-bearing reason we use canonical_url, not url."""
    payload = {
        "articles": [
            _article(url="https://example.com/post?utm_source=newsletter&utm_medium=email"),
            _article(url="https://example.com/post?fbclid=XYZ"),
        ]
    }
    items = parse_gdelt_response(payload, source_name="x")
    assert len(items) == 2
    assert items[0].dedup_key == items[1].dedup_key == "https://example.com/post"


def test_parse_handles_missing_seendate():
    payload = {"articles": [_article(seendate=None)]}
    items = parse_gdelt_response(payload, source_name="x")
    assert len(items) == 1
    assert items[0].published_at is None


def test_parse_handles_malformed_seendate():
    payload = {"articles": [_article(seendate="not-a-timestamp")]}
    items = parse_gdelt_response(payload, source_name="x")
    assert len(items) == 1
    assert items[0].published_at is None


def test_parse_handles_missing_optional_fields():
    raw = {"url": "https://x.example/a", "title": "minimal"}
    items = parse_gdelt_response({"articles": [raw]}, source_name="x")
    assert len(items) == 1
    item = items[0]
    assert item.metadata["domain"] == ""
    assert item.metadata["language"] == ""
    assert item.metadata["source_country"] == ""
    assert "social_image" not in item.metadata
    assert item.published_at is None


def test_parse_returns_empty_on_missing_articles_key():
    items = parse_gdelt_response({"status": "ok"}, source_name="x")
    assert items == []


def test_parse_returns_empty_on_articles_not_a_list():
    items = parse_gdelt_response({"articles": "broken"}, source_name="x")
    assert items == []


def test_parse_returns_empty_on_malformed_json():
    items = parse_gdelt_response("{not valid json", source_name="x")
    assert items == []


def test_parse_returns_empty_on_empty_body():
    assert parse_gdelt_response("", source_name="x") == []
    assert parse_gdelt_response("   ", source_name="x") == []


def test_parse_returns_empty_on_unexpected_input_type():
    assert parse_gdelt_response(123, source_name="x") == []  # type: ignore[arg-type]
    assert parse_gdelt_response(None, source_name="x") == []  # type: ignore[arg-type]


def test_parse_tolerates_control_chars_in_titles():
    """Real-world GDELT quirk: titles occasionally contain raw control chars
    that strict json.loads rejects. The fallback parse should rescue them."""
    body = '{"articles": [{"url": "https://x.example/a", "title": "AI\x01quirk"}]}'
    items = parse_gdelt_response(body, source_name="x")
    assert len(items) == 1
    assert "AI" in items[0].title


def test_parse_skips_non_dict_article_entry():
    payload = {"articles": ["not a dict", _article(url="https://x.example/keep", title="ok")]}
    items = parse_gdelt_response(payload, source_name="x")
    assert [i.title for i in items] == ["ok"]


def test_parse_omits_empty_query_metadata_keys():
    payload = {"articles": [_article()]}
    items = parse_gdelt_response(payload, source_name="x")  # no query / query_url
    assert "query" not in items[0].metadata
    assert "query_url" not in items[0].metadata


# ── fetch_gdelt with MockTransport ────────────────────────────────────────────


async def test_fetch_constructs_query_url_with_all_required_params(patch_client):
    captured: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url)
        return httpx.Response(200, json={"articles": []})

    patch_client(handler)
    await fetch_gdelt(_task(query="AI funding"))

    assert len(captured) == 1
    url = captured[0]
    assert str(url).startswith(API_URL + "?")
    params = dict(url.params)
    assert params["query"] == "AI funding"
    assert params["mode"] == "ArtList"
    assert params["format"] == "JSON"
    assert params["timespan"] == DEFAULT_TIMESPAN
    assert params["maxrecords"] == str(MAX_RECORDS)
    assert params["sort"] == "DateDesc"


async def test_fetch_returns_normalized_items(patch_client):
    body = {
        "articles": [
            _article(url="https://x.example/a", title="A"),
            _article(url="https://y.example/b", title="B"),
        ]
    }

    def handler(_request):
        return httpx.Response(200, json=body)

    patch_client(handler)
    items = await fetch_gdelt(_task())

    assert {i.title for i in items} == {"A", "B"}
    # The fetcher injects the resolved query+query_url into per-item metadata
    assert all(item.metadata["query"] == "AI funding" for item in items)
    assert all(item.metadata["query_url"].startswith(API_URL + "?") for item in items)


async def test_fetch_5xx_propagates_as_http_status_error(patch_client):
    """Failing the GDELT call is a task-level failure — runner records `failed`."""

    def handler(_request):
        return httpx.Response(503, text="upstream unavailable")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_gdelt(_task())


async def test_fetch_4xx_propagates_as_http_status_error(patch_client):
    def handler(_request):
        return httpx.Response(400, text="bad request")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_gdelt(_task())


async def test_fetch_degrades_to_empty_on_unexpected_body(patch_client):
    """If GDELT returns 200 with a body that has no `articles`, we get [],
    not an exception. The run is more useful with zero GDELT items than failed."""

    def handler(_request):
        return httpx.Response(200, json={"status": "noresults"})

    patch_client(handler)
    items = await fetch_gdelt(_task())
    assert items == []


async def test_fetch_records_ua_with_contact(patch_client):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["ua"] = request.headers.get("User-Agent")
        return httpx.Response(200, json={"articles": []})

    patch_client(handler)
    await fetch_gdelt(_task())
    assert "ClawFeed-Intel" in (captured["ua"] or "")
    assert "+contact:" in (captured["ua"] or "")


async def test_fetch_rejects_non_gdelt_task():
    from clawfeed_intel.sources import RssTask

    bad = ResolvedTask(
        task=RssTask(kind="rss", url="https://x.example/feed"),
        category="scratch",
        origin="yaml",
        source_id=None,
        source_name="x",
    )
    with pytest.raises(TypeError, match="expected GdeltTask"):
        await fetch_gdelt(bad)


# ── registration ──────────────────────────────────────────────────────────────


def test_gdelt_fetcher_is_registered():
    assert FETCHER_REGISTRY[KIND] is fetch_gdelt


def test_kind_constant_matches_source_task_discriminator():
    assert KIND == "gdelt"
    GdeltTask(kind=KIND, query="anything")
