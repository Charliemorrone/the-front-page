"""Tests for the RSS / Atom fetcher.

Two test surfaces:

1. ``parse_feed_text`` is a pure function — exercised with static fixture
   XML to pin extraction behaviour across feed shapes (RSS 2.0, Atom,
   missing fields, malformed input).
2. ``fetch_rss`` does the HTTP — exercised with ``httpx.MockTransport``,
   never live network. The trafilatura fallback is patched at the
   ``_trafilatura_extract`` boundary so we don't pull in trafilatura's own
   behaviour or fetch real article pages.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import pytest

from clawfeed_intel.fetchers import FETCHER_REGISTRY
from clawfeed_intel.fetchers import rss as rss_mod
from clawfeed_intel.fetchers.rss import KIND, fetch_rss, parse_feed_text
from clawfeed_intel.sources import ResolvedTask, RssTask


# ── fixtures ──────────────────────────────────────────────────────────────────


RSS_2_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Example Feed</title>
  <link>https://example.com/</link>
  <description>An example feed</description>
  <item>
    <title>First post</title>
    <link>https://example.com/posts/1?utm_source=rss&amp;utm_campaign=daily</link>
    <pubDate>Sun, 04 May 2026 12:00:00 GMT</pubDate>
    <description><![CDATA[<p>The first <b>summary</b> with markup.</p>]]></description>
    <author>alice@example.com (Alice)</author>
    <category>news</category>
    <category>ai</category>
  </item>
  <item>
    <title>Second post</title>
    <link>https://example.com/posts/2</link>
    <pubDate>Sun, 04 May 2026 13:00:00 GMT</pubDate>
    <description>Plain summary.</description>
  </item>
</channel>
</rss>
"""

ATOM_FIXTURE = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <id>urn:atom:example</id>
  <updated>2026-05-04T12:00:00Z</updated>
  <entry>
    <id>urn:atom:1</id>
    <title>Atom Entry</title>
    <link href="https://atom.example.com/post/1"/>
    <updated>2026-05-04T12:00:00Z</updated>
    <published>2026-05-04T11:00:00Z</published>
    <author><name>Bob</name></author>
    <content type="html">&lt;p&gt;Atom content body with &lt;b&gt;tags&lt;/b&gt; preserved.&lt;/p&gt;</content>
  </entry>
</feed>
"""


def _task(
    url: str = "https://example.com/feed",
    source_name: str = "feed",
    source_id: int | None = None,
) -> ResolvedTask:
    return ResolvedTask(
        task=RssTask(kind="rss", url=url),
        category="scratch",
        origin="db" if source_id is not None else "yaml",
        source_id=source_id,
        source_name=source_name,
    )


@pytest.fixture
def patch_client(monkeypatch):
    """Replace ``rss.build_client`` with one that uses an httpx.MockTransport.

    Returns a function that takes a request handler. The handler signature
    is the standard ``httpx.MockTransport`` one: ``(request) -> Response``.
    """

    def _patch(handler):
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

        monkeypatch.setattr("clawfeed_intel.fetchers.rss.build_client", fake_build_client)

    return _patch


# ── parse_feed_text (pure) ────────────────────────────────────────────────────


def test_parse_rss2_extracts_two_entries():
    items = parse_feed_text(
        RSS_2_FIXTURE, source_name="example", feed_url="https://example.com/feed"
    )

    assert len(items) == 2
    first = items[0]
    assert first.source_type == "rss"
    assert first.title == "First post"
    assert first.author == "alice@example.com (Alice)"
    assert first.published_at == "2026-05-04T12:00:00+00:00"
    # Tracking params stripped via canonicalize_url
    assert first.canonical_url == "https://example.com/posts/1"
    assert first.dedup_key == "https://example.com/posts/1"
    # HTML stripped from content
    assert "summary with markup" in first.content
    assert "<b>" not in first.content
    assert first.excerpt and first.excerpt[:5] == first.content[:5]
    assert first.content_hash is not None
    assert first.metadata["feed_url"] == "https://example.com/feed"
    assert first.metadata["feed_title"] == "Example Feed"
    assert first.metadata["tags"] == ["news", "ai"]
    # Original HTML preserved for forensics
    assert "<b>summary</b>" in first.raw_payload["summary_html"]


def test_parse_atom_uses_content_block_and_published_over_updated():
    items = parse_feed_text(ATOM_FIXTURE, source_name="atom")
    assert len(items) == 1
    item = items[0]
    assert item.title == "Atom Entry"
    assert item.canonical_url == "https://atom.example.com/post/1"
    # Atom content (richer than summary) is preferred
    assert "atom content body" in item.content.lower()
    assert "tags preserved" in item.content.lower()
    assert "<b>" not in item.content
    # published wins over updated when both are present
    assert item.published_at == "2026-05-04T11:00:00+00:00"
    assert item.author == "Bob"


def test_parse_skips_entries_without_link():
    body = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>No link</title><description>...</description></item>
<item><title>Has link</title><link>https://x.example/a</link><description>...</description></item>
</channel></rss>"""
    items = parse_feed_text(body, source_name="x")
    assert [i.title for i in items] == ["Has link"]


def test_parse_empty_feed_returns_empty():
    body = "<?xml version='1.0'?><rss version='2.0'><channel></channel></rss>"
    assert parse_feed_text(body, source_name="empty") == []


def test_parse_garbage_input_returns_empty_not_raises():
    """feedparser is permissive; garbage produces no entries but no exception."""
    assert parse_feed_text("not xml at all", source_name="bad") == []
    assert parse_feed_text("", source_name="bad") == []


def test_dedup_key_collapses_tracking_variants():
    body = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>A</title><link>https://x.example/a?utm_source=newsletter</link><description>x</description></item>
<item><title>A2</title><link>https://x.example/a?fbclid=abc</link><description>y</description></item>
</channel></rss>"""
    items = parse_feed_text(body, source_name="x")
    keys = {i.dedup_key for i in items}
    assert keys == {"https://x.example/a"}


def test_published_falls_back_to_updated_when_no_published():
    """If only <updated> is set (no <published>), use it."""
    body = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>x</id><title>X</title><link href="https://x.example/x"/>
<updated>2026-05-04T08:00:00Z</updated><summary>x</summary></entry>
</feed>"""
    items = parse_feed_text(body, source_name="x")
    assert len(items) == 1
    assert items[0].published_at == "2026-05-04T08:00:00+00:00"


# ── fetch_rss with MockTransport ──────────────────────────────────────────────


async def test_fetch_rss_returns_items_via_mock_transport(patch_client, monkeypatch):
    """Both fixture entries have thin summaries, so the trafilatura fallback
    would otherwise fire — patch it out so this test is focused on the feed
    HTTP path. The fallback has its own dedicated tests below."""

    async def no_fallback(_client, _url):
        return None

    monkeypatch.setattr(rss_mod, "_trafilatura_extract", no_fallback)

    captured: dict = {"urls": []}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["urls"].append(str(request.url))
        captured["ua"] = request.headers.get("User-Agent")
        return httpx.Response(
            200, text=RSS_2_FIXTURE, headers={"Content-Type": "application/rss+xml"}
        )

    patch_client(handler)
    items = await fetch_rss(_task())

    assert len(items) == 2
    assert "ClawFeed-Intel" in (captured.get("ua") or "")
    assert "+contact:" in (captured.get("ua") or "")
    assert captured["urls"][0] == "https://example.com/feed"


async def test_fetch_rss_raises_on_http_error(patch_client):
    """5xx must raise so the runner records a 'failed' outcome."""

    def handler(_request):
        return httpx.Response(503, text="upstream unavailable")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_rss(_task())


async def test_fetch_rss_raises_on_404(patch_client):
    def handler(_request):
        return httpx.Response(404, text="not found")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_rss(_task())


async def test_fetch_rss_rejects_non_rss_task():
    from clawfeed_intel.sources import GdeltTask

    bad_task = ResolvedTask(
        task=GdeltTask(kind="gdelt", query="anything"),
        category="scratch",
        origin="yaml",
        source_id=None,
        source_name="x",
    )
    with pytest.raises(TypeError, match="expected RssTask"):
        await fetch_rss(bad_task)


# ── trafilatura fallback ──────────────────────────────────────────────────────


THIN_FEED = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>Big article</title><link>https://x.example/article</link>
<description>tiny.</description><pubDate>Sun, 04 May 2026 12:00:00 GMT</pubDate></item>
</channel></rss>"""

LONG_FEED = (
    """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>X</title><link>https://x.example/x</link>
<description>"""
    + ("A" * 800)
    + """</description></item>
</channel></rss>"""
)


async def test_trafilatura_fallback_replaces_thin_summary(patch_client, monkeypatch):
    def handler(_request):
        return httpx.Response(200, text=THIN_FEED)

    patch_client(handler)

    full_text = "Full extracted article body. " * 30  # well over THIN_SUMMARY_CHARS

    async def fake_extract(_client, _url):
        return full_text

    monkeypatch.setattr(rss_mod, "_trafilatura_extract", fake_extract)

    items = await fetch_rss(_task("https://x.example/feed"))
    assert len(items) == 1
    item = items[0]
    assert item.content == full_text
    assert item.metadata.get("trafilatura") is True
    assert len(item.content) >= rss_mod.THIN_SUMMARY_CHARS
    # excerpt regenerated from new content
    assert item.excerpt == full_text[: rss_mod.EXCERPT_CHARS]


async def test_trafilatura_skipped_when_summary_long_enough(patch_client, monkeypatch):
    def handler(_request):
        return httpx.Response(200, text=LONG_FEED)

    patch_client(handler)

    called = {"n": 0}

    async def fake_extract(_client, _url):
        called["n"] += 1
        return "should not be used"

    monkeypatch.setattr(rss_mod, "_trafilatura_extract", fake_extract)
    items = await fetch_rss(_task("https://x.example/feed"))

    assert len(items) == 1
    assert items[0].metadata.get("trafilatura") is None
    assert called["n"] == 0


async def test_trafilatura_failure_falls_back_to_summary(patch_client, monkeypatch):
    """A failed extraction must not abort the run; we keep the original summary."""

    def handler(_request):
        return httpx.Response(200, text=THIN_FEED)

    patch_client(handler)

    async def fake_extract(_client, _url):
        return None  # extractor failed / page was non-extractable

    monkeypatch.setattr(rss_mod, "_trafilatura_extract", fake_extract)
    items = await fetch_rss(_task("https://x.example/feed"))

    assert len(items) == 1
    assert items[0].content == "tiny."
    assert items[0].metadata.get("trafilatura") is None


async def test_trafilatura_extract_returns_none_on_4xx(patch_client):
    """Direct test of _trafilatura_extract: a 4xx article URL returns None."""
    article_responses = {
        "https://x.example/blocked": httpx.Response(403, text="forbidden"),
    }

    def handler(request):
        return article_responses.get(str(request.url), httpx.Response(404))

    patch_client(handler)

    # Build a client through the patched factory so we hit the mock transport.
    async with rss_mod.build_client() as client:
        result = await rss_mod._trafilatura_extract(client, "https://x.example/blocked")
    assert result is None


# ── registration ──────────────────────────────────────────────────────────────


def test_rss_fetcher_is_registered():
    """Importing the fetchers package registers RSS in FETCHER_REGISTRY."""
    assert FETCHER_REGISTRY[KIND] is fetch_rss


def test_kind_constant_matches_source_task_discriminator():
    """The fetcher's KIND must match the RssTask discriminator value, otherwise
    SourcePlan.tasks_by_kind() routing breaks silently."""
    assert KIND == "rss"
    # Constructing the task with the same discriminator value must work.
    RssTask(kind=KIND, url="https://x.example/feed")
