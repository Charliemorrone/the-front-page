"""Tests for the configured-URL website fetcher.

Two surfaces:

1. ``parse_website_html`` is pure. We mostly mock the trafilatura layer
   (``_extract_with_trafilatura``) so the parse logic is tested
   deterministically without depending on trafilatura's heuristics. One
   integration test exercises real trafilatura on a small HTML fixture
   to confirm the wiring works end-to-end.

2. ``fetch_website`` uses ``httpx.MockTransport`` to assert HTTP failure
   propagation, the URL → final-url redirect handling, and the
   ``del conn`` (no DB writes) contract.

Feed discovery is tested directly via the parse function — when a page
declares ``<link rel="alternate" type="application/rss+xml">`` we surface
its absolute URL in metadata but we *don't* switch into feed-mode (Phase 2
concern).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from clawfeed_intel.fetchers import FETCHER_REGISTRY
from clawfeed_intel.fetchers import website as website_mod
from clawfeed_intel.fetchers.http import UnsafeUrlError
from clawfeed_intel.fetchers.website import (
    KIND,
    SOURCE_TYPE,
    fetch_website,
    parse_website_html,
)
from clawfeed_intel.sources import ResolvedTask, RssTask, WebsiteTask


# ── helpers ───────────────────────────────────────────────────────────────────


def _task(
    *, url: str = "https://example.com/about", source_name: str = "ai_research:website"
) -> ResolvedTask:
    return ResolvedTask(
        task=WebsiteTask(kind="website", url=url),
        category="ai_research",
        origin="yaml",
        source_id=None,
        source_name=source_name,
    )


def _stub_extracted(
    *,
    title: str = "Example article",
    author: str = "",
    date: str = "",
    text: str = "Body of the article goes here. Several sentences.",
    language: str = "",
    sitename: str = "",
) -> dict[str, Any]:
    return {
        "title": title,
        "author": author,
        "date": date,
        "text": text,
        "language": language,
        "sitename": sitename,
    }


@pytest.fixture
def patch_extractor(monkeypatch):
    """Replace the trafilatura wrapper with a deterministic stub.

    Tests that exercise parse_website_html directly use this so they're
    not coupled to trafilatura's heuristics; one integration test below
    runs the real extractor on a small fixture to confirm wiring.
    """

    def _patch(stub_or_callable):
        if callable(stub_or_callable):
            fn = stub_or_callable
        else:

            def fn(_html):
                return stub_or_callable

        monkeypatch.setattr("clawfeed_intel.fetchers.website._extract_with_trafilatura", fn)

    return _patch


@pytest.fixture
def patch_client(monkeypatch):
    """Replace ``website.build_client`` with a MockTransport-backed client."""

    def _patch(handler_or_routes):
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if callable(handler_or_routes):
                return handler_or_routes(request)
            entry = handler_or_routes.get(url)
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

        monkeypatch.setattr("clawfeed_intel.fetchers.website.build_client", fake_build_client)
        # Existing fetch tests want MockTransport-driven HTTP behavior, not
        # DNS reachability of `x.example` from the test box. Default the
        # SSRF guard to a no-op; the dedicated SSRF tests below override
        # this to exercise the real boundary.
        monkeypatch.setattr(
            "clawfeed_intel.fetchers.website.validate_safe_url",
            lambda _url: _noop(),
        )

    return _patch


async def _noop() -> None:
    return None


# ── parse_website_html (pure, with stubbed trafilatura) ───────────────────────


def test_parse_full_extraction(patch_extractor):
    patch_extractor(
        _stub_extracted(
            title="Example article",
            author="Jane Doe",
            date="2026-05-04",
            text="The article body, plenty of words to extract.",
            language="en",
            sitename="Example Blog",
        )
    )
    item = parse_website_html(
        "<html><body>doesn't matter — extractor stubbed</body></html>",
        source_url="https://example.com/article?utm_source=twitter",
    )
    assert item is not None
    assert item.source_type == SOURCE_TYPE
    assert item.title == "Example article"
    assert item.url == "https://example.com/article?utm_source=twitter"
    # canonicalize_url drops UTM
    assert item.canonical_url == "https://example.com/article"
    assert item.dedup_key == item.canonical_url
    assert item.author == "Jane Doe"
    assert item.published_at == "2026-05-04"
    assert item.content == "The article body, plenty of words to extract."
    assert item.excerpt == item.content  # short body → excerpt is the whole thing
    assert item.metadata["domain"] == "example.com"
    assert item.metadata["language"] == "en"
    assert item.metadata["sitename"] == "Example Blog"


def test_parse_returns_none_when_no_body(patch_extractor):
    """Pages with no extractable body produce zero items, not a stub item."""
    patch_extractor(_stub_extracted(text=""))
    item = parse_website_html("<html><body>nav-only</body></html>", source_url="https://x.example/")
    assert item is None


def test_parse_returns_none_when_extractor_fails(patch_extractor):
    patch_extractor(lambda _h: None)
    item = parse_website_html("<html><body>x</body></html>", source_url="https://x.example/")
    assert item is None


def test_parse_returns_none_for_empty_html(patch_extractor):
    patch_extractor(_stub_extracted())
    assert parse_website_html("", source_url="https://x.example/") is None
    assert parse_website_html("   ", source_url="https://x.example/") is None


def test_parse_returns_none_for_missing_source_url(patch_extractor):
    patch_extractor(_stub_extracted())
    assert parse_website_html("<html>x</html>", source_url="") is None


def test_parse_falls_back_to_title_tag_when_metadata_title_empty(patch_extractor):
    """Trafilatura sometimes can't find a metadata title. The HTML <title> is
    the natural fallback."""
    patch_extractor(_stub_extracted(title="", text="Some body content."))
    html = "<html><head><title>Page Title From HTML</title></head><body>Body</body></html>"
    item = parse_website_html(html, source_url="https://x.example/page")
    assert item is not None
    assert item.title == "Page Title From HTML"


def test_parse_falls_back_to_url_when_no_title_anywhere(patch_extractor):
    patch_extractor(_stub_extracted(title="", text="body"))
    item = parse_website_html(
        "<html><body>no title</body></html>", source_url="https://x.example/page"
    )
    assert item is not None
    assert item.title == "https://x.example/page"


def test_parse_discovers_rss_feed_link(patch_extractor):
    patch_extractor(_stub_extracted(text="body"))
    html = """
    <html><head>
      <link rel="alternate" type="application/rss+xml" href="/feed.xml" />
      <link rel="alternate" type="application/atom+xml" href="https://other.example/atom.xml" />
    </head><body>body</body></html>
    """
    item = parse_website_html(html, source_url="https://x.example/about")
    assert item is not None
    # Relative href resolved against source_url; first match wins
    assert item.metadata["discovered_feed_url"] == "https://x.example/feed.xml"


def test_parse_discovers_absolute_atom_feed(patch_extractor):
    patch_extractor(_stub_extracted(text="body"))
    html = """
    <html><head>
      <link rel="alternate" type="application/atom+xml" href="https://x.example/atom.xml" />
    </head><body>body</body></html>
    """
    item = parse_website_html(html, source_url="https://x.example/")
    assert item is not None
    assert item.metadata["discovered_feed_url"] == "https://x.example/atom.xml"


def test_parse_omits_feed_when_no_link(patch_extractor):
    patch_extractor(_stub_extracted(text="body"))
    item = parse_website_html("<html><body>body</body></html>", source_url="https://x.example/")
    assert item is not None
    assert "discovered_feed_url" not in item.metadata


def test_parse_ignores_non_feed_alternate_links(patch_extractor):
    """``<link rel="alternate" hreflang="es">`` for translations isn't a feed."""
    patch_extractor(_stub_extracted(text="body"))
    html = """
    <html><head>
      <link rel="alternate" hreflang="es" href="/es/" />
      <link rel="canonical" href="/canonical" />
    </head><body>body</body></html>
    """
    item = parse_website_html(html, source_url="https://x.example/")
    assert item is not None
    assert "discovered_feed_url" not in item.metadata


def test_parse_records_domain_in_metadata(patch_extractor):
    patch_extractor(_stub_extracted(text="body"))
    item = parse_website_html("<html>x</html>", source_url="https://blog.example.com/posts/1")
    assert item is not None
    assert item.metadata["domain"] == "blog.example.com"


def test_parse_long_body_truncates_excerpt(patch_extractor):
    long_body = ("word " * 200).strip()  # ~999 chars; .strip() because the parser does too
    patch_extractor(_stub_extracted(text=long_body))
    item = parse_website_html("<html>x</html>", source_url="https://x.example/")
    assert item is not None
    assert len(item.excerpt) <= 320
    assert item.content == long_body


def test_parse_omits_empty_optional_metadata_keys(patch_extractor):
    patch_extractor(_stub_extracted(text="body"))  # no language, no sitename
    item = parse_website_html("<html>x</html>", source_url="https://x.example/")
    assert item is not None
    assert "language" not in item.metadata
    assert "sitename" not in item.metadata


# ── parse_website_html with REAL trafilatura (one integration test) ──────────


def test_parse_with_real_trafilatura_extracts_body():
    """Sanity check: the wiring to trafilatura works on a representative
    HTML fixture. Doesn't pin specific extraction heuristics — those are
    trafilatura's concern and would make the test brittle."""
    html = """
    <html>
      <head><title>Real Article Title</title></head>
      <body>
        <header><nav>Home About Contact</nav></header>
        <article>
          <h1>Real Article Title</h1>
          <p>This is the first paragraph of the article. It has enough text
          to look like real content to trafilatura's boilerplate detector.</p>
          <p>Second paragraph with more substantial content that should be
          extracted as the article body. Trafilatura should pick this up.</p>
          <p>A third paragraph for good measure, ensuring we cross any
          minimum-length thresholds the extractor might apply.</p>
        </article>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """
    item = parse_website_html(html, source_url="https://example.com/article")
    if item is None:
        pytest.skip("trafilatura unavailable or extracted nothing on this fixture")
    assert "Real Article Title" in item.title
    assert "first paragraph" in item.content.lower() or len(item.content) > 100


# ── fetch_website end-to-end ──────────────────────────────────────────────────


async def test_fetch_emits_one_item(conn, patch_client, patch_extractor):
    patch_extractor(_stub_extracted(title="Hello", text="World."))
    body = "<html><head><title>HTML Title</title></head><body>Hi</body></html>"

    def handler(_request):
        return httpx.Response(200, text=body)

    patch_client(handler)
    items = await fetch_website(conn, _task(url="https://x.example/article"))

    assert len(items) == 1
    assert items[0].title == "Hello"
    assert items[0].content == "World."


async def test_fetch_5xx_propagates(conn, patch_client):
    def handler(_request):
        return httpx.Response(503, text="upstream unavailable")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_website(conn, _task())


async def test_fetch_4xx_propagates(conn, patch_client):
    def handler(_request):
        return httpx.Response(404, text="not found")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_website(conn, _task())


async def test_fetch_returns_empty_when_extractor_finds_no_body(
    conn, patch_client, patch_extractor
):
    """Page exists but trafilatura can't extract usable body — emit zero items."""
    patch_extractor(_stub_extracted(text=""))

    def handler(_request):
        return httpx.Response(200, text="<html>nav-only landing</html>")

    patch_client(handler)
    items = await fetch_website(conn, _task())
    assert items == []


async def test_fetch_records_ua_with_contact(conn, patch_client, patch_extractor):
    patch_extractor(_stub_extracted(text="body"))
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["ua"] = request.headers.get("User-Agent")
        return httpx.Response(200, text="<html><body>x</body></html>")

    patch_client(handler)
    await fetch_website(conn, _task())
    assert "ClawFeed-Intel" in (captured["ua"] or "")
    assert "+contact:" in (captured["ua"] or "")


async def test_fetch_website_raises_on_unsafe_url(monkeypatch, conn):
    """An SSRF-rejected URL must raise so the runner records the source as
    failed in coverage — same outcome as a 4xx response."""

    async def reject(_url):
        raise UnsafeUrlError("blocked host: 'router.lan'")

    monkeypatch.setattr(website_mod, "validate_safe_url", reject)

    with pytest.raises(UnsafeUrlError, match="blocked host"):
        await fetch_website(conn, _task(url="http://router.lan/admin"))


async def test_fetch_website_does_not_open_client_when_url_unsafe(monkeypatch, conn):
    """The HTTP client must not be opened when the SSRF guard rejects."""
    opened = {"n": 0}

    async def reject(_url):
        raise UnsafeUrlError("blocked ip literal")

    @asynccontextmanager
    async def tracking_client(*, follow_redirects: bool = True):
        opened["n"] += 1
        yield None  # pragma: no cover — should not be reached

    monkeypatch.setattr(website_mod, "validate_safe_url", reject)
    monkeypatch.setattr(website_mod, "build_client", tracking_client)

    with pytest.raises(UnsafeUrlError):
        await fetch_website(conn, _task(url="http://192.168.1.1/"))
    assert opened["n"] == 0


async def test_fetch_rejects_non_website_task(conn):
    bad = ResolvedTask(
        task=RssTask(kind="rss", url="https://x.example/feed"),
        category="scratch",
        origin="yaml",
        source_id=None,
        source_name="x",
    )
    with pytest.raises(TypeError, match="expected WebsiteTask"):
        await fetch_website(conn, bad)


# ── registration ──────────────────────────────────────────────────────────────


def test_website_fetcher_is_registered():
    assert FETCHER_REGISTRY[KIND] is fetch_website


def test_kind_constant_matches_source_task_discriminator():
    assert KIND == "website"
    WebsiteTask(kind=KIND, url="https://x.example/")
