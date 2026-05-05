"""Configured-URL website fetcher.

Daily-brief use case: a small set of curated URLs whose content can't be
captured via RSS (e.g. a vendor product page, a research-group team page, a
specific announcement post). The architecture doc is explicit: configured
URLs only — *no* site-wide crawl in v1.

Behavior per the architecture doc fallback:

1. Fetch the configured page.
2. Extract title + main text via trafilatura's ``bare_extraction``.
3. Discover ``<link rel="alternate" type="application/rss+xml">`` if present
   and surface it in metadata. Phase 1 does NOT auto-switch the source kind
   to RSS (that's a dashboard/Node coordination concern); the discovery
   signal lets the brief surface "this website has a feed; consider
   reconfiguring as RSS" without the worker silently mutating config.

Two-layer pattern, consistent with the other fetchers:

- :func:`parse_website_html` is pure — HTML in, ``FetchedItem`` (or
  ``None``) out. Trafilatura runs synchronously inside it; tests pass
  small representative HTML fixtures.
- :func:`fetch_website` does the HTTP + offloads the parse to a thread
  via :func:`asyncio.to_thread`. Trafilatura is CPU-bound and blocks the
  event loop otherwise.

Failure model (matches every other fetcher):
- HTTP 4xx/5xx → :class:`httpx.HTTPStatusError` propagates so the runner
  records ``failed``.
- Trafilatura returns no extractable content → empty list (one configured
  URL produces zero items rather than aborting the run).
- Per-extraction exceptions → log + return empty list.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import Any
from urllib.parse import urljoin

from selectolax.parser import HTMLParser

from .. import normalize
from ..sources import ResolvedTask, WebsiteTask
from .base import FETCHER_REGISTRY, FetchedItem
from .http import build_client

log = logging.getLogger(__name__)

KIND = "website"
SOURCE_TYPE = "website"
EXCERPT_CHARS = 320

# Page-body cap before trafilatura — pathological pages (e.g. site indexes
# embedded as inline HTML) shouldn't blow memory or extraction time.
MAX_PAGE_BYTES = 2 * 1024 * 1024


async def fetch_website(conn: sqlite3.Connection, task: ResolvedTask) -> list[FetchedItem]:
    """Fetch one configured URL and emit a single ``FetchedItem`` per page.

    ``conn`` is part of the unified fetcher contract; the website fetcher
    does not touch the database.
    """
    del conn
    if not isinstance(task.task, WebsiteTask):
        raise TypeError(f"fetch_website expected WebsiteTask, got {type(task.task).__name__}")

    url = task.task.url
    async with build_client() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        body = resp.content[:MAX_PAGE_BYTES]
        final_url = str(resp.url)

    html = body.decode("utf-8", errors="replace")
    # Run trafilatura off the event loop — its DOM walk + boilerplate
    # detection takes 50–200ms on real pages and would otherwise block the
    # gather'd sibling tasks. The parse is pure given inputs.
    item = await asyncio.to_thread(parse_website_html, html, source_url=final_url)
    return [item] if item is not None else []


# ── parsing (pure) ────────────────────────────────────────────────────────────


def parse_website_html(html: str, *, source_url: str) -> FetchedItem | None:
    """Convert one HTML page into a ``FetchedItem`` (or ``None`` on no body).

    Pure: same input always produces the same output. Trafilatura's
    ``bare_extraction`` is synchronous; the async wrapper offloads it via
    :func:`asyncio.to_thread`.
    """
    if not isinstance(html, str) or not html.strip():
        return None
    if not source_url:
        return None

    extracted = _extract_with_trafilatura(html)
    if extracted is None:
        return None

    body = (extracted.get("text") or "").strip()
    if not body:
        # No extractable main content — e.g. a navigation-only landing
        # page. Skip rather than emit a useless item.
        return None

    title = (extracted.get("title") or "").strip() or _fallback_title(html, source_url)
    if not title:
        return None

    try:
        canonical_url = normalize.canonicalize_url(source_url)
    except (TypeError, ValueError):
        canonical_url = source_url

    author = (extracted.get("author") or "").strip()
    published_at = _normalize_date(extracted.get("date"))
    feed_url = _discover_feed_link(html, source_url=source_url)

    metadata: dict[str, Any] = {
        "domain": _domain_of(source_url),
    }
    language = (extracted.get("language") or "").strip()
    if language:
        metadata["language"] = language
    sitename = (extracted.get("sitename") or "").strip()
    if sitename:
        metadata["sitename"] = sitename
    if feed_url:
        metadata["discovered_feed_url"] = feed_url

    return FetchedItem(
        source_type=SOURCE_TYPE,
        dedup_key=canonical_url,
        title=title,
        url=source_url,
        canonical_url=canonical_url,
        content=body,
        excerpt=body[:EXCERPT_CHARS],
        author=author,
        published_at=published_at,
        content_hash=normalize.content_hash(title, body),
        metadata=metadata,
        raw_payload={
            "title": title,
            "author": author,
            "date": extracted.get("date") or "",
            "language": language,
            "sitename": sitename,
        },
    )


def _extract_with_trafilatura(html: str) -> dict[str, Any] | None:
    """Best-effort wrapper around ``trafilatura.bare_extraction``.

    Returns ``None`` if trafilatura is unavailable or extraction fails. The
    function is at module scope (not nested) so tests can monkeypatch it
    cleanly when they want to bypass the real extractor.
    """
    try:
        import trafilatura
    except ImportError:
        return None

    try:
        result = trafilatura.bare_extraction(html, with_metadata=True)
    except Exception:
        log.debug("website: trafilatura extraction failed", exc_info=True)
        return None

    if result is None:
        return None
    # trafilatura returns a Document-like object in newer versions; fall
    # back to dict access for older releases.
    if isinstance(result, dict):
        return result
    return {
        "title": getattr(result, "title", "") or "",
        "author": getattr(result, "author", "") or "",
        "date": getattr(result, "date", "") or "",
        "text": getattr(result, "text", "") or "",
        "language": getattr(result, "language", "") or "",
        "sitename": getattr(result, "sitename", "") or "",
    }


def _fallback_title(html: str, source_url: str) -> str:
    """Extract ``<title>`` text when trafilatura's metadata title is empty.

    Last-resort fallback uses the URL path so we never emit an item with
    an empty title (the dedup_key is a URL, but the title is what shows
    up in the brief).
    """
    try:
        tree = HTMLParser(html)
    except Exception:
        return source_url
    title_node = tree.css_first("title")
    if title_node is not None:
        text = title_node.text(strip=True)
        if text:
            return text
    return source_url


def _discover_feed_link(html: str, *, source_url: str) -> str | None:
    """Find the first ``<link rel="alternate" type="application/(rss|atom)+xml">``.

    Returns the absolute URL of the discovered feed, or ``None`` if no feed
    link is declared on the page. Sites occasionally declare a feed via
    ``<link rel="feed">`` but the architecture doc's example is the
    rel=alternate form; we accept either.
    """
    try:
        tree = HTMLParser(html)
    except Exception:
        return None
    for link in tree.css("link"):
        rel = (link.attributes.get("rel") or "").lower().strip()
        if rel not in {"alternate", "feed"}:
            continue
        type_attr = (link.attributes.get("type") or "").lower().strip()
        if type_attr not in {
            "application/rss+xml",
            "application/atom+xml",
            "application/xml",
        }:
            continue
        href = (link.attributes.get("href") or "").strip()
        if not href:
            continue
        return urljoin(source_url, href)
    return None


def _normalize_date(value: Any) -> str | None:
    """Trafilatura returns dates as ``YYYY-MM-DD`` (date) or ISO timestamps.

    We pass the string through untouched if it looks date-shaped; the
    downstream pipeline treats published_at as a timestamp string and
    sorts it lexicographically, so date-only forms still work.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned


def _domain_of(url: str) -> str:
    from urllib.parse import urlsplit

    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


# Register on import.
FETCHER_REGISTRY[KIND] = fetch_website
