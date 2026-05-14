"""RSS / Atom fetcher.

Architecture doc says: parse with ``feedparser`` first, then fall back to
``trafilatura`` on the article page when the feed entry's summary is too
thin to be useful for relevance/summary stages. That's exactly what this
module does.

Two layers, deliberately:

- :func:`parse_feed_text` is pure — XML in, ``FetchedItem`` out. Easy to
  unit-test against fixture XML strings.
- :func:`fetch_rss` wraps the HTTP fetch around it and adds the trafilatura
  fallback for thin entries. Tests use ``httpx.MockTransport``; production
  uses :func:`http.build_client`.

Trafilatura fallback is best-effort: a failed article fetch falls back to
the entry's summary, never aborts the rest of the feed.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx
from selectolax.parser import HTMLParser

from .. import normalize
from ..sources import ResolvedTask, RssTask
from .base import FETCHER_REGISTRY, FetchedItem
from .http import UnsafeUrlError, build_client, validate_safe_url

log = logging.getLogger(__name__)

KIND = "rss"

# Below this many characters of summary text, we try the article page.
# 400 is a deliberate compromise: short enough to catch teaser feeds, long
# enough that "headline + short blurb" feeds (which extract poorly anyway)
# don't trigger N extra HTTP requests every run.
THIN_SUMMARY_CHARS = 400

EXCERPT_CHARS = 320

# Cap article body before handing to trafilatura — pathological pages
# (some news sites embed entire site indexes as HTML) shouldn't blow memory.
MAX_ARTICLE_BYTES = 2 * 1024 * 1024


async def fetch_rss(conn: sqlite3.Connection, task: ResolvedTask) -> list[FetchedItem]:
    """Fetch one RSS or Atom feed and return normalized items.

    ``conn`` is part of the unified fetcher contract; this fetcher does not
    touch the database (only GitHub does).
    """
    del conn
    if not isinstance(task.task, RssTask):
        raise TypeError(f"fetch_rss expected RssTask, got {type(task.task).__name__}")

    feed_url = task.task.url
    # Defense-in-depth: the feed URL is user-configured (lower-leverage than
    # the trafilatura path below, which follows links extracted from
    # third-party feed bodies), but DNS rebinding and pasted-from-the-web
    # URLs can still point at LAN hosts. UnsafeUrlError propagates so the
    # runner records this source as failed in coverage — same outcome as a
    # 4xx.
    await validate_safe_url(feed_url)
    async with build_client() as client:
        feed_text = await _fetch_feed(client, feed_url)
        items = await asyncio.to_thread(
            parse_feed_text,
            feed_text,
            source_name=task.source_name,
            feed_url=feed_url,
        )
        items = await _enrich_thin_items(client, items)
    return items


async def _fetch_feed(client: httpx.AsyncClient, url: str) -> str:
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.text


# ── parsing (pure) ────────────────────────────────────────────────────────────


def parse_feed_text(
    text: str,
    *,
    source_name: str,
    feed_url: str = "",
) -> list[FetchedItem]:
    """Parse feed XML/Atom into :class:`FetchedItem`s.

    Side-effect-free aside from logging on per-entry conversion failures.
    feedparser is lenient with malformed inputs; we mirror that — garbage
    input returns an empty list rather than raising.
    """
    parsed = feedparser.parse(text)
    feed_meta = getattr(parsed, "feed", None)
    feed_title = (getattr(feed_meta, "title", "") or "").strip() if feed_meta else ""

    items: list[FetchedItem] = []
    for entry in getattr(parsed, "entries", []) or []:
        try:
            item = _entry_to_item(
                entry,
                source_name=source_name,
                feed_url=feed_url,
                feed_title=feed_title,
            )
        except Exception:
            log.exception("rss: failed to convert entry from %s", source_name)
            continue
        if item is not None:
            items.append(item)
    return items


def _entry_to_item(
    entry: Any,
    *,
    source_name: str,
    feed_url: str,
    feed_title: str,
) -> FetchedItem | None:
    link = (getattr(entry, "link", "") or "").strip()
    if not link:
        return None
    try:
        canonical_url = normalize.canonicalize_url(link)
    except (TypeError, ValueError):
        return None

    title = (getattr(entry, "title", "") or "").strip()
    author = _entry_author(entry)
    summary_html = _entry_summary_html(entry)
    content = _strip_html(summary_html)
    excerpt = content[:EXCERPT_CHARS]
    published_at = _published_iso(entry)
    content_hash_value = normalize.content_hash(title, content)

    metadata: dict[str, Any] = {
        "feed_url": feed_url,
        "feed_title": feed_title,
    }
    tags = _entry_tags(entry)
    if tags:
        metadata["tags"] = tags

    return FetchedItem(
        source_type=KIND,
        dedup_key=canonical_url,
        title=title,
        url=link,
        canonical_url=canonical_url,
        content=content,
        excerpt=excerpt,
        author=author,
        published_at=published_at,
        content_hash=content_hash_value,
        metadata=metadata,
        raw_payload={"summary_html": summary_html},
    )


def _entry_summary_html(entry: Any) -> str:
    """Pick the richest HTML body available from a feedparser entry.

    Atom puts full content in ``entry.content`` (a list of dicts); RSS 2.0
    typically uses ``entry.summary``. ``content`` wins when both exist
    because Atom ``content`` is usually fuller.
    """
    contents = getattr(entry, "content", None)
    if isinstance(contents, list):
        for c in contents:
            value = c.get("value") if isinstance(c, dict) else getattr(c, "value", None)
            if value:
                return value
    return getattr(entry, "summary", "") or ""


def _entry_author(entry: Any) -> str:
    author = getattr(entry, "author", "")
    if author:
        return author.strip()
    # feedparser sometimes exposes Atom <author><name>...</name></author> only
    # via author_detail.
    detail = getattr(entry, "author_detail", None)
    if isinstance(detail, dict):
        name = detail.get("name") or ""
    else:
        name = getattr(detail, "name", "") if detail else ""
    return (name or "").strip()


def _entry_tags(entry: Any) -> list[str]:
    raw = getattr(entry, "tags", None) or []
    out: list[str] = []
    for tag in raw:
        term = tag.get("term") if isinstance(tag, dict) else getattr(tag, "term", None)
        if term:
            out.append(str(term))
    return out


def _strip_html(html: str) -> str:
    if not html:
        return ""
    return HTMLParser(html).text(separator=" ", strip=True)


def _published_iso(entry: Any) -> str | None:
    """Convert feedparser's struct_time to UTC ISO 8601.

    Prefers ``published_parsed`` over ``updated_parsed`` so re-emitted feed
    entries (very common — outlets bump ``updated`` for typo fixes) don't
    appear to be brand-new every run.
    """
    for attr in ("published_parsed", "updated_parsed"):
        st = getattr(entry, attr, None)
        if not st:
            continue
        try:
            dt = datetime(*st[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        return dt.isoformat(timespec="seconds")
    return None


# ── trafilatura fallback ──────────────────────────────────────────────────────


async def _enrich_thin_items(
    client: httpx.AsyncClient, items: list[FetchedItem]
) -> list[FetchedItem]:
    """For each item with a thin summary, try to upgrade ``content`` with
    trafilatura-extracted full text. Per-item failures keep the original."""
    enriched: list[FetchedItem] = []
    for item in items:
        if len(item.content) >= THIN_SUMMARY_CHARS or not item.url:
            enriched.append(item)
            continue
        full_text = await _trafilatura_extract(client, item.url)
        if not full_text or len(full_text) <= len(item.content):
            enriched.append(item)
            continue
        enriched.append(_replace_content(item, full_text))
    return enriched


def _replace_content(item: FetchedItem, full_text: str) -> FetchedItem:
    return FetchedItem(
        source_type=item.source_type,
        dedup_key=item.dedup_key,
        title=item.title,
        url=item.url,
        canonical_url=item.canonical_url,
        content=full_text,
        excerpt=full_text[:EXCERPT_CHARS],
        author=item.author,
        published_at=item.published_at,
        content_hash=normalize.content_hash(item.title, full_text),
        metadata={**item.metadata, "trafilatura": True},
        raw_payload=item.raw_payload,
    )


async def _trafilatura_extract(client: httpx.AsyncClient, url: str) -> str | None:
    """Best-effort full-text extraction. Returns ``None`` on any failure.

    Lives behind a function boundary so tests can patch it without pulling
    in trafilatura itself or running real HTTP.

    The article URL comes from inside a third-party feed body, so the SSRF
    guard runs *here*, not just at the top of ``fetch_rss``: a poisoned
    feed could include a link to ``http://192.168.1.1/admin`` that the
    feed URL itself wouldn't reveal. UnsafeUrlError is logged and
    swallowed (returns ``None``) to preserve the fallback's existing
    best-effort contract — we keep the entry's original summary instead.
    """
    try:
        await validate_safe_url(url)
    except UnsafeUrlError as exc:
        log.warning("trafilatura: refusing unsafe url %s (%s)", url, exc)
        return None
    try:
        resp = await client.get(url)
        if resp.status_code >= 400:
            return None
        body = resp.content[:MAX_ARTICLE_BYTES]
    except Exception:
        log.debug("trafilatura: HTTP fetch failed for %s", url, exc_info=True)
        return None

    try:
        import trafilatura
    except ImportError:
        return None

    try:
        html = body.decode("utf-8", errors="replace")
        return await asyncio.to_thread(trafilatura.extract, html)
    except Exception:
        log.debug("trafilatura: extract failed for %s", url, exc_info=True)
        return None


# Register on import.
FETCHER_REGISTRY[KIND] = fetch_rss
