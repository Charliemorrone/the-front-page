"""GDELT DOC 2.0 fetcher.

Daily-brief use case: GDELT is the open-news backbone — broad global media
coverage, recent time-window search, JSON output, well-suited to keyword
queries on companies, products, people, and topics. Each editorial category
in :file:`config/intel-sources.yaml` declares its own GDELT query
(``startup_funding`` and ``ai_coding_tools`` already do); the resolver hands
us one ``GdeltTask`` per query and we issue one HTTP call per task.

Two-layer pattern, consistent with the other fetchers:

- :func:`parse_gdelt_response` is pure — JSON text or dict in,
  ``FetchedItem``s out. Tested with hand-written fixtures.
- :func:`fetch_gdelt` does the HTTP. Single GET to the DOC 2.0
  ``/api/v2/doc/doc`` endpoint with ``mode=ArtList&format=JSON``.

Failure model:
- 4xx / 5xx → :class:`httpx.HTTPStatusError` propagates so the runner records
  ``failed`` and the rest of the run continues (matches every other fetcher).
- Malformed body (non-JSON, control chars in titles, missing ``articles``) →
  empty list, no raise. GDELT degrades to silent emptiness more often than it
  fails outright; we'd rather have zero items than abort the run.
- Per-entry conversion exception → log + skip; siblings continue.

Time window: hardcoded to ``timespan=24h``. The daily-brief profile is 24h
by definition (``profile.daily_window_hours: 24``); when a non-daily run type
needs it, add a ``timespan`` field to :class:`GdeltTask` and thread it
through. No premature config knob.

Deduplication: GDELT does not provide a stable per-article identifier, so
:func:`normalize.canonicalize_url` of the article URL is the dedup key.
Cross-source folding of syndicated copies (same headline, different domain)
is the dedup-clustering layer's job, not this fetcher's.

Content body: GDELT's ArtList mode returns titles + metadata only — no
article body. ``content`` is therefore an empty string and the title carries
the relevance signal forward. Article-body extraction would multiply the
request count by 250× per task and is out of scope; downstream relevance
filtering judges from titles plus domain/sourcecountry context.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

from .. import normalize
from ..sources import GdeltTask, ResolvedTask
from .base import FETCHER_REGISTRY, FetchedItem
from .http import build_client

log = logging.getLogger(__name__)

KIND = "gdelt"

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_TIMESPAN = "24h"
MAX_RECORDS = 250  # GDELT DOC 2.0 ArtList ceiling
EXCERPT_CHARS = 320

# GDELT seendate format: ``YYYYMMDDTHHMMSSZ`` (always Z / UTC).
_SEENDATE_FMT = "%Y%m%dT%H%M%SZ"


async def fetch_gdelt(task: ResolvedTask) -> list[FetchedItem]:
    """Run one GDELT query and return normalized items."""
    if not isinstance(task.task, GdeltTask):
        raise TypeError(f"fetch_gdelt expected GdeltTask, got {type(task.task).__name__}")

    query_url = _build_query_url(task.task.query)
    async with build_client() as client:
        resp = await client.get(query_url)
        resp.raise_for_status()
        body = resp.text

    return parse_gdelt_response(
        body,
        source_name=task.source_name,
        query=task.task.query,
        query_url=query_url,
    )


def _build_query_url(query: str) -> str:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "JSON",
        "timespan": DEFAULT_TIMESPAN,
        "maxrecords": str(MAX_RECORDS),
        "sort": "DateDesc",
    }
    return f"{API_URL}?{urlencode(params)}"


# ── parsing (pure) ────────────────────────────────────────────────────────────


def parse_gdelt_response(
    body: str | dict[str, Any],
    *,
    source_name: str,
    query: str = "",
    query_url: str = "",
) -> list[FetchedItem]:
    """Parse a GDELT DOC 2.0 ``ArtList`` JSON response into ``FetchedItem``s.

    Accepts either the raw response text or the already-parsed dict; the
    text form is the common case in production while tests prefer dicts.
    Returns an empty list on any malformed / unexpected shape — GDELT is
    flaky enough that a partial-day blank is the correct degradation.
    """
    payload = _coerce_json(body)
    if payload is None:
        return []

    articles = payload.get("articles")
    if not isinstance(articles, list):
        return []

    items: list[FetchedItem] = []
    for raw in articles:
        try:
            item = _article_to_item(raw, query=query, query_url=query_url)
        except Exception:
            log.exception("gdelt: failed to convert article from %s", source_name)
            continue
        if item is not None:
            items.append(item)
    return items


def _coerce_json(body: str | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(body, dict):
        return body
    if not isinstance(body, str) or not body.strip():
        return None
    try:
        # GDELT occasionally emits raw control chars inside titles; strict
        # mode rejects them. Falling back to ``strict=False`` keeps the
        # pipeline alive on those days.
        return json.loads(body)
    except json.JSONDecodeError:
        try:
            return json.loads(body, strict=False)
        except json.JSONDecodeError:
            return None


def _article_to_item(
    raw: Any,
    *,
    query: str,
    query_url: str,
) -> FetchedItem | None:
    if not isinstance(raw, dict):
        return None

    url = (raw.get("url") or "").strip()
    if not url:
        return None
    try:
        canonical_url = normalize.canonicalize_url(url)
    except (TypeError, ValueError):
        return None

    title = (raw.get("title") or "").strip()
    if not title:
        # Without a title, GDELT articles aren't useful for the brief;
        # skip silently (some upstream syndication blanks them).
        return None

    published_at = _seendate_to_iso(raw.get("seendate"))

    metadata: dict[str, Any] = {
        "domain": (raw.get("domain") or "").strip(),
        "language": (raw.get("language") or "").strip(),
        "source_country": (raw.get("sourcecountry") or "").strip(),
    }
    social_image = (raw.get("socialimage") or "").strip()
    if social_image:
        metadata["social_image"] = social_image
    if query:
        metadata["query"] = query
    if query_url:
        metadata["query_url"] = query_url

    return FetchedItem(
        source_type=KIND,
        dedup_key=canonical_url,
        title=title,
        url=url,
        canonical_url=canonical_url,
        # GDELT ArtList carries no article body — title is the only signal.
        content="",
        excerpt=title[:EXCERPT_CHARS],
        author="",
        published_at=published_at,
        content_hash=normalize.content_hash(title, ""),
        metadata=metadata,
        raw_payload=dict(raw),
    )


def _seendate_to_iso(value: Any) -> str | None:
    """GDELT seendate is ``YYYYMMDDTHHMMSSZ`` (always Z). Normalize to ISO."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.strptime(value, _SEENDATE_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.isoformat(timespec="seconds")


# Register on import.
FETCHER_REGISTRY[KIND] = fetch_gdelt
