"""Hacker News fetcher (Firebase API).

Daily-brief use case: pull the current ``top`` / ``best`` / ``new`` /
``ask`` / ``show`` lists, fan out item-detail fetches, return normalized
items. Algolia HN Search is reserved for topical search (Phase 7), per the
architecture doc.

Two-layer pattern, consistent with the other fetchers:

- :func:`parse_hn_item` is pure — raw HN dict → ``FetchedItem`` (or
  ``None`` for deleted/dead/comment items).
- :func:`fetch_hn` does the HTTP: list endpoint → truncate → fan out item
  fetches under a small semaphore → assemble.

Concurrency: HN's Firebase CDN has no published rate limit but punishes
sustained fan-out. ``CONCURRENCY = 10`` keeps wall-time on a 200-item batch
at a few seconds without provoking throttling. Per-item HTTP failures are
caught and logged; one bad item does not abort the task.

Window scoping is deliberately *not* applied here: the ``topstories``
endpoint is an attention snapshot, not a time window. A week-old story
that's currently #1 represents today's developer attention. Downstream
stages should reason about freshness using ``metadata.list`` and ``score``
rather than ``published_at``.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

import httpx
from selectolax.parser import HTMLParser

from .. import normalize
from ..sources import HnTask, ResolvedTask
from .base import FETCHER_REGISTRY, FetchedItem
from .http import build_client

log = logging.getLogger(__name__)

KIND = "hn"

API_BASE = "https://hacker-news.firebaseio.com/v0"
DEFAULT_LIMIT = 200
CONCURRENCY = 10
EXCERPT_CHARS = 320

DISCUSSION_URL_TMPL = "https://news.ycombinator.com/item?id={item_id}"

# Maps the HnTask.list discriminator to its Firebase endpoint filename.
_LIST_ENDPOINTS: dict[str, str] = {
    "top": "topstories.json",
    "best": "beststories.json",
    "new": "newstories.json",
    "show": "showstories.json",
    "ask": "askstories.json",
}

# Item types we consider story-like. Comments are filtered defensively even
# though list endpoints don't return them.
_ITEM_TYPES_KEPT: frozenset[str] = frozenset({"story", "job", "poll"})


async def fetch_hn(conn: sqlite3.Connection, task: ResolvedTask) -> list[FetchedItem]:
    """Fetch one HN list and return normalized items.

    ``conn`` is part of the unified fetcher contract; HN does not touch the
    database (only GitHub does).
    """
    del conn
    if not isinstance(task.task, HnTask):
        raise TypeError(f"fetch_hn expected HnTask, got {type(task.task).__name__}")

    list_path = _LIST_ENDPOINTS[task.task.list]
    limit = task.task.limit or DEFAULT_LIMIT
    min_score = task.task.min_score

    async with build_client() as client:
        ids = await _fetch_id_list(client, list_path)
        ids = ids[:limit]
        raw_items = await _fetch_items_concurrently(client, ids)

    return _build_items(raw_items, list_name=task.task.list, min_score=min_score)


async def _fetch_id_list(client: httpx.AsyncClient, list_path: str) -> list[int]:
    """GET the list endpoint; HN returns a JSON array of item IDs (up to ~500).

    Non-200 raises so the runner can record a ``failed`` outcome.
    """
    resp = await client.get(f"{API_BASE}/{list_path}")
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        log.warning("hn: list endpoint %s returned non-list payload", list_path)
        return []
    return [int(i) for i in payload if isinstance(i, int)]


async def _fetch_items_concurrently(
    client: httpx.AsyncClient, ids: list[int]
) -> list[dict[str, Any] | None]:
    """Fetch each item under a concurrency cap, swallowing per-item failures.

    Returns the raw HN dicts (or ``None`` for fetch failures / deleted items)
    in the same order as *ids*; the assembly step filters them out. Returning
    ``None`` rather than raising keeps the failure mode local — one flaky
    item should not poison the task.
    """
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(item_id: int) -> dict[str, Any] | None:
        async with sem:
            try:
                resp = await client.get(f"{API_BASE}/item/{item_id}.json")
                resp.raise_for_status()
                payload = resp.json()
            except Exception:
                log.debug("hn: item %d fetch failed", item_id, exc_info=True)
                return None
            if not isinstance(payload, dict):
                return None
            return payload

    return await asyncio.gather(*(one(i) for i in ids))


def _build_items(
    raw_items: list[dict[str, Any] | None],
    *,
    list_name: str,
    min_score: int | None,
) -> list[FetchedItem]:
    out: list[FetchedItem] = []
    for raw in raw_items:
        if raw is None:
            continue
        try:
            item = parse_hn_item(raw, list_name=list_name)
        except Exception:
            log.exception("hn: failed to convert item %r", raw.get("id"))
            continue
        if item is None:
            continue
        if min_score is not None and (raw.get("score") or 0) < min_score:
            continue
        out.append(item)
    return out


# ── parsing (pure) ────────────────────────────────────────────────────────────


def parse_hn_item(raw: dict[str, Any], *, list_name: str) -> FetchedItem | None:
    """Convert one HN Firebase item dict into a :class:`FetchedItem`.

    Returns ``None`` for items we shouldn't surface (deleted, dead, comments,
    untitled). Side-effect-free.
    """
    if not isinstance(raw, dict):
        return None
    if raw.get("deleted") or raw.get("dead"):
        return None

    item_id = raw.get("id")
    if not isinstance(item_id, int):
        return None

    item_type = (raw.get("type") or "story").strip()
    if item_type not in _ITEM_TYPES_KEPT:
        return None

    title = (raw.get("title") or "").strip()
    if not title:
        # An item with no title is unusable for a brief; skip with no log
        # noise — the HN list does occasionally include placeholder rows.
        return None

    text_html = raw.get("text") or ""
    content = _strip_html(text_html) if text_html else ""

    discussion_url = DISCUSSION_URL_TMPL.format(item_id=item_id)
    external_url = (raw.get("url") or "").strip()
    primary_url = external_url or discussion_url
    try:
        canonical = normalize.canonicalize_url(primary_url)
    except (TypeError, ValueError):
        canonical = primary_url

    published_at = _epoch_to_iso(raw.get("time"))
    score = int(raw.get("score") or 0)
    descendants = int(raw.get("descendants") or 0)

    metadata: dict[str, Any] = {
        "hn_id": item_id,
        "list": list_name,
        "type": item_type,
        "score": score,
        "descendants": descendants,
        "discussion_url": discussion_url,
    }
    if external_url:
        metadata["external_url"] = external_url

    return FetchedItem(
        source_type=KIND,
        dedup_key=normalize.hn_dedup_key(item_id),
        title=title,
        url=primary_url,
        canonical_url=canonical,
        content=content,
        excerpt=content[:EXCERPT_CHARS],
        author=(raw.get("by") or "").strip(),
        published_at=published_at,
        content_hash=normalize.content_hash(title, content),
        metadata=metadata,
        raw_payload=_compact_raw(raw),
    )


def _compact_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop large nested fields (`kids`, `parts`) before persisting raw_payload.

    Comment trees and poll-option lists can balloon — we keep their counts
    via ``descendants`` in metadata, which is what the brief actually uses.
    """
    return {k: v for k, v in raw.items() if k not in {"kids", "parts"}}


def _epoch_to_iso(value: Any) -> str | None:
    """HN ``time`` is unix epoch seconds (UTC). Normalize to project ISO shape."""
    if not isinstance(value, (int, float)):
        return None
    try:
        dt = datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return dt.isoformat(timespec="seconds")


def _strip_html(html: str) -> str:
    if not html:
        return ""
    return HTMLParser(html).text(separator=" ", strip=True)


# Register on import.
FETCHER_REGISTRY[KIND] = fetch_hn
