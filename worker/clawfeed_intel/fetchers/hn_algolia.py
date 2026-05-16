"""Hacker News Algolia search fetcher (Phase 7c topical-search source).

Counterpart to the daily Firebase fetcher (:mod:`fetchers.hn`): one
:class:`HnAlgoliaTask` per query variant, mapped to an Algolia HN
Search request, results normalized as :class:`FetchedItem`. Reserved
for topical search per the architecture doc — daily runs use the
Firebase list endpoints only.

**Dedup with the Firebase fetcher** is load-bearing: both emit
``source_type="hn"`` and the same ``hn_dedup_key(<id>)``, so the
same HN item discovered via either path collapses to one row on
``UNIQUE(source_type, dedup_key)``. ``metadata.discovered_via``
tells us which API surfaced the row (``"algolia"`` vs ``"firebase"``).
Same pattern as the daily GitHub fetcher (`kind` differs, `source_type`
collapses).

Window scoping is mandatory here (unlike the Firebase fetcher's
attention-snapshot semantics): topical search is time-bounded by the
operator's ``--window-days`` flag, and Algolia's ``numericFilters``
lets us push the filter to the server rather than fetching thousands
of historical hits to discard most. The orchestrator (Phase 7e) maps
the run's window start to an epoch and the fetcher composes the
``created_at_i>{epoch}`` filter string.

Two-layer pattern:

- :func:`parse_algolia_hit` is pure — one Algolia hit dict →
  :class:`FetchedItem` (or ``None`` for comments / un-titled hits).
- :func:`fetch_hn_algolia` does the HTTP: build params + dispatch +
  assemble.

Single-page fetching for Phase 7c. Algolia's default of 50 hits per
page is a reasonable cap per query variant; the topic planner emits
~3-8 variants so the total cap is ~150-400 hits per topic run.
Pagination is a future optimization if real runs show truncation.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from selectolax.parser import HTMLParser

from .. import normalize
from ..sources import HnAlgoliaTask, ResolvedTask
from .base import FETCHER_REGISTRY, FetchedItem
from .http import build_client

log = logging.getLogger(__name__)

KIND = "hn_algolia"
SOURCE_TYPE = "hn"  # Shared with the Firebase fetcher for cross-API dedup.

API_URL = "https://hn.algolia.com/api/v1/search"
DISCUSSION_URL_TMPL = "https://news.ycombinator.com/item?id={item_id}"
EXCERPT_CHARS = 320


async def fetch_hn_algolia(conn: sqlite3.Connection, task: ResolvedTask) -> list[FetchedItem]:
    """Fetch one Algolia search for the given query, return normalized items."""
    del conn
    if not isinstance(task.task, HnAlgoliaTask):
        raise TypeError(f"fetch_hn_algolia expected HnAlgoliaTask, got {type(task.task).__name__}")

    params = _build_params(task.task)
    async with build_client() as client:
        resp = await client.get(API_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

    return parse_algolia_response(payload, query=task.task.query)


def _build_params(task: HnAlgoliaTask) -> dict[str, Any]:
    """Construct Algolia query parameters. Pure for fixture testing.

    ``numericFilters`` is only set when the orchestrator supplies a
    window — omitting it lets a future "all-time topic search" flow
    work without code changes.
    """
    params: dict[str, Any] = {
        "query": task.query,
        "tags": task.tags,
        "hitsPerPage": task.hits_per_page,
    }
    if task.window_start_epoch is not None:
        params["numericFilters"] = f"created_at_i>{task.window_start_epoch}"
    return params


# ── Response parsing (pure) ──────────────────────────────────────────────────


def parse_algolia_response(payload: Any, *, query: str) -> list[FetchedItem]:
    """Normalize an Algolia search response into :class:`FetchedItem`s.

    Defensive against shape drift: a non-dict payload, a missing
    ``hits`` key, or a non-list ``hits`` value all yield an empty
    list rather than raising. Per-hit conversion failures are logged
    + skipped so one bad row doesn't poison the batch.
    """
    if not isinstance(payload, dict):
        log.warning("hn_algolia: non-dict payload type=%s", type(payload).__name__)
        return []
    hits = payload.get("hits")
    if not isinstance(hits, list):
        return []

    out: list[FetchedItem] = []
    for hit in hits:
        try:
            item = parse_algolia_hit(hit, query=query)
        except Exception:
            log.exception("hn_algolia: failed to convert hit %r", _hit_object_id(hit))
            continue
        if item is not None:
            out.append(item)
    return out


def parse_algolia_hit(hit: Any, *, query: str) -> FetchedItem | None:
    """Convert one Algolia hit into a :class:`FetchedItem`.

    Returns ``None`` for hits we shouldn't surface (non-dict, no
    ``objectID``, no usable title, non-story). The ``_tags`` array is
    the canonical hit-type discriminator; we filter to stories only
    here because the topic planner asks Algolia for ``tags="story"``,
    but the defensive filter catches API drift.
    """
    if not isinstance(hit, dict):
        return None

    object_id = hit.get("objectID")
    if not isinstance(object_id, str) or not object_id.isdigit():
        return None
    item_id = int(object_id)

    tags = hit.get("_tags")
    if isinstance(tags, list) and not any(t == "story" for t in tags):
        # Algolia returned a comment / poll / job despite the story
        # tag filter — defensive skip.
        return None

    title = (hit.get("title") or hit.get("story_title") or "").strip()
    if not title:
        return None

    external_url = (hit.get("url") or hit.get("story_url") or "").strip()
    discussion_url = DISCUSSION_URL_TMPL.format(item_id=item_id)
    primary_url = external_url or discussion_url
    try:
        canonical = normalize.canonicalize_url(primary_url)
    except (TypeError, ValueError):
        canonical = primary_url

    story_html = hit.get("story_text") or ""
    content = _strip_html(story_html) if story_html else ""

    published_at = _epoch_to_iso(hit.get("created_at_i"))
    points = int(hit.get("points") or 0)
    num_comments = int(hit.get("num_comments") or 0)

    metadata: dict[str, Any] = {
        "hn_id": item_id,
        "discovered_via": "algolia",
        "query": query,
        "score": points,
        "descendants": num_comments,
        "discussion_url": discussion_url,
    }
    if external_url:
        metadata["external_url"] = external_url
    if isinstance(tags, list):
        metadata["tags"] = list(tags)

    return FetchedItem(
        source_type=SOURCE_TYPE,
        dedup_key=normalize.hn_dedup_key(item_id),
        title=title,
        url=primary_url,
        canonical_url=canonical,
        content=content,
        excerpt=content[:EXCERPT_CHARS],
        author=(hit.get("author") or "").strip(),
        published_at=published_at,
        content_hash=normalize.content_hash(title, content),
        metadata=metadata,
        raw_payload=_compact_hit(hit),
    )


def _hit_object_id(hit: Any) -> Any:
    """Best-effort extraction for the error-log line; must not raise.

    The caller invokes this from a ``log.exception`` clause inside an
    exception handler — if this raises, the outer ``except Exception``
    is already exited and the failure escapes the per-hit error
    isolation. Catching every exception here is the right tradeoff:
    losing the object-id in one log line is far less costly than
    aborting the whole response parse.
    """
    if not isinstance(hit, dict):
        return None
    try:
        return hit.get("objectID")
    except Exception:
        return None


def _compact_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Drop the per-field-match-highlight metadata Algolia adds.

    ``_highlightResult`` and ``_snippetResult`` are roughly 3-5x the
    size of the actual hit data and carry no signal a downstream
    stage uses. Stripping them keeps ``raw_payload`` lean.
    """
    return {k: v for k, v in hit.items() if k not in {"_highlightResult", "_snippetResult"}}


def _epoch_to_iso(value: Any) -> str | None:
    """Algolia's ``created_at_i`` is unix epoch seconds (UTC)."""
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
FETCHER_REGISTRY[KIND] = fetch_hn_algolia
