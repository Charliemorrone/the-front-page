"""Reddit fetcher (public JSON listing endpoints).

Daily-brief use case: surface posts from a small set of curated subreddits
(``r/MachineLearning``, ``r/LocalLLaMA``, ``r/programming``, ``r/startups``,
…). Reddit is supplementary signal — it informs the brief, it doesn't
dominate it. The architecture doc is explicit: "curated subreddits,
conservative limits, local relevance threshold, clear source weighting."

Two-layer pattern, consistent with the other fetchers:

- :func:`parse_reddit_listing` is pure — Reddit ``Listing`` JSON in,
  ``FetchedItem``s out. Tested with hand-built fixtures.
- :func:`fetch_reddit` does the HTTP. One GET per ``RedditTask`` to
  ``https://www.reddit.com/r/<sub>/<sort>.json?limit=<n>``.

Compliance: Reddit's API guidance requires a clear UA with contact info.
That's what :mod:`fetchers.http` provides via ``CLAWFEED_CONTACT_EMAIL``.
The polite-rate is enforced at the harness level (one task = one request);
broader rate-limit budgeting (≤60 req/min unauthenticated) is a non-issue at
Phase-1 source counts but worth flagging when more subreddits are added.

Cross-subreddit dedup: the same external article submitted to two different
subreddits produces two distinct posts with two different fullnames
(``t3_abc`` vs ``t3_xyz``). We keep both — they're independent attention
signals, not duplicates. Cross-source folding of the underlying article
happens later via content_hash clustering, not this fetcher.

Failure model:
- 4xx / 5xx (including 429 rate-limit) → :class:`httpx.HTTPStatusError`
  propagates so the runner records ``failed``. Backoff is a future concern;
  daily cadence rarely trips Reddit's quota.
- Malformed body, wrong ``kind``, missing ``children`` → empty list, no
  raise (matches every other fetcher).
- Removed posts, stickied mod posts, untitled posts, comment-type entries
  → skipped silently.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlencode

from .. import normalize
from ..sources import RedditTask, ResolvedTask
from .base import FETCHER_REGISTRY, FetchedItem
from .http import build_client

log = logging.getLogger(__name__)

KIND = "reddit"

API_HOST = "https://www.reddit.com"
DEFAULT_LIMIT = 100  # Reddit listing endpoint ceiling per request
EXCERPT_CHARS = 320

# Reddit "kind" tag for link/post entries. ``t1`` = comment, ``t3`` = link,
# ``t4`` = message, ``t5`` = subreddit. We only surface posts.
_POST_KIND = "t3"


async def fetch_reddit(task: ResolvedTask) -> list[FetchedItem]:
    """Fetch one subreddit listing and return normalized items."""
    if not isinstance(task.task, RedditTask):
        raise TypeError(f"fetch_reddit expected RedditTask, got {type(task.task).__name__}")

    subreddit = task.task.subreddit
    sort = task.task.sort
    limit = task.task.limit or DEFAULT_LIMIT

    url = _build_listing_url(subreddit, sort, limit)
    async with build_client() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        body = resp.text

    return parse_reddit_listing(
        body,
        source_name=task.source_name,
        subreddit=subreddit,
        sort=sort,
    )


def _build_listing_url(subreddit: str, sort: str, limit: int) -> str:
    # ``quote`` defends against unusual characters in the subreddit name —
    # Reddit only allows [A-Za-z0-9_], but URL-encoding is cheap insurance
    # against config typos.
    sub = quote(subreddit, safe="")
    return f"{API_HOST}/r/{sub}/{sort}.json?{urlencode({'limit': limit})}"


# ── parsing (pure) ────────────────────────────────────────────────────────────


def parse_reddit_listing(
    body: str | dict[str, Any],
    *,
    source_name: str,
    subreddit: str,
    sort: str,
) -> list[FetchedItem]:
    """Parse a Reddit ``Listing`` JSON response into ``FetchedItem``s.

    Accepts either raw response text or an already-parsed dict. Returns an
    empty list on any malformed or unexpected shape — matches the other
    fetchers' lenient stance.
    """
    payload = _coerce_json(body)
    if payload is None:
        return []
    if payload.get("kind") != "Listing":
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    children = data.get("children")
    if not isinstance(children, list):
        return []

    items: list[FetchedItem] = []
    for child in children:
        try:
            item = _child_to_item(child, subreddit=subreddit, sort=sort)
        except Exception:
            log.exception("reddit: failed to convert child from %s", source_name)
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
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _child_to_item(
    child: Any,
    *,
    subreddit: str,
    sort: str,
) -> FetchedItem | None:
    if not isinstance(child, dict):
        return None
    if child.get("kind") != _POST_KIND:
        # Listings shouldn't include comments/messages, but be defensive.
        return None

    data = child.get("data")
    if not isinstance(data, dict):
        return None

    # Removed-by-mod / removed-by-anti-spam / removed-by-deleted posts have a
    # non-null `removed_by_category`. They're worthless to the brief; skip.
    if data.get("removed_by_category"):
        return None
    # Stickied posts are usually subreddit rules / weekly-thread announcements,
    # not news signals. Skip.
    if data.get("stickied"):
        return None

    fullname = (data.get("name") or "").strip()
    post_id = (data.get("id") or "").strip()
    if not fullname:
        # ``name`` is the fullname (``t3_<id>``); without it we can't build a
        # stable dedup key. Fall back to constructing one from id+kind.
        if not post_id:
            return None
        fullname = f"{_POST_KIND}_{post_id}"

    title = (data.get("title") or "").strip()
    if not title:
        return None

    permalink = (data.get("permalink") or "").strip()
    discussion_url = f"{API_HOST}{permalink}" if permalink else ""

    is_self = bool(data.get("is_self"))
    raw_url = (data.get("url") or "").strip()
    if is_self:
        # Self posts: URL is the discussion. Reddit's `url` field for
        # self posts duplicates the permalink, but we normalize on
        # discussion_url for consistency.
        primary_url = discussion_url or raw_url
    else:
        primary_url = raw_url or discussion_url

    if not primary_url:
        return None

    try:
        canonical_url = normalize.canonicalize_url(primary_url)
    except (TypeError, ValueError):
        canonical_url = primary_url

    selftext = data.get("selftext") or ""
    # Reddit HTML-encodes some characters in selftext (``&amp;``, ``&gt;``,
    # ``&#x200B;`` zero-width space). Decode so the text reads cleanly when
    # the LLM consumes it. Markdown formatting (``*emphasis*``, ``> quote``)
    # is left intact — it's information, not noise.
    content = html.unescape(selftext) if isinstance(selftext, str) else ""

    metadata: dict[str, Any] = {
        "subreddit": subreddit,
        "sort": sort,
        "post_id": post_id,
        "fullname": fullname,
        "score": int(data.get("score") or 0),
        "num_comments": int(data.get("num_comments") or 0),
        "discussion_url": discussion_url,
        "is_self": is_self,
        "over_18": bool(data.get("over_18")),
    }
    domain = (data.get("domain") or "").strip()
    if domain:
        metadata["domain"] = domain
    flair = (data.get("link_flair_text") or "").strip() if data.get("link_flair_text") else ""
    if flair:
        metadata["flair"] = flair
    if not is_self and raw_url:
        metadata["external_url"] = raw_url

    return FetchedItem(
        source_type=KIND,
        dedup_key=normalize.reddit_dedup_key(fullname),
        title=title,
        url=primary_url,
        canonical_url=canonical_url,
        content=content,
        excerpt=content[:EXCERPT_CHARS] if content else title[:EXCERPT_CHARS],
        author=(data.get("author") or "").strip(),
        published_at=_epoch_to_iso(data.get("created_utc")),
        content_hash=normalize.content_hash(title, content),
        metadata=metadata,
        raw_payload=_compact_raw(data),
    )


def _compact_raw(data: dict[str, Any]) -> dict[str, Any]:
    """Drop large duplicative fields before persisting raw_payload.

    ``selftext_html`` is Reddit's HTML-rendered version of ``selftext`` and
    can be 5-10× larger; we already keep the plain ``selftext`` in
    ``content``. ``preview`` and ``media_embed`` carry image/video metadata
    that's irrelevant to the daily brief.
    """
    drop = {"selftext_html", "preview", "media_embed", "secure_media_embed"}
    return {k: v for k, v in data.items() if k not in drop}


def _epoch_to_iso(value: Any) -> str | None:
    """Reddit ``created_utc`` is a float epoch (UTC)."""
    if not isinstance(value, (int, float)):
        return None
    try:
        dt = datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return dt.isoformat(timespec="seconds")


# Register on import.
FETCHER_REGISTRY[KIND] = fetch_reddit
