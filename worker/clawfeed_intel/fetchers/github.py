"""GitHub fetcher (Trending HTML + REST Search API).

Two task kinds register here:

- ``github_trending`` — scrape ``https://github.com/trending[/<language>]`` for
  the daily trending list, then enrich each discovered repo via the REST
  repos API for current stars/forks/topics/pushed_at.
- ``github_search`` — call ``api.github.com/search/repositories`` with the
  task's query; the search response already includes the rich metadata so
  no per-repo enrichment is needed.

Both kinds emit ``FetchedItem``s with ``source_type="github"``; their
provenance is recorded in ``metadata.discovered_via`` (``trending`` or
``search``).

Velocity (the headline use case "repos gaining traction") works the same way
for both kinds: each observed repo is appended to ``github_repo_observations``
via :func:`db.record_repo_observation`, and :func:`db.get_repo_velocity`
reads back the time-ordered star delta over the trailing window. That
delta is attached to each item's metadata so downstream relevance and
summary stages can reason about momentum, not just appearance. Day-1
returns no velocity (single observation) — accepted in the architecture
doc's open risks.

Auth: ``GITHUB_TOKEN`` env var, if set, is sent as
``Authorization: Bearer <token>`` on api.github.com requests (5000 req/hr
authenticated vs 60 req/hr unauthenticated). The trending HTML page is
served by github.com (not api.github.com) and doesn't require auth.

Failure model:
- Trending HTML / search REST 4xx/5xx → :class:`httpx.HTTPStatusError`
  propagates so the runner records ``failed``.
- Per-repo enrichment failure (one repo in the trending list 5xx's, or
  the API rejects a request) → log + skip that repo, continue with the
  rest. Matches SEC's partial-failure approach.
- Malformed HTML / JSON → empty list, no raise.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
from typing import Any

import httpx
from selectolax.parser import HTMLParser

from .. import db, normalize
from ..sources import GithubSearchTask, GithubTrendingTask, ResolvedTask
from .base import FETCHER_REGISTRY, FetchedItem
from .http import build_client

log = logging.getLogger(__name__)

KIND_TRENDING = "github_trending"
KIND_SEARCH = "github_search"
SOURCE_TYPE = "github"

API_HOST = "https://api.github.com"
TRENDING_HOST = "https://github.com"
TRENDING_PATH = "/trending"

GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
ENRICH_CONCURRENCY = 8
SEARCH_PER_PAGE = 30  # Reasonable Phase-1 ceiling; max is 100
TITLE_CAP_CHARS = 200
EXCERPT_CHARS = 320

# Velocity is computed over a rolling 7-day window by default. Architecture
# doc envisions short- and longer-window comparisons; we ship the short
# window first since it's the headline "gaining traction" signal.
VELOCITY_WINDOW_DAYS = 7

# "★ 12,345" / "★ 12.3k" — the trending page's total-stars badge.
_STAR_TOTAL_RE = re.compile(r"([\d,\.]+)\s*k?", re.IGNORECASE)
# "234 stars today" / "1,234 stars this week" — the trending page's
# discovery-velocity badge. We capture this for the brief but the
# stored-observation delta is the authoritative signal.
_STARS_PERIOD_RE = re.compile(
    r"([\d,]+)\s+stars?\s+(today|this\s+week|this\s+month)", re.IGNORECASE
)

# GitHub uses several top-level paths for non-repo content. A trending-card
# parser shouldn't mistake those for repos — they have the same ``/x/y`` shape
# but ``x`` is a reserved bucket, not an owner.
_RESERVED_TOP_LEVEL_PATHS: frozenset[str] = frozenset(
    {
        "topics",
        "marketplace",
        "sponsors",
        "settings",
        "codespaces",
        "organizations",
        "orgs",
        "users",
        "search",
        "explore",
        "trending",
        "collections",
        "events",
        "notifications",
        "issues",
        "pulls",
        "stars",
        "watching",
        "new",
    }
)


# ── github_trending entry point ───────────────────────────────────────────────


async def fetch_github_trending(conn: sqlite3.Connection, task: ResolvedTask) -> list[FetchedItem]:
    """Discover trending repos via HTML, enrich via REST, record observations."""
    if not isinstance(task.task, GithubTrendingTask):
        raise TypeError(
            f"fetch_github_trending expected GithubTrendingTask, got {type(task.task).__name__}"
        )
    language = task.task.language

    async with build_client() as client:
        html = await _fetch_trending_html(client, language)
        discovered = parse_trending_html(html, language=language)
        if not discovered:
            return []
        enriched = await _enrich_trending_repos(client, discovered)

    items: list[FetchedItem] = []
    for repo, td in enriched:
        if repo is None:
            continue
        # Stars-today is per-day discovery momentum surfaced by GitHub on the
        # trending page. We carry it forward as a corroborating signal but
        # the stored-observation delta is the authoritative velocity.
        items.append(
            _record_and_build_item(
                conn,
                repo,
                discovered_via="trending",
                extra_metadata=(
                    {"stars_today": td.stars_today} if td.stars_today is not None else {}
                ),
            )
        )
    return items


# ── github_search entry point ─────────────────────────────────────────────────


async def fetch_github_search(conn: sqlite3.Connection, task: ResolvedTask) -> list[FetchedItem]:
    """Run one GitHub repository search and return normalized items."""
    if not isinstance(task.task, GithubSearchTask):
        raise TypeError(
            f"fetch_github_search expected GithubSearchTask, got {type(task.task).__name__}"
        )
    query = task.task.query
    query_url = _build_search_url(query)

    async with build_client() as client:
        resp = await client.get(query_url, headers=_auth_headers())
        resp.raise_for_status()
        body = resp.text

    repos = parse_search_response(body)
    items: list[FetchedItem] = []
    for repo in repos:
        items.append(
            _record_and_build_item(
                conn,
                repo,
                discovered_via="search",
                extra_metadata={"query": query, "query_url": query_url},
            )
        )
    return items


# ── trending HTML pipeline ────────────────────────────────────────────────────


class _TrendingDiscovered:
    """One repo as parsed from the trending HTML — pre-enrichment."""

    __slots__ = ("full_name", "description", "language", "stars_total", "stars_today")

    def __init__(
        self,
        *,
        full_name: str,
        description: str,
        language: str | None,
        stars_total: int | None,
        stars_today: int | None,
    ) -> None:
        self.full_name = full_name
        self.description = description
        self.language = language
        self.stars_total = stars_total
        self.stars_today = stars_today


async def _fetch_trending_html(client: httpx.AsyncClient, language: str | None) -> str:
    url = _build_trending_url(language)
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.text


def _build_trending_url(language: str | None) -> str:
    if language:
        # Trending paths use lowercase / hyphenated language slugs.
        slug = language.strip().lower().replace(" ", "-")
        return f"{TRENDING_HOST}{TRENDING_PATH}/{slug}"
    return f"{TRENDING_HOST}{TRENDING_PATH}"


def parse_trending_html(html: str, *, language: str | None = None) -> list[_TrendingDiscovered]:
    """Extract trending repos from a GitHub trending HTML page.

    Returns ``[]`` on empty / malformed input. The selectors are conservative:
    we look for ``article.Box-row`` (the per-repo card class), which has been
    stable for several years and is reasonably central to the page layout.
    """
    del language  # accepted for API symmetry; not needed here
    if not isinstance(html, str) or not html.strip():
        return []
    try:
        tree = HTMLParser(html)
    except Exception:
        log.exception("github: trending HTML parse failed")
        return []

    out: list[_TrendingDiscovered] = []
    for article in tree.css("article.Box-row"):
        try:
            d = _trending_card_to_discovered(article)
        except Exception:
            log.exception("github: failed to parse trending card")
            continue
        if d is not None:
            out.append(d)
    return out


def _trending_card_to_discovered(article: Any) -> _TrendingDiscovered | None:
    # The repo link sits in <h2 class="h3 lh-condensed"><a href="/owner/repo">…
    link = article.css_first("h2 a")
    if link is None:
        return None
    href = (link.attributes.get("href") or "").strip()
    if not href.startswith("/"):
        return None
    full_name = href.lstrip("/").rstrip("/")
    if "/" not in full_name:
        return None
    owner_segment = full_name.split("/", 1)[0]
    if owner_segment.lower() in _RESERVED_TOP_LEVEL_PATHS:
        return None

    desc_node = article.css_first("p.col-9, p.col-md-9, p.color-fg-muted")
    description = (desc_node.text(strip=True) if desc_node is not None else "").strip()

    lang_node = article.css_first('[itemprop="programmingLanguage"]')
    language = lang_node.text(strip=True) if lang_node is not None else None

    stars_total = _parse_int_with_k(_first_text(article.css('a[href$="/stargazers"]')))
    stars_today = _parse_stars_period(_first_text(article.css(".float-sm-right")))

    return _TrendingDiscovered(
        full_name=full_name,
        description=description,
        language=language or None,
        stars_total=stars_total,
        stars_today=stars_today,
    )


def _first_text(nodes: Any) -> str:
    for node in nodes:
        text = node.text(strip=True)
        if text:
            return text
    return ""


def _parse_int_with_k(text: str) -> int | None:
    """Trending shows star counts like ``12,345`` or ``12.3k``."""
    if not text:
        return None
    m = _STAR_TOTAL_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    has_k = "k" in text.lower()
    try:
        value = float(raw)
    except ValueError:
        return None
    if has_k:
        value *= 1000
    return int(value)


def _parse_stars_period(text: str) -> int | None:
    if not text:
        return None
    m = _STARS_PERIOD_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


async def _enrich_trending_repos(
    client: httpx.AsyncClient, discovered: list[_TrendingDiscovered]
) -> list[tuple[dict[str, Any] | None, _TrendingDiscovered]]:
    """Hit the REST repos API for each discovered repo, under a small semaphore.

    Per-repo failures swallow into ``None``; the rest of the batch still ships.
    """
    sem = asyncio.Semaphore(ENRICH_CONCURRENCY)

    async def _one(td: _TrendingDiscovered) -> tuple[dict[str, Any] | None, _TrendingDiscovered]:
        async with sem:
            try:
                resp = await client.get(f"{API_HOST}/repos/{td.full_name}", headers=_auth_headers())
                resp.raise_for_status()
                payload = resp.json()
            except Exception:
                log.warning("github: enrichment failed for %s", td.full_name, exc_info=True)
                return None, td
            if not isinstance(payload, dict):
                return None, td
            return payload, td

    return await asyncio.gather(*(_one(td) for td in discovered))


# ── search REST pipeline ──────────────────────────────────────────────────────


def _build_search_url(query: str) -> str:
    from urllib.parse import urlencode

    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": str(SEARCH_PER_PAGE),
    }
    return f"{API_HOST}/search/repositories?{urlencode(params)}"


def parse_search_response(body: str | dict[str, Any]) -> list[dict[str, Any]]:
    """Extract repo dicts from a Search-Repositories response.

    Returns ``[]`` for any malformed shape. Search responses already carry
    the rich metadata (stars, forks, topics, etc.); no per-repo enrichment.
    """
    import json

    if isinstance(body, dict):
        payload: Any = body
    elif isinstance(body, str) and body.strip():
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return []
    else:
        return []

    if not isinstance(payload, dict):
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


# ── shared item-building + observation recording ─────────────────────────────


def _record_and_build_item(
    conn: sqlite3.Connection,
    repo: dict[str, Any],
    *,
    discovered_via: str,
    extra_metadata: dict[str, Any] | None = None,
) -> FetchedItem:
    """Record one observation, read the resulting velocity, return the FetchedItem.

    Order matters: we record the *current* observation first so that
    :func:`db.get_repo_velocity` sees today's data point as the latest.
    For a brand-new repo this still returns ``None`` (need ≥2 observations);
    for a repo seen on a prior run, today's stars become the latest endpoint.
    """
    full_name = (repo.get("full_name") or "").strip()
    stars = int(repo.get("stargazers_count") or repo.get("watchers_count") or 0)
    forks = _maybe_int(repo.get("forks_count"))
    watchers = _maybe_int(repo.get("watchers_count"))
    open_issues = _maybe_int(repo.get("open_issues_count"))
    language = repo.get("language") or None
    topics = repo.get("topics") if isinstance(repo.get("topics"), list) else []
    pushed_at = repo.get("pushed_at") or None

    velocity: db.RepoVelocity | None = None
    if full_name:
        try:
            db.record_repo_observation(
                conn,
                full_name=full_name,
                stars=stars,
                forks=forks,
                watchers=watchers,
                open_issues=open_issues,
                language=language,
                topics=topics,
                pushed_at=pushed_at,
                discovered_via=discovered_via,  # type: ignore[arg-type]
            )
            velocity = db.get_repo_velocity(
                conn, full_name=full_name, window_days=VELOCITY_WINDOW_DAYS
            )
        except Exception:
            log.exception("github: failed to record/read observation for %s", full_name)

    return _repo_to_fetched_item(
        repo,
        full_name=full_name,
        stars=stars,
        forks=forks,
        watchers=watchers,
        open_issues=open_issues,
        language=language,
        topics=list(topics),
        pushed_at=pushed_at,
        discovered_via=discovered_via,
        velocity=velocity,
        extra_metadata=extra_metadata or {},
    )


def _repo_to_fetched_item(
    repo: dict[str, Any],
    *,
    full_name: str,
    stars: int,
    forks: int | None,
    watchers: int | None,
    open_issues: int | None,
    language: str | None,
    topics: list[str],
    pushed_at: str | None,
    discovered_via: str,
    velocity: db.RepoVelocity | None,
    extra_metadata: dict[str, Any],
) -> FetchedItem:
    description = (repo.get("description") or "").strip()
    html_url = (repo.get("html_url") or "").strip()
    if not html_url and full_name:
        html_url = f"https://github.com/{full_name}"
    try:
        canonical_url = normalize.canonicalize_url(html_url) if html_url else ""
    except (TypeError, ValueError):
        canonical_url = html_url

    title = full_name
    if description:
        # ``owner/repo — Description``. Em-dash separator is unambiguous.
        title = f"{full_name} — {description}"
    if len(title) > TITLE_CAP_CHARS:
        title = title[: TITLE_CAP_CHARS - 1].rstrip() + "…"

    owner = repo.get("owner") or {}
    author = (owner.get("login") if isinstance(owner, dict) else "") or ""

    metadata: dict[str, Any] = {
        "full_name": full_name,
        "stars": stars,
        "discovered_via": discovered_via,
    }
    if forks is not None:
        metadata["forks"] = forks
    if watchers is not None:
        metadata["watchers"] = watchers
    if open_issues is not None:
        metadata["open_issues"] = open_issues
    if language:
        metadata["language"] = language
    if topics:
        metadata["topics"] = topics
    if pushed_at:
        metadata["pushed_at"] = pushed_at
    created_at = repo.get("created_at")
    if created_at:
        metadata["created_at"] = created_at
    homepage = (repo.get("homepage") or "").strip()
    if homepage:
        metadata["homepage"] = homepage
    if velocity is not None:
        metadata["velocity"] = {
            "star_delta": velocity.star_delta,
            "fork_delta": velocity.fork_delta,
            "days_observed": round(velocity.days_observed, 4),
            "earliest_stars": velocity.earliest_stars,
            "latest_stars": velocity.latest_stars,
            "earliest_at": velocity.earliest_at,
            "latest_at": velocity.latest_at,
            "observation_count": velocity.observation_count,
        }
    metadata.update(extra_metadata)

    return FetchedItem(
        source_type=SOURCE_TYPE,
        dedup_key=normalize.github_dedup_key(full_name),
        title=title,
        url=html_url,
        canonical_url=canonical_url,
        content=description,
        excerpt=description[:EXCERPT_CHARS] if description else "",
        author=author,
        published_at=pushed_at,
        content_hash=normalize.content_hash(full_name, description),
        metadata=metadata,
        raw_payload=_compact_repo(repo),
    )


def _compact_repo(repo: dict[str, Any]) -> dict[str, Any]:
    """Drop large duplicative fields before persisting raw_payload.

    GitHub repo responses include ``permissions``, ``security_and_analysis``,
    and a long list of *_url template fields that bloat the payload without
    informing the brief.
    """
    drop = {
        "permissions",
        "security_and_analysis",
        "temp_clone_token",
        "network_count",
        "subscribers_count",
    }
    out: dict[str, Any] = {}
    for k, v in repo.items():
        if k in drop:
            continue
        if isinstance(k, str) and k.endswith("_url") and k != "html_url":
            continue
        out[k] = v
    return out


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _auth_headers() -> dict[str, str]:
    """Construct GitHub API auth headers from the optional GITHUB_TOKEN env var.

    Without a token: 60 req/hr unauthenticated. With a token: 5000 req/hr.
    Trending HTML doesn't require auth, but sending the header doesn't hurt
    (github.com ignores it). We also send the recommended API-version header
    so GitHub's API can route to the stable schema.
    """
    headers = {"X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get(GITHUB_TOKEN_ENV, "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# Register both kinds on import.
FETCHER_REGISTRY[KIND_TRENDING] = fetch_github_trending
FETCHER_REGISTRY[KIND_SEARCH] = fetch_github_search
