"""Tests for the GitHub fetcher (Trending HTML + REST Search API).

Three test surfaces:

1. ``parse_trending_html`` and ``parse_search_response`` are pure —
   exercised with hand-built fixtures: full-fields happy paths, missing
   optional fields, malformed inputs, syndication parsing edge cases.

2. ``fetch_github_trending`` and ``fetch_github_search`` use
   ``httpx.MockTransport`` and a real :func:`temp_db` to assert: query URL
   construction, the trending → enrichment two-call pattern, per-repo
   enrichment failure isolation, ``GITHUB_TOKEN`` auth header propagation,
   the velocity round-trip (record observation → compute velocity →
   attach to FetchedItem.metadata), and registration.

3. The **load-bearing velocity test** writes an older observation to the
   DB, runs the fetcher, and asserts that the resulting FetchedItem
   carries a non-zero ``metadata.velocity.star_delta`` matching the
   computed delta. This is the architecture doc's hard requirement under
   test ("real velocity from stored observations, not Trending alone").
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from clawfeed_intel import db
from clawfeed_intel.fetchers import FETCHER_REGISTRY
from clawfeed_intel.fetchers.github import (
    API_HOST,
    KIND_SEARCH,
    KIND_TRENDING,
    SOURCE_TYPE,
    TRENDING_HOST,
    fetch_github_search,
    fetch_github_trending,
    parse_search_response,
    parse_trending_html,
)
from clawfeed_intel.sources import (
    GithubSearchTask,
    GithubTrendingTask,
    ResolvedTask,
    RssTask,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _trending_task(
    *, language: str | None = None, source_name: str = "github_traction:trending"
) -> ResolvedTask:
    return ResolvedTask(
        task=GithubTrendingTask(kind="github_trending", language=language),
        category="github_traction",
        origin="yaml",
        source_id=None,
        source_name=source_name,
    )


def _search_task(
    *,
    query: str = "topic:llm",
    source_name: str = "ai_coding_tools:github_search",
) -> ResolvedTask:
    return ResolvedTask(
        task=GithubSearchTask(kind="github_search", query=query),
        category="ai_coding_tools",
        origin="yaml",
        source_id=None,
        source_name=source_name,
    )


def _repo_dict(
    full_name: str = "anthropics/cookbook",
    *,
    description: str = "Examples for the Claude API",
    stars: int = 1234,
    forks: int = 56,
    open_issues: int = 7,
    language: str = "Python",
    topics: list[str] | None = None,
    pushed_at: str = "2026-05-04T08:00:00Z",
    homepage: str | None = "https://example.com",
) -> dict[str, Any]:
    owner, repo = full_name.split("/", 1)
    return {
        "id": 12345,
        "full_name": full_name,
        "name": repo,
        "owner": {"login": owner, "id": 1},
        "description": description,
        "stargazers_count": stars,
        "watchers_count": stars,
        "forks_count": forks,
        "open_issues_count": open_issues,
        "language": language,
        "topics": topics if topics is not None else ["ai", "llm"],
        "pushed_at": pushed_at,
        "created_at": "2025-01-01T00:00:00Z",
        "html_url": f"https://github.com/{full_name}",
        "homepage": homepage,
    }


def _trending_card_html(
    full_name: str = "anthropics/cookbook",
    *,
    description: str = "Examples for the Claude API",
    language: str | None = "Python",
    stars_total: str = "12,345",
    stars_today: str | None = "234 stars today",
) -> str:
    lang_block = f'<span itemprop="programmingLanguage">{language}</span>' if language else ""
    today_block = f'<span class="float-sm-right">{stars_today}</span>' if stars_today else ""
    return f"""
    <article class="Box-row">
      <h2 class="h3 lh-condensed">
        <a href="/{full_name}">{full_name}</a>
      </h2>
      <p class="col-9 color-fg-muted my-1 pr-4">{description}</p>
      <div class="f6 color-fg-muted mt-2">
        {lang_block}
        <a href="/{full_name}/stargazers">★ {stars_total}</a>
        {today_block}
      </div>
    </article>
    """


def _trending_page(*cards: str) -> str:
    return f"<html><body><main>{''.join(cards)}</main></body></html>"


@pytest.fixture
def patch_client(monkeypatch):
    """Replace ``github.build_client`` with a MockTransport-backed client."""

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

        monkeypatch.setattr("clawfeed_intel.fetchers.github.build_client", fake_build_client)

    return _patch


@pytest.fixture
def github_conn(temp_db):
    c = db.connect(temp_db)
    try:
        yield c
    finally:
        c.close()


# ── parse_trending_html ───────────────────────────────────────────────────────


def test_parse_trending_full_card():
    html = _trending_page(
        _trending_card_html(
            "anthropics/cookbook",
            description="Examples for the Claude API",
            language="Python",
            stars_total="12,345",
            stars_today="234 stars today",
        )
    )
    discovered = parse_trending_html(html)
    assert len(discovered) == 1
    d = discovered[0]
    assert d.full_name == "anthropics/cookbook"
    assert d.description == "Examples for the Claude API"
    assert d.language == "Python"
    assert d.stars_total == 12345
    assert d.stars_today == 234


def test_parse_trending_multiple_cards():
    html = _trending_page(
        _trending_card_html("a/one", description="d1"),
        _trending_card_html("b/two", description="d2"),
        _trending_card_html("c/three", description="d3"),
    )
    discovered = parse_trending_html(html)
    assert [d.full_name for d in discovered] == ["a/one", "b/two", "c/three"]


def test_parse_trending_card_without_description():
    html = _trending_page(_trending_card_html("a/one", description=""))
    discovered = parse_trending_html(html)
    assert len(discovered) == 1
    assert discovered[0].description == ""


def test_parse_trending_card_without_language():
    html = _trending_page(_trending_card_html("a/one", language=None))
    discovered = parse_trending_html(html)
    assert len(discovered) == 1
    assert discovered[0].language is None


def test_parse_trending_stars_total_handles_k_suffix():
    html = _trending_page(_trending_card_html("a/one", stars_total="12.3k"))
    discovered = parse_trending_html(html)
    assert discovered[0].stars_total == 12300


def test_parse_trending_stars_today_optional():
    html = _trending_page(_trending_card_html("a/one", stars_today=None))
    discovered = parse_trending_html(html)
    assert discovered[0].stars_today is None


def test_parse_trending_stars_period_handles_thousands_separator():
    html = _trending_page(_trending_card_html("a/one", stars_today="1,234 stars this week"))
    discovered = parse_trending_html(html)
    assert discovered[0].stars_today == 1234


def test_parse_trending_skips_malformed_card():
    """Card with no h2 link → skipped silently; sibling cards survive."""
    html = (
        "<html><body><main>"
        '<article class="Box-row"><p>no link here</p></article>'
        + _trending_card_html("good/one")
        + "</main></body></html>"
    )
    discovered = parse_trending_html(html)
    assert [d.full_name for d in discovered] == ["good/one"]


def test_parse_trending_skips_non_repo_href():
    """Trending HTML occasionally includes promotional cards with a non-repo
    href (e.g. ``/topics/llm``). We skip these — we only want repos."""
    html = (
        "<html><body><main>"
        '<article class="Box-row"><h2 class="h3"><a href="/topics/llm">topics/llm</a></h2></article>'
        + _trending_card_html("real/repo")
        + "</main></body></html>"
    )
    discovered = parse_trending_html(html)
    assert [d.full_name for d in discovered] == ["real/repo"]


def test_parse_trending_returns_empty_on_blank_input():
    assert parse_trending_html("") == []
    assert parse_trending_html("   ") == []


def test_parse_trending_returns_empty_on_no_cards():
    assert parse_trending_html("<html><body><h1>no repos</h1></body></html>") == []


def test_parse_trending_accepts_string_only():
    assert parse_trending_html(None) == []  # type: ignore[arg-type]
    assert parse_trending_html(123) == []  # type: ignore[arg-type]


# ── parse_search_response ─────────────────────────────────────────────────────


def test_parse_search_full_response():
    body = {
        "total_count": 2,
        "incomplete_results": False,
        "items": [_repo_dict("a/one"), _repo_dict("b/two")],
    }
    repos = parse_search_response(json.dumps(body))
    assert len(repos) == 2
    assert {r["full_name"] for r in repos} == {"a/one", "b/two"}


def test_parse_search_accepts_dict():
    body = {"items": [_repo_dict()]}
    repos = parse_search_response(body)
    assert len(repos) == 1


def test_parse_search_returns_empty_on_missing_items():
    assert parse_search_response({"total_count": 0}) == []


def test_parse_search_returns_empty_on_non_list_items():
    assert parse_search_response({"items": "broken"}) == []


def test_parse_search_returns_empty_on_malformed_json():
    assert parse_search_response("{not valid") == []


def test_parse_search_returns_empty_on_empty_body():
    assert parse_search_response("") == []
    assert parse_search_response("  ") == []


def test_parse_search_filters_non_dict_items():
    body = {"items": ["broken", _repo_dict("a/keep"), 42]}
    repos = parse_search_response(body)
    assert [r["full_name"] for r in repos] == ["a/keep"]


# ── fetch_github_search end-to-end ────────────────────────────────────────────


async def test_search_fetches_and_records_observations(patch_client, github_conn, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    body = {"items": [_repo_dict("anthropics/cookbook", stars=12000)]}
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=body)

    patch_client(handler)
    items = await fetch_github_search(github_conn, _search_task(query="topic:llm"))

    assert len(items) == 1
    item = items[0]
    assert item.source_type == SOURCE_TYPE
    assert item.dedup_key == "anthropics/cookbook"
    assert "anthropics/cookbook" in item.title
    assert "Examples for the Claude API" in item.title
    assert item.metadata["full_name"] == "anthropics/cookbook"
    assert item.metadata["stars"] == 12000
    assert item.metadata["discovered_via"] == "search"
    assert item.metadata["query"] == "topic:llm"
    assert item.metadata["topics"] == ["ai", "llm"]
    assert item.metadata["language"] == "Python"
    # Observation recorded
    rows = github_conn.execute(
        "SELECT full_name, stars, discovered_via FROM github_repo_observations"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["full_name"] == "anthropics/cookbook"
    assert rows[0]["stars"] == 12000
    assert rows[0]["discovered_via"] == "search"
    # Day-1: no velocity yet (single observation)
    assert "velocity" not in item.metadata


async def test_search_attaches_velocity_when_prior_observation_exists(
    patch_client, github_conn, monkeypatch
):
    """Load-bearing test for the architecture doc's hard requirement: velocity
    is computed from stored observations, attached to the item, and reflects
    the time-ordered delta."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    # Seed a 2-day-old observation
    older = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(timespec="seconds")
    db.record_repo_observation(
        github_conn,
        full_name="anthropics/cookbook",
        stars=10000,
        forks=400,
        discovered_via="trending",
        observed_at=older,
    )

    body = {"items": [_repo_dict("anthropics/cookbook", stars=12000, forks=500)]}

    def handler(_request):
        return httpx.Response(200, json=body)

    patch_client(handler)
    items = await fetch_github_search(github_conn, _search_task())

    assert len(items) == 1
    velocity = items[0].metadata.get("velocity")
    assert velocity is not None
    assert velocity["star_delta"] == 2000
    assert velocity["fork_delta"] == 100
    assert velocity["earliest_stars"] == 10000
    assert velocity["latest_stars"] == 12000
    assert velocity["observation_count"] == 2
    assert 1.9 < velocity["days_observed"] < 2.1


async def test_search_constructs_query_url_with_required_params(patch_client, github_conn):
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        return httpx.Response(200, json={"items": []})

    patch_client(handler)
    await fetch_github_search(github_conn, _search_task(query="topic:llm OR topic:agent"))

    assert len(captured) == 1
    url = captured[0]
    assert url.startswith(f"{API_HOST}/search/repositories?")
    assert "sort=stars" in url
    assert "order=desc" in url
    assert "per_page=30" in url
    assert "q=topic" in url


async def test_search_5xx_propagates(patch_client, github_conn):
    def handler(_request):
        return httpx.Response(503, text="upstream unavailable")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_github_search(github_conn, _search_task())


async def test_search_403_rate_limit_propagates(patch_client, github_conn):
    """API rate-limit returns 403; runner should record `failed`."""

    def handler(_request):
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded", "documentation_url": "..."},
        )

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_github_search(github_conn, _search_task())


async def test_search_sends_auth_header_when_token_set(patch_client, github_conn, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_secret_xxx")
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        captured["api_version"] = request.headers.get("X-GitHub-Api-Version")
        return httpx.Response(200, json={"items": []})

    patch_client(handler)
    await fetch_github_search(github_conn, _search_task())

    assert captured["auth"] == "Bearer ghp_test_secret_xxx"
    assert captured["api_version"] == "2022-11-28"


async def test_search_omits_auth_header_when_token_unset(patch_client, github_conn, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"items": []})

    patch_client(handler)
    await fetch_github_search(github_conn, _search_task())

    assert captured["auth"] is None


async def test_search_rejects_non_search_task(github_conn):
    bad = ResolvedTask(
        task=RssTask(kind="rss", url="https://x.example/feed"),
        category="scratch",
        origin="yaml",
        source_id=None,
        source_name="x",
    )
    with pytest.raises(TypeError, match="expected GithubSearchTask"):
        await fetch_github_search(github_conn, bad)


# ── fetch_github_trending end-to-end ──────────────────────────────────────────


async def test_trending_two_step_flow(patch_client, github_conn, monkeypatch):
    """Trending → discovers repos via HTML → enriches each via /repos REST."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    trending_html = _trending_page(
        _trending_card_html("alice/one", description="Project one"),
        _trending_card_html("bob/two", description="Project two"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{TRENDING_HOST}/trending":
            return httpx.Response(200, text=trending_html)
        if url == f"{API_HOST}/repos/alice/one":
            return httpx.Response(200, json=_repo_dict("alice/one", stars=500))
        if url == f"{API_HOST}/repos/bob/two":
            return httpx.Response(200, json=_repo_dict("bob/two", stars=750))
        return httpx.Response(404, json={"unexpected": url})

    patch_client(handler)
    items = await fetch_github_trending(github_conn, _trending_task())

    assert {i.dedup_key for i in items} == {"alice/one", "bob/two"}
    assert all(i.metadata["discovered_via"] == "trending" for i in items)
    # Stars-today carried forward from the trending page
    assert all(i.metadata.get("stars_today") == 234 for i in items)
    # Observations recorded for both
    rows = github_conn.execute(
        "SELECT full_name FROM github_repo_observations ORDER BY full_name"
    ).fetchall()
    assert [r["full_name"] for r in rows] == ["alice/one", "bob/two"]


async def test_trending_language_path(patch_client, github_conn, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        captured.append(url)
        if url.startswith(f"{TRENDING_HOST}/trending"):
            return httpx.Response(200, text=_trending_page())
        return httpx.Response(404, text="")

    patch_client(handler)
    await fetch_github_trending(github_conn, _trending_task(language="Python"))

    assert captured[0] == f"{TRENDING_HOST}/trending/python"


async def test_trending_per_repo_enrichment_failure_skips_only_that_repo(
    patch_client, github_conn, monkeypatch
):
    """SEC-style partial-failure semantics: one bad enrichment must not
    drop the rest of the trending list."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    trending_html = _trending_page(
        _trending_card_html("alice/good", description="OK"),
        _trending_card_html("bob/broken", description="will 5xx"),
        _trending_card_html("carol/good", description="also OK"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{TRENDING_HOST}/trending":
            return httpx.Response(200, text=trending_html)
        if url == f"{API_HOST}/repos/alice/good":
            return httpx.Response(200, json=_repo_dict("alice/good"))
        if url == f"{API_HOST}/repos/bob/broken":
            return httpx.Response(503, text="boom")
        if url == f"{API_HOST}/repos/carol/good":
            return httpx.Response(200, json=_repo_dict("carol/good"))
        return httpx.Response(404, text="")

    patch_client(handler)
    items = await fetch_github_trending(github_conn, _trending_task())
    assert {i.dedup_key for i in items} == {"alice/good", "carol/good"}


async def test_trending_html_5xx_propagates(patch_client, github_conn):
    def handler(_request):
        return httpx.Response(503, text="trending down")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_github_trending(github_conn, _trending_task())


async def test_trending_attaches_velocity_after_prior_observation(
    patch_client, github_conn, monkeypatch
):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    older = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(timespec="seconds")
    db.record_repo_observation(
        github_conn,
        full_name="alice/one",
        stars=400,
        discovered_via="trending",
        observed_at=older,
    )
    trending_html = _trending_page(_trending_card_html("alice/one"))

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{TRENDING_HOST}/trending":
            return httpx.Response(200, text=trending_html)
        if url == f"{API_HOST}/repos/alice/one":
            return httpx.Response(200, json=_repo_dict("alice/one", stars=600))
        return httpx.Response(404, text="")

    patch_client(handler)
    items = await fetch_github_trending(github_conn, _trending_task())

    assert len(items) == 1
    velocity = items[0].metadata.get("velocity")
    assert velocity is not None
    assert velocity["star_delta"] == 200


async def test_trending_returns_empty_when_no_cards(patch_client, github_conn):
    """Empty trending page should not perform any enrichment requests."""
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        captured.append(url)
        return httpx.Response(200, text="<html><body><h1>no cards</h1></body></html>")

    patch_client(handler)
    items = await fetch_github_trending(github_conn, _trending_task())
    assert items == []
    # Only the trending HTML call — no repo enrichment calls
    assert captured == [f"{TRENDING_HOST}/trending"]


async def test_trending_rejects_non_trending_task(github_conn):
    bad = ResolvedTask(
        task=RssTask(kind="rss", url="https://x.example/feed"),
        category="scratch",
        origin="yaml",
        source_id=None,
        source_name="x",
    )
    with pytest.raises(TypeError, match="expected GithubTrendingTask"):
        await fetch_github_trending(github_conn, bad)


# ── item shape (via fetch_github_search end-to-end) ──────────────────────────


async def test_item_title_uses_em_dash_separator_with_description(
    patch_client, github_conn, monkeypatch
):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def handler(_request):
        return httpx.Response(
            200,
            json={"items": [_repo_dict("o/r", description="Cool LLM toolkit")]},
        )

    patch_client(handler)
    items = await fetch_github_search(github_conn, _search_task())
    assert items[0].title == "o/r — Cool LLM toolkit"


async def test_item_title_falls_back_to_full_name_when_no_description(
    patch_client, github_conn, monkeypatch
):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def handler(_request):
        return httpx.Response(200, json={"items": [_repo_dict("o/r", description="")]})

    patch_client(handler)
    items = await fetch_github_search(github_conn, _search_task())
    assert items[0].title == "o/r"


async def test_item_title_truncated_when_overlong(patch_client, github_conn, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    long_desc = "x" * 500

    def handler(_request):
        return httpx.Response(200, json={"items": [_repo_dict("o/r", description=long_desc)]})

    patch_client(handler)
    items = await fetch_github_search(github_conn, _search_task())
    assert len(items[0].title) <= 200
    assert items[0].title.endswith("…")


async def test_item_strips_url_template_fields_from_raw_payload(
    patch_client, github_conn, monkeypatch
):
    """GitHub repo responses include 30+ ``*_url`` template fields that
    bloat raw_payload without informing the brief."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    repo = _repo_dict("o/r")
    repo.update(
        {
            "issues_url": "https://api.github.com/repos/o/r/issues{/number}",
            "tags_url": "https://api.github.com/repos/o/r/tags",
            "permissions": {"admin": True, "push": True, "pull": True},
        }
    )

    def handler(_request):
        return httpx.Response(200, json={"items": [repo]})

    patch_client(handler)
    items = await fetch_github_search(github_conn, _search_task())
    raw = items[0].raw_payload
    # html_url is preserved; *_url templates dropped
    assert "html_url" in raw
    assert "issues_url" not in raw
    assert "tags_url" not in raw
    assert "permissions" not in raw


# ── registration ──────────────────────────────────────────────────────────────


def test_both_kinds_registered():
    assert FETCHER_REGISTRY[KIND_TRENDING] is fetch_github_trending
    assert FETCHER_REGISTRY[KIND_SEARCH] is fetch_github_search


def test_kind_constants_match_source_task_discriminators():
    assert KIND_TRENDING == "github_trending"
    assert KIND_SEARCH == "github_search"
    GithubTrendingTask(kind=KIND_TRENDING)
    GithubSearchTask(kind=KIND_SEARCH, query="anything")
