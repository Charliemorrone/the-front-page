"""Tests for the arXiv API fetcher.

Two test surfaces, mirroring the RSS fetcher:

1. ``parse_atom_response`` is pure — exercised with hand-written Atom XML
   fixtures that cover modern (``2405.12345``), legacy (``math/0506203``),
   missing fields, multiple authors, primary vs secondary categories, DOI
   extension, whitespace-collapsed titles/abstracts, and malformed input.
2. ``fetch_arxiv`` is exercised with ``httpx.MockTransport``. We verify the
   request URL — search_query OR-joining, sortBy, max_results — and the
   contact-bearing User-Agent.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from clawfeed_intel.fetchers import FETCHER_REGISTRY
from clawfeed_intel.fetchers.arxiv import (
    KIND,
    MAX_RESULTS,
    fetch_arxiv,
    parse_atom_response,
)
from clawfeed_intel.sources import ArxivTask, ResolvedTask


# ── fixtures ──────────────────────────────────────────────────────────────────


ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <link href="http://arxiv.org/api/query?search_query=cat:cs.AI" rel="self" type="application/atom+xml"/>
  <title type="html">arXiv Query: search_query=cat:cs.AI</title>
  <id>http://arxiv.org/api/abc</id>
  <updated>2026-05-04T12:00:00Z</updated>
  <opensearch:totalResults>2</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>500</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2405.12345v1</id>
    <updated>2026-05-04T10:00:00Z</updated>
    <published>2026-05-04T08:00:00Z</published>
    <title>A Novel
  Approach to LLM
  Reasoning</title>
    <summary>  We present a new
  approach to reasoning over LLMs that improves
  performance.  </summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <arxiv:doi>10.1234/example</arxiv:doi>
    <link href="https://arxiv.org/abs/2405.12345v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="https://arxiv.org/pdf/2405.12345v1" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/math/0506203v2</id>
    <updated>2026-05-04T11:00:00Z</updated>
    <published>2005-06-09T00:00:00Z</published>
    <title>Legacy Math Paper</title>
    <summary>An old paper, revised today.</summary>
    <author><name>Carol Researcher</name></author>
    <link href="https://arxiv.org/abs/math/0506203v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="https://arxiv.org/pdf/math/0506203v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="math.GT" scheme="http://arxiv.org/schemas/atom"/>
    <category term="math.GT" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


EMPTY_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <id>http://arxiv.org/api/empty</id>
  <updated>2026-05-04T12:00:00Z</updated>
</feed>
"""


def _task(
    categories: list[str] | None = None,
    source_name: str = "arxiv:cs.AI",
) -> ResolvedTask:
    return ResolvedTask(
        task=ArxivTask(kind="arxiv", categories=categories or ["cs.AI", "cs.LG"]),
        category="ai_research",
        origin="yaml",
        source_id=None,
        source_name=source_name,
    )


@pytest.fixture
def patch_client(monkeypatch):
    """Replace ``arxiv.build_client`` with a MockTransport-backed client."""

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

        monkeypatch.setattr("clawfeed_intel.fetchers.arxiv.build_client", fake_build_client)

    return _patch


# ── parse_atom_response (pure) ────────────────────────────────────────────────


def test_parse_extracts_modern_id_with_version_kept_in_dedup_key():
    items = parse_atom_response(ATOM_FIXTURE, source_name="arxiv:cs.AI")

    modern = next(i for i in items if "2405" in i.dedup_key)
    assert modern.dedup_key == "2405.12345v1"
    assert modern.metadata["arxiv_id"] == "2405.12345v1"
    assert modern.canonical_url == "https://arxiv.org/abs/2405.12345v1"
    # Version distinct from a future v2 — clustering will recognize them
    # as the same logical paper.
    assert "2405.12345v2" != modern.dedup_key


def test_parse_extracts_legacy_id_with_subject_slash():
    items = parse_atom_response(ATOM_FIXTURE, source_name="arxiv:math.GT")
    legacy = next(i for i in items if "math" in i.dedup_key)
    assert legacy.dedup_key == "math/0506203v2"
    assert legacy.metadata["arxiv_id"] == "math/0506203v2"
    assert legacy.canonical_url == "https://arxiv.org/abs/math/0506203v2"


def test_parse_collapses_whitespace_in_title_and_summary():
    items = parse_atom_response(ATOM_FIXTURE, source_name="arxiv")
    modern = next(i for i in items if "2405" in i.dedup_key)
    assert modern.title == "A Novel Approach to LLM Reasoning"
    assert modern.content.startswith("We present a new approach")
    assert "\n" not in modern.content
    # Multiple internal spaces collapsed
    assert "  " not in modern.content
    assert "  " not in modern.title


def test_parse_joins_multiple_authors_with_comma_space():
    items = parse_atom_response(ATOM_FIXTURE, source_name="arxiv")
    modern = next(i for i in items if "2405" in i.dedup_key)
    assert modern.author == "Alice Smith, Bob Jones"
    # Raw author list preserved for downstream stages (e.g., entity extraction)
    assert modern.raw_payload["authors"] == ["Alice Smith", "Bob Jones"]


def test_parse_records_primary_and_all_categories():
    items = parse_atom_response(ATOM_FIXTURE, source_name="arxiv")
    modern = next(i for i in items if "2405" in i.dedup_key)
    assert modern.metadata["primary_category"] == "cs.AI"
    # Primary appears first; secondary follows; no duplicates.
    assert modern.metadata["categories"] == ["cs.AI", "cs.LG"]


def test_parse_records_pdf_url_and_doi_when_present():
    items = parse_atom_response(ATOM_FIXTURE, source_name="arxiv")
    modern = next(i for i in items if "2405" in i.dedup_key)
    assert modern.metadata["pdf_url"] == "https://arxiv.org/pdf/2405.12345v1"
    assert modern.metadata["doi"] == "10.1234/example"

    legacy = next(i for i in items if "math" in i.dedup_key)
    assert "doi" not in legacy.metadata  # not all entries have DOI


def test_parse_records_published_at_in_utc_iso():
    items = parse_atom_response(ATOM_FIXTURE, source_name="arxiv")
    modern = next(i for i in items if "2405" in i.dedup_key)
    assert modern.published_at == "2026-05-04T08:00:00+00:00"


def test_parse_handles_naive_timestamp_as_utc():
    """If arXiv ever omits the Z suffix, we assume UTC rather than dropping
    the timestamp — the API spec is UTC-only."""
    body = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <id>http://arxiv.org/abs/2401.99999v1</id>
  <title>Naive Timestamp</title>
  <summary>x</summary>
  <published>2026-05-04T08:00:00</published>
  <link href="https://arxiv.org/abs/2401.99999v1" rel="alternate" type="text/html"/>
</entry>
</feed>"""
    items = parse_atom_response(body, source_name="arxiv")
    assert len(items) == 1
    assert items[0].published_at == "2026-05-04T08:00:00+00:00"


def test_parse_empty_feed_returns_empty():
    assert parse_atom_response(EMPTY_FIXTURE, source_name="arxiv") == []


def test_parse_malformed_xml_returns_empty_not_raises():
    """A one-off arXiv glitch shouldn't fail the run — log and return empty."""
    assert parse_atom_response("<not-xml", source_name="arxiv") == []
    assert parse_atom_response("", source_name="arxiv") == []


def test_parse_skips_entry_without_id():
    body = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <title>No id</title>
  <summary>x</summary>
  <link href="https://arxiv.org/abs/2401.99999v1" rel="alternate" type="text/html"/>
</entry>
<entry>
  <id>http://arxiv.org/abs/2401.00001v1</id>
  <title>Has id</title>
  <summary>y</summary>
  <link href="https://arxiv.org/abs/2401.00001v1" rel="alternate" type="text/html"/>
</entry>
</feed>"""
    items = parse_atom_response(body, source_name="arxiv")
    assert [i.title for i in items] == ["Has id"]


def test_parse_skips_entry_without_alternate_link():
    """Without the alternate link there's no abstract URL, no canonical_url."""
    body = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <id>http://arxiv.org/abs/2401.00001v1</id>
  <title>No link</title>
  <summary>x</summary>
</entry>
</feed>"""
    items = parse_atom_response(body, source_name="arxiv")
    # Falls back to using <id> itself as URL since it carries the abs path.
    assert len(items) == 1
    assert items[0].url == "http://arxiv.org/abs/2401.00001v1"


def test_dedup_key_is_arxiv_id_not_canonical_url():
    """arXiv is the one fetcher whose dedup_key is the structured ID, not a
    URL. Different mirror URLs of the same paper must collide in raw_items."""
    items = parse_atom_response(ATOM_FIXTURE, source_name="arxiv")
    keys = {i.dedup_key for i in items}
    urls = {i.canonical_url for i in items}
    assert keys == {"2405.12345v1", "math/0506203v2"}
    assert keys != urls  # by design


# ── fetch_arxiv with MockTransport ────────────────────────────────────────────


async def test_fetch_arxiv_builds_query_url(patch_client):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["ua"] = request.headers.get("User-Agent")
        return httpx.Response(200, text=ATOM_FIXTURE)

    patch_client(handler)
    items = await fetch_arxiv(_task(["cs.AI", "cs.LG"]))
    assert len(items) == 2

    parts = urlsplit(captured["url"])
    assert parts.scheme == "https"
    assert parts.netloc == "export.arxiv.org"
    assert parts.path == "/api/query"

    qs = parse_qs(parts.query)
    assert qs["search_query"] == ["cat:cs.AI OR cat:cs.LG"]
    assert qs["sortBy"] == ["submittedDate"]
    assert qs["sortOrder"] == ["descending"]
    assert qs["start"] == ["0"]
    assert qs["max_results"] == [str(MAX_RESULTS)]

    assert "ClawFeed-Intel" in (captured["ua"] or "")
    assert "+contact:" in (captured["ua"] or "")


async def test_fetch_arxiv_single_category_query(patch_client):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, text=ATOM_FIXTURE)

    patch_client(handler)
    await fetch_arxiv(_task(["cs.AI"]))

    parts = urlsplit(captured["url"])
    qs = parse_qs(parts.query)
    assert qs["search_query"] == ["cat:cs.AI"]


async def test_fetch_arxiv_records_query_url_in_metadata(patch_client):
    def handler(_request):
        return httpx.Response(200, text=ATOM_FIXTURE)

    patch_client(handler)
    items = await fetch_arxiv(_task(["cs.AI"]))
    for item in items:
        assert "query_url" in item.metadata
        assert "search_query=cat" in item.metadata["query_url"]


async def test_fetch_arxiv_raises_on_5xx(patch_client):
    """Upstream errors must propagate so the runner records 'failed'."""

    def handler(_request):
        return httpx.Response(503, text="upstream unavailable")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_arxiv(_task())


async def test_fetch_arxiv_rejects_non_arxiv_task():
    from clawfeed_intel.sources import RssTask

    bad_task = ResolvedTask(
        task=RssTask(kind="rss", url="https://x.example/feed"),
        category="scratch",
        origin="yaml",
        source_id=None,
        source_name="x",
    )
    with pytest.raises(TypeError, match="expected ArxivTask"):
        await fetch_arxiv(bad_task)


# ── registration ──────────────────────────────────────────────────────────────


def test_arxiv_fetcher_is_registered():
    assert FETCHER_REGISTRY[KIND] is fetch_arxiv


def test_kind_constant_matches_source_task_discriminator():
    assert KIND == "arxiv"
    ArxivTask(kind=KIND, categories=["cs.AI"])
