"""Tests for the SEC EDGAR fetcher.

Two test surfaces:

1. ``parse_atom_response`` is pure — exercised with hand-written EDGAR Atom
   covering accession-number extraction, the standard ``"<FORM> - <COMPANY>
   (<CIK>) (Filer)"`` title shape, the title-parse-failure fallback path,
   timezone normalization (-04:00 → +00:00), entries without alternate
   link or accession id, malformed XML.
2. ``fetch_sec`` uses ``httpx.MockTransport`` to assert per-form request
   construction, the merge-and-dedup-by-accession behavior, the partial-
   success policy (one form fails → still return the other), the all-
   forms-fail propagation, the contact-bearing User-Agent (SEC compliance),
   and the non-SecEdgarTask type guard.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from clawfeed_intel.fetchers import FETCHER_REGISTRY
from clawfeed_intel.fetchers.sec import (
    API_URL,
    KIND,
    RESULT_COUNT,
    fetch_sec,
    parse_atom_response,
)
from clawfeed_intel.sources import ResolvedTask, SecEdgarTask


# ── fixtures ──────────────────────────────────────────────────────────────────


D_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Latest Filings - Mon, 04 May 2026 12:00:00 EDT</title>
  <updated>2026-05-04T12:00:00-04:00</updated>
  <id>https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&amp;type=D</id>
  <entry>
    <category scheme="https://www.sec.gov/" term="form-type" label="form type"/>
    <id>urn:tag:sec.gov,2008:accession-number=0001234567-26-000001</id>
    <link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/0001234567-26-000001-index.htm"/>
    <summary type="html">D - VECTORFORGE TECHNOLOGIES INC (0001234567) (Filer)</summary>
    <title>D - VECTORFORGE TECHNOLOGIES INC (0001234567) (Filer)</title>
    <updated>2026-05-04T11:30:00-04:00</updated>
  </entry>
  <entry>
    <category scheme="https://www.sec.gov/" term="form-type" label="form type"/>
    <id>urn:tag:sec.gov,2008:accession-number=0009876543-26-000017</id>
    <link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/9876543/000987654326000017/0009876543-26-000017-index.htm"/>
    <summary type="html">D - QUANTUM AGENTS LLC (0009876543) (Filer)</summary>
    <title>D - QUANTUM AGENTS LLC (0009876543) (Filer)</title>
    <updated>2026-05-04T10:15:00-04:00</updated>
  </entry>
</feed>
"""


D_AMENDMENT_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Latest Filings - D/A</title>
  <updated>2026-05-04T12:00:00-04:00</updated>
  <id>https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&amp;type=D%2FA</id>
  <entry>
    <id>urn:tag:sec.gov,2008:accession-number=0001234567-26-000001</id>
    <link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/0001234567-26-000001-index.htm"/>
    <summary type="html">D/A - VECTORFORGE TECHNOLOGIES INC (0001234567) (Filer)</summary>
    <title>D/A - VECTORFORGE TECHNOLOGIES INC (0001234567) (Filer)</title>
    <updated>2026-05-04T11:45:00-04:00</updated>
  </entry>
  <entry>
    <id>urn:tag:sec.gov,2008:accession-number=0005555555-26-000003</id>
    <link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/5555555/000555555526000003/0005555555-26-000003-index.htm"/>
    <summary type="html">D/A - SHIPCRAFT INC (0005555555) (Filer)</summary>
    <title>D/A - SHIPCRAFT INC (0005555555) (Filer)</title>
    <updated>2026-05-04T09:00:00-04:00</updated>
  </entry>
</feed>
"""


def _task(
    *,
    forms: list[str] | None = None,
    source_name: str = "sec:funding",
) -> ResolvedTask:
    return ResolvedTask(
        task=SecEdgarTask(kind="sec_edgar", forms=forms or ["D", "D/A"]),
        category="startup_funding",
        origin="yaml",
        source_id=None,
        source_name=source_name,
    )


@pytest.fixture
def patch_client(monkeypatch):
    """Replace ``sec.build_client`` with a MockTransport-backed client."""

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

        monkeypatch.setattr("clawfeed_intel.fetchers.sec.build_client", fake_build_client)

    return _patch


# ── parse_atom_response (pure) ────────────────────────────────────────────────


def test_parse_extracts_accession_form_company_cik():
    items = parse_atom_response(D_FIXTURE, source_name="sec:funding", form="D")

    assert len(items) == 2
    first = items[0]
    assert first.source_type == "sec_edgar"
    assert first.dedup_key == "0001234567-26-000001"
    assert first.metadata["accession_number"] == "0001234567-26-000001"
    assert first.metadata["form_type"] == "D"
    assert first.metadata["company_name"] == "VECTORFORGE TECHNOLOGIES INC"
    assert first.metadata["cik"] == "0001234567"
    assert first.author == "VECTORFORGE TECHNOLOGIES INC"
    assert first.title.startswith("D -")
    assert first.url.startswith("https://www.sec.gov/Archives/edgar/data/")
    # canonicalize_url strips the ``www.`` prefix; preserve the original in
    # `url` for citation, use the canonical form for dedup.
    assert first.canonical_url.startswith("https://sec.gov/Archives/edgar/data/")
    assert first.content_hash is not None


def test_parse_normalizes_published_at_to_utc():
    """SEC emits ``-04:00`` (Eastern); we want ``+00:00`` for project consistency."""
    items = parse_atom_response(D_FIXTURE, source_name="sec", form="D")
    first = items[0]
    # 11:30 EDT (-04:00) → 15:30 UTC
    assert first.published_at == "2026-05-04T15:30:00+00:00"


def test_parse_records_query_url_in_metadata_when_supplied():
    items = parse_atom_response(
        D_FIXTURE,
        source_name="sec",
        query_url="https://www.sec.gov/cgi-bin/browse-edgar?type=D",
        form="D",
    )
    assert items[0].metadata["query_url"] == "https://www.sec.gov/cgi-bin/browse-edgar?type=D"


def test_parse_uses_requested_form_when_title_unparseable():
    """If EDGAR ever changes its title format, we degrade gracefully:
    keep the entry, pull the requested form into metadata."""
    body = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <id>urn:tag:sec.gov,2008:accession-number=0000000001-26-999999</id>
  <link rel="alternate" type="text/html" href="https://www.sec.gov/x"/>
  <title>Some Unparseable Format</title>
  <summary>x</summary>
  <updated>2026-05-04T12:00:00-04:00</updated>
</entry>
</feed>"""
    items = parse_atom_response(body, source_name="sec", form="D")
    assert len(items) == 1
    item = items[0]
    assert item.dedup_key == "0000000001-26-999999"
    assert item.metadata["form_type"] == "D"  # falls back to the request
    assert item.metadata["company_name"] == ""
    assert item.metadata["cik"] == ""


def test_parse_records_requested_form_when_titles_form_differs():
    """Some EDGAR pipelines emit D/A entries on the D query. We surface
    the title's form (truth) and stamp the request as ``requested_form``."""
    body = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <id>urn:tag:sec.gov,2008:accession-number=0000000002-26-000001</id>
  <link rel="alternate" type="text/html" href="https://www.sec.gov/y"/>
  <title>D/A - EXAMPLE CO (0000000002) (Filer)</title>
  <summary>D/A - EXAMPLE CO</summary>
  <updated>2026-05-04T12:00:00-04:00</updated>
</entry>
</feed>"""
    items = parse_atom_response(body, source_name="sec", form="D")
    assert len(items) == 1
    assert items[0].metadata["form_type"] == "D/A"
    assert items[0].metadata["requested_form"] == "D"


def test_parse_skips_entry_without_accession_id():
    body = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <id>urn:tag:sec.gov,2008:other-thing=foo</id>
  <link rel="alternate" type="text/html" href="https://www.sec.gov/x"/>
  <title>D - X CO (0000000001) (Filer)</title>
  <summary>x</summary>
  <updated>2026-05-04T12:00:00-04:00</updated>
</entry>
<entry>
  <id>urn:tag:sec.gov,2008:accession-number=0000000002-26-000002</id>
  <link rel="alternate" type="text/html" href="https://www.sec.gov/y"/>
  <title>D - Y CO (0000000002) (Filer)</title>
  <summary>y</summary>
  <updated>2026-05-04T12:00:00-04:00</updated>
</entry>
</feed>"""
    items = parse_atom_response(body, source_name="sec", form="D")
    assert [i.dedup_key for i in items] == ["0000000002-26-000002"]


def test_parse_skips_entry_without_alternate_link():
    body = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <id>urn:tag:sec.gov,2008:accession-number=0000000001-26-000001</id>
  <title>D - X CO (0000000001) (Filer)</title>
  <summary>x</summary>
  <updated>2026-05-04T12:00:00-04:00</updated>
</entry>
</feed>"""
    items = parse_atom_response(body, source_name="sec", form="D")
    assert items == []


def test_parse_empty_feed_returns_empty():
    body = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<id>x</id><updated>2026-05-04T12:00:00-04:00</updated>
</feed>"""
    assert parse_atom_response(body, source_name="sec", form="D") == []


def test_parse_malformed_xml_returns_empty_not_raises():
    assert parse_atom_response("<not-xml", source_name="sec", form="D") == []
    assert parse_atom_response("", source_name="sec", form="D") == []


def test_dedup_key_is_accession_not_url():
    """SEC's accession number is the canonical filing identifier — globally
    unique across all filers. URL-based dedup would miss e.g. mirrored
    filing pages."""
    items = parse_atom_response(D_FIXTURE, source_name="sec", form="D")
    keys = {i.dedup_key for i in items}
    assert keys == {"0001234567-26-000001", "0009876543-26-000017"}


# ── fetch_sec with MockTransport ──────────────────────────────────────────────


async def test_fetch_builds_per_form_query_urls(patch_client, conn):
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        url = str(request.url)
        if "type=D&" in url or url.endswith("type=D"):
            return httpx.Response(200, text=D_FIXTURE)
        if "type=D%2FA" in url or "type=D/A" in url:
            return httpx.Response(200, text=D_AMENDMENT_FIXTURE)
        return httpx.Response(404)

    patch_client(handler)
    items = await fetch_sec(conn, _task(forms=["D", "D/A"]))

    # Two requests, one per form
    assert len(captured) == 2
    paths = sorted({urlsplit(u).path for u in captured})
    assert paths == ["/cgi-bin/browse-edgar"]

    # Verify query parameter shapes
    first_qs = parse_qs(urlsplit(captured[0]).query)
    assert first_qs["action"] == ["getcurrent"]
    assert first_qs["output"] == ["atom"]
    assert first_qs["count"] == [str(RESULT_COUNT)]

    # Items: 2 from D + 2 from D/A, but accession 0001234567-26-000001
    # appears in both fixtures → deduped to one.
    assert len(items) == 3
    keys = sorted(i.dedup_key for i in items)
    assert keys == [
        "0001234567-26-000001",
        "0005555555-26-000003",
        "0009876543-26-000017",
    ]


async def test_fetch_records_query_url_in_metadata(patch_client, conn):
    def handler(_request):
        return httpx.Response(200, text=D_FIXTURE)

    patch_client(handler)
    items = await fetch_sec(conn, _task(forms=["D"]))
    for item in items:
        assert API_URL in item.metadata.get("query_url", "")


async def test_fetch_partial_failure_returns_other_form_results(patch_client, conn):
    """One form 5xx → other form's items still returned; task is 'succeeded'."""

    def handler(request):
        url = str(request.url)
        if "type=D&" in url or url.endswith("type=D"):
            return httpx.Response(200, text=D_FIXTURE)
        if "type=D%2FA" in url or "type=D/A" in url:
            return httpx.Response(503, text="upstream unavailable")
        return httpx.Response(404)

    patch_client(handler)
    items = await fetch_sec(conn, _task(forms=["D", "D/A"]))

    # Only the D fixture's items survive; partial degradation, not failure.
    assert {i.dedup_key for i in items} == {
        "0001234567-26-000001",
        "0009876543-26-000017",
    }


async def test_fetch_total_failure_propagates(patch_client, conn):
    """All forms fail → re-raise so the runner records the task as failed."""

    def handler(_request):
        return httpx.Response(503, text="upstream unavailable")

    patch_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_sec(conn, _task(forms=["D", "D/A"]))


async def test_fetch_records_ua_with_contact(patch_client, conn):
    """SEC explicitly requires a contact-bearing UA. Verify it's set."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["ua"] = request.headers.get("User-Agent")
        return httpx.Response(200, text=D_FIXTURE)

    patch_client(handler)
    await fetch_sec(conn, _task(forms=["D"]))
    assert "ClawFeed-Intel" in (captured["ua"] or "")
    assert "+contact:" in (captured["ua"] or "")


async def test_fetch_rejects_non_sec_task(conn):
    from clawfeed_intel.sources import RssTask

    bad = ResolvedTask(
        task=RssTask(kind="rss", url="https://x.example/feed"),
        category="scratch",
        origin="yaml",
        source_id=None,
        source_name="x",
    )
    with pytest.raises(TypeError, match="expected SecEdgarTask"):
        await fetch_sec(conn, bad)


# ── registration ──────────────────────────────────────────────────────────────


def test_sec_fetcher_is_registered():
    assert FETCHER_REGISTRY[KIND] is fetch_sec


def test_kind_constant_matches_source_task_discriminator():
    assert KIND == "sec_edgar"
    SecEdgarTask(kind=KIND, forms=["D"])
