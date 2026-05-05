"""SEC EDGAR fetcher.

Daily-brief use case: surface recent filings of specific form types
(Form D and D/A for startup-financing signals; potentially 8-K, S-1,
etc. as the editorial config grows). The legacy ``getcurrent`` Atom
endpoint at ``https://www.sec.gov/cgi-bin/browse-edgar`` is the right tool
for this — it returns the most recent filings of a form type, in real
time, in a stable Atom shape that's been live for over a decade. The
modern ``data.sec.gov`` JSON API is per-CIK only; that's a Phase 7
(topical search) concern, not a daily-brief concern.

Two-layer pattern, consistent with the other fetchers:

- :func:`parse_atom_response` is pure — Atom XML in, ``FetchedItem`` out.
- :func:`fetch_sec` does the HTTP. One request per form in the task's
  ``forms`` list, fanned out concurrently, with results merged and
  deduplicated by accession number (the same filing can in principle be
  surfaced by both ``D`` and a separate query, though in practice form
  filtering is exact).

Failure model:
- All-form failure → re-raise so the runner records ``failed``.
- Partial failure (one form fails, another succeeds) → return the
  successes; log the failures. A noisy SEC outage shouldn't drop the
  filings we did manage to retrieve.
- Per-entry parse failure → log + skip; sibling entries continue.

SEC compliance: the polite, contact-bearing User-Agent from
:mod:`fetchers.http` is what SEC explicitly requires. Their published
ceiling of ≤10 req/s is met trivially — we issue at most one request per
form per task, so 1-2 requests per fetch.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import httpx

from .. import normalize
from ..sources import ResolvedTask, SecEdgarTask
from .base import FETCHER_REGISTRY, FetchedItem
from .http import build_client

log = logging.getLogger(__name__)

KIND = "sec_edgar"

API_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
RESULT_COUNT = 100  # max for getcurrent Atom; default 40
EXCERPT_CHARS = 320

_NS = {"atom": "http://www.w3.org/2005/Atom"}

# `<id>urn:tag:sec.gov,2008:accession-number=0001234567-26-000001</id>` →
# accession-number group. Applied with re.search so leading-text variations
# (some SEC tools emit `tag:sec.gov,2008:...` without `urn:`) all match.
_ACCESSION_RE = re.compile(r"accession-number=([0-9-]+)")

# Title shape: ``<FORM> - <COMPANY NAME> (<10-digit-CIK>) (Filer|Issuer)``
# stable across decades of EDGAR Atom output. We don't depend on the
# optional structured ``<content>`` block, whose namespace handling is
# inconsistent across SEC's serializers.
_TITLE_RE = re.compile(r"^(?P<form>[A-Z0-9/.\-]+)\s*-\s*(?P<company>.+?)\s*\((?P<cik>\d{6,10})\)")


async def fetch_sec(task: ResolvedTask) -> list[FetchedItem]:
    """Fetch recent SEC filings for the form types named in *task*.

    One request per form, fanned out concurrently. Partial successes are
    returned; total failure re-raises so the runner records the task as
    failed.
    """
    if not isinstance(task.task, SecEdgarTask):
        raise TypeError(f"fetch_sec expected SecEdgarTask, got {type(task.task).__name__}")
    # task.task.ciks is reserved for Phase 7 topical search where
    # someone wants filings from specific filers (e.g. Khosla Ventures).
    # Daily-brief use case is form-only; CIK filter would be a post-step.

    forms = task.task.forms
    async with build_client() as client:
        results = await asyncio.gather(
            *(_fetch_atom_for_form(client, form) for form in forms),
            return_exceptions=True,
        )

    items: list[FetchedItem] = []
    seen_accessions: set[str] = set()
    failures: list[BaseException] = []

    for form, result in zip(forms, results, strict=True):
        if isinstance(result, BaseException):
            failures.append(result)
            log.warning("sec: form %r fetch failed: %s", form, result)
            continue
        atom_text, query_url = result
        for item in parse_atom_response(
            atom_text, source_name=task.source_name, query_url=query_url, form=form
        ):
            if item.dedup_key in seen_accessions:
                continue
            seen_accessions.add(item.dedup_key)
            items.append(item)

    # Total failure: every form errored. Surface the first error so the
    # runner records the task as failed rather than silently empty.
    if failures and len(failures) == len(forms):
        raise failures[0]
    return items


async def _fetch_atom_for_form(client: httpx.AsyncClient, form: str) -> tuple[str, str]:
    url = _build_query_url(form)
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.text, url


def _build_query_url(form: str) -> str:
    params = {
        "action": "getcurrent",
        "type": form,
        "company": "",
        "dateb": "",
        "owner": "include",
        "count": str(RESULT_COUNT),
        "output": "atom",
    }
    return f"{API_URL}?{urlencode(params)}"


# ── parsing (pure) ────────────────────────────────────────────────────────────


def parse_atom_response(
    text: str,
    *,
    source_name: str,
    query_url: str = "",
    form: str = "",
) -> list[FetchedItem]:
    """Parse an EDGAR Atom feed into :class:`FetchedItem`s.

    Malformed XML returns an empty list (matches the other fetchers' lenient
    stance — a one-off SEC glitch shouldn't fail the run).
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        log.warning("sec: failed to parse Atom response from %s", source_name)
        return []

    items: list[FetchedItem] = []
    for entry in root.findall("atom:entry", _NS):
        try:
            item = _entry_to_item(
                entry, source_name=source_name, query_url=query_url, requested_form=form
            )
        except Exception:
            log.exception("sec: failed to convert entry from %s", source_name)
            continue
        if item is not None:
            items.append(item)
    return items


def _entry_to_item(
    entry: ET.Element,
    *,
    source_name: str,
    query_url: str,
    requested_form: str,
) -> FetchedItem | None:
    accession = _accession_number(entry)
    if not accession:
        return None

    url = _alternate_link(entry)
    if not url:
        return None
    try:
        canonical_url = normalize.canonicalize_url(url)
    except (TypeError, ValueError):
        return None

    title_text = _text_of(entry.find("atom:title", _NS)).strip()
    summary_text = _text_of(entry.find("atom:summary", _NS)).strip()
    parsed = _parse_title(title_text)

    content = summary_text or title_text
    published_at = _iso_or_none(_text_of(entry.find("atom:updated", _NS)))

    metadata: dict[str, Any] = {
        "accession_number": accession,
        "filing_url": url,
        # The `form` we surface is whatever the title actually says; the
        # request type is the *requested* form, which can differ for amendments
        # (a "D/A" might be returned by a "D" query in some setups).
        "form_type": parsed.form or requested_form,
        "company_name": parsed.company,
        "cik": parsed.cik,
    }
    if requested_form and parsed.form and requested_form != parsed.form:
        metadata["requested_form"] = requested_form
    if query_url:
        metadata["query_url"] = query_url

    return FetchedItem(
        source_type=KIND,
        dedup_key=normalize.sec_dedup_key(accession),
        title=title_text,
        url=url,
        canonical_url=canonical_url,
        content=content,
        excerpt=content[:EXCERPT_CHARS],
        author=parsed.company,
        published_at=published_at,
        content_hash=normalize.content_hash(title_text, content),
        metadata=metadata,
        raw_payload={"summary": summary_text} if summary_text else {},
    )


# ── helpers ───────────────────────────────────────────────────────────────────


class _ParsedTitle:
    """Light value type so the `_parse_title` return is self-documenting."""

    __slots__ = ("form", "company", "cik")

    def __init__(self, form: str = "", company: str = "", cik: str = "") -> None:
        self.form = form
        self.company = company
        self.cik = cik


def _parse_title(title: str) -> _ParsedTitle:
    """Extract (form, company, cik) from EDGAR's `<title>` shape.

    Returns a :class:`_ParsedTitle` with empty strings on no match so the
    caller can degrade gracefully. We never drop an item over title parse
    failure — accession number is enough to dedup.
    """
    if not title:
        return _ParsedTitle()
    m = _TITLE_RE.match(title)
    if not m:
        return _ParsedTitle()
    return _ParsedTitle(
        form=m.group("form").strip(),
        company=m.group("company").strip(),
        cik=m.group("cik"),
    )


def _accession_number(entry: ET.Element) -> str:
    raw = _text_of(entry.find("atom:id", _NS)).strip()
    if not raw:
        return ""
    m = _ACCESSION_RE.search(raw)
    return m.group(1) if m else ""


def _alternate_link(entry: ET.Element) -> str:
    for link in entry.findall("atom:link", _NS):
        if link.attrib.get("rel") == "alternate":
            return link.attrib.get("href", "").strip()
    return ""


def _text_of(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text


def _iso_or_none(text: str) -> str | None:
    """SEC's Atom uses zoned timestamps like ``2026-05-04T12:30:00-04:00``.
    Normalize to UTC for project-wide consistency."""
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.strip())
    except ValueError:
        return None
    if dt.tzinfo is None:
        # SEC always emits an offset; if we ever see a naive timestamp
        # it's a bug in the feed, not our problem to interpret.
        return None
    from datetime import timezone

    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


# Register on import.
FETCHER_REGISTRY[KIND] = fetch_sec
