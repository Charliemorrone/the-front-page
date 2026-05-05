"""arXiv API fetcher.

The arXiv API (``https://export.arxiv.org/api/query``) returns Atom 1.0 with
arxiv-specific extensions (``arxiv:primary_category``, ``arxiv:doi``). We
parse with stdlib ``xml.etree.ElementTree`` rather than feedparser because:

- the arxiv extension elements are exposed inconsistently across feedparser
  versions, and we want them in metadata;
- the source is trusted (arXiv runs the endpoint), so XXE concerns that
  would otherwise push us to ``defusedxml`` don't apply;
- the parser stays small and dependency-free.

Layered the same way as the RSS fetcher:

- :func:`parse_atom_response` is pure — Atom XML in, ``FetchedItem`` out.
- :func:`fetch_arxiv` issues one HTTP query per task, joining the task's
  category list with ``OR``.

Sort order is ``submittedDate desc``; ``max_results=500`` comfortably covers
the 24-hour volume for any plausible AI-research category mix. Items
outside the run's window are filtered downstream by ``published_at``;
the fetcher does not try to express the window as an arXiv date filter
(arXiv's ``submittedDate:[...]`` syntax has known fence-post issues that
would silently drop legitimate items).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import httpx

from .. import normalize
from ..sources import ArxivTask, ResolvedTask
from .base import FETCHER_REGISTRY, FetchedItem
from .http import build_client

log = logging.getLogger(__name__)

KIND = "arxiv"

API_URL = "https://export.arxiv.org/api/query"
MAX_RESULTS = 500
EXCERPT_CHARS = 320

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Match the ID portion of <id>http://arxiv.org/abs/<ID>v<N></id>. Captures
# both modern (``2405.12345v1``) and legacy (``math/0506203v1``) layouts;
# the version suffix is preserved so revisions remain distinct rows.
_ABS_PATH_PREFIX = "/abs/"

_WHITESPACE_RE = re.compile(r"\s+")


async def fetch_arxiv(task: ResolvedTask) -> list[FetchedItem]:
    """Fetch arXiv submissions for the categories named in *task*."""
    if not isinstance(task.task, ArxivTask):
        raise TypeError(f"fetch_arxiv expected ArxivTask, got {type(task.task).__name__}")

    query_url = _build_query_url(task.task.categories)
    async with build_client() as client:
        atom_text = await _fetch_atom(client, query_url)
    return parse_atom_response(atom_text, source_name=task.source_name, query_url=query_url)


def _build_query_url(categories: list[str]) -> str:
    search_query = " OR ".join(f"cat:{cat}" for cat in categories)
    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": "0",
        "max_results": str(MAX_RESULTS),
    }
    return f"{API_URL}?{urlencode(params)}"


async def _fetch_atom(client: httpx.AsyncClient, url: str) -> str:
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.text


# ── parsing (pure) ────────────────────────────────────────────────────────────


def parse_atom_response(text: str, *, source_name: str, query_url: str = "") -> list[FetchedItem]:
    """Parse an arXiv Atom response into :class:`FetchedItem`s.

    Malformed XML returns an empty list (matches the RSS fetcher's lenient
    stance — a one-off arXiv outage shouldn't fail the run, just produce an
    empty pool that coverage can flag).
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        log.warning("arxiv: failed to parse Atom response from %s", source_name)
        return []

    items: list[FetchedItem] = []
    for entry in root.findall("atom:entry", _NS):
        try:
            item = _entry_to_item(entry, source_name=source_name, query_url=query_url)
        except Exception:
            log.exception("arxiv: failed to convert entry from %s", source_name)
            continue
        if item is not None:
            items.append(item)
    return items


def _entry_to_item(entry: ET.Element, *, source_name: str, query_url: str) -> FetchedItem | None:
    arxiv_id = _extract_arxiv_id(entry)
    if not arxiv_id:
        return None

    abs_url = _alternate_link(entry) or _id_url(entry)
    if not abs_url:
        return None
    try:
        canonical_url = normalize.canonicalize_url(abs_url)
    except (TypeError, ValueError):
        return None

    title = _clean_whitespace(_text_of(entry.find("atom:title", _NS)))
    summary = _clean_whitespace(_text_of(entry.find("atom:summary", _NS)))
    authors = _authors(entry)
    published_at = _iso_or_none(_text_of(entry.find("atom:published", _NS)))

    primary, all_categories = _categories(entry)
    pdf_url = _pdf_link(entry)
    doi = _text_of(entry.find("arxiv:doi", _NS)) or None

    metadata: dict[str, Any] = {
        "arxiv_id": arxiv_id,
        "primary_category": primary,
        "categories": all_categories,
        "abs_url": abs_url,
    }
    if pdf_url:
        metadata["pdf_url"] = pdf_url
    if doi:
        metadata["doi"] = doi.strip()
    if query_url:
        metadata["query_url"] = query_url

    return FetchedItem(
        source_type=KIND,
        dedup_key=normalize.arxiv_dedup_key(arxiv_id),
        title=title,
        url=abs_url,
        canonical_url=canonical_url,
        content=summary,
        excerpt=summary[:EXCERPT_CHARS],
        author=", ".join(authors),
        published_at=published_at,
        content_hash=normalize.content_hash(title, summary),
        metadata=metadata,
        raw_payload={"authors": authors},
    )


# ── helpers ───────────────────────────────────────────────────────────────────


def _text_of(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text


def _clean_whitespace(value: str) -> str:
    """arXiv titles and abstracts are wrapped with newlines for line-length;
    collapse to single spaces so downstream stages don't see formatting noise."""
    return _WHITESPACE_RE.sub(" ", value).strip()


def _id_url(entry: ET.Element) -> str:
    return _text_of(entry.find("atom:id", _NS)).strip()


def _extract_arxiv_id(entry: ET.Element) -> str:
    """Pull the arXiv ID (with version) from ``<id>``.

    Versioned format: ``http://arxiv.org/abs/2405.12345v1``.
    Legacy with subject slash: ``http://arxiv.org/abs/math/0506203v2``.
    Both yield everything after ``/abs/``.
    """
    raw = _id_url(entry)
    if not raw:
        return ""
    idx = raw.find(_ABS_PATH_PREFIX)
    if idx < 0:
        return ""
    return raw[idx + len(_ABS_PATH_PREFIX) :].strip()


def _alternate_link(entry: ET.Element) -> str:
    """The ``<link rel="alternate" type="text/html">`` href is the abstract page."""
    for link in entry.findall("atom:link", _NS):
        if link.attrib.get("rel") == "alternate" and link.attrib.get("type") == "text/html":
            return link.attrib.get("href", "").strip()
    return ""


def _pdf_link(entry: ET.Element) -> str:
    for link in entry.findall("atom:link", _NS):
        if link.attrib.get("type") == "application/pdf":
            return link.attrib.get("href", "").strip()
    return ""


def _authors(entry: ET.Element) -> list[str]:
    out: list[str] = []
    for author in entry.findall("atom:author", _NS):
        name = _text_of(author.find("atom:name", _NS)).strip()
        if name:
            out.append(name)
    return out


def _categories(entry: ET.Element) -> tuple[str, list[str]]:
    """Return ``(primary_category, all_categories)``.

    ``arxiv:primary_category`` is authoritative for the primary; ``<category>``
    elements are the full set, including the primary. We deduplicate while
    preserving order to keep the primary-first convention.
    """
    primary_node = entry.find("arxiv:primary_category", _NS)
    primary = primary_node.attrib.get("term", "").strip() if primary_node is not None else ""

    seen: set[str] = set()
    ordered: list[str] = []
    if primary:
        ordered.append(primary)
        seen.add(primary)
    for cat in entry.findall("atom:category", _NS):
        term = cat.attrib.get("term", "").strip()
        if term and term not in seen:
            ordered.append(term)
            seen.add(term)
    return primary, ordered


def _iso_or_none(text: str) -> str | None:
    """Normalize arXiv's ISO-8601 timestamps (``...Z``) to ``+00:00`` form.

    arXiv always emits UTC; we treat a missing offset as UTC rather than
    dropping the field, since the alternative (returning ``None``) would
    make a syntactically-naive timestamp invisible to the relevance window.
    """
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.strip())
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


# Register on import.
FETCHER_REGISTRY[KIND] = fetch_arxiv
