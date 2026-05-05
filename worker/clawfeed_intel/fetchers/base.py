"""Fetcher contract: what fetchers consume, what they produce.

A fetcher is an async callable: ``(Connection, ResolvedTask) -> list[FetchedItem]``.
No class hierarchy, no Protocol required at runtime — keeping the surface
flat means each of the eight fetchers (RSS / arXiv / HN / SEC / GDELT /
Reddit / GitHub / website) can be one async function in its own module.

The connection is passed because some fetchers need DB access during the
fetch — most prominently GitHub, which records per-repo star/fork
observations to ``github_repo_observations`` and reads velocity back to
attach to each item's metadata. Fetchers that don't need DB access (every
fetcher except GitHub today) accept the parameter and ignore it.

``FetchedItem`` shapes match :func:`db.upsert_raw_item`'s parameters so the
runner can pass items through with no translation. ``source_type`` carries
the fetcher kind (``rss``, ``arxiv``, …) — that's what
``raw_items.source_type`` stores and what ``upsert_raw_item`` keys conflicts
on, alongside ``dedup_key``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..sources import ResolvedTask


@dataclass(frozen=True)
class FetchedItem:
    """One normalized item produced by a fetcher.

    Constructed from the fetcher's normalized view of a raw response.
    Producers must already have run :func:`normalize.canonicalize_url` on
    ``url`` to populate ``canonical_url`` and :func:`normalize.content_hash`
    on the body to populate ``content_hash``; the runner does not redo this
    work.
    """

    source_type: str
    dedup_key: str
    title: str
    url: str
    canonical_url: str
    content: str
    excerpt: str = ""
    author: str = ""
    published_at: str | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def upsert_kwargs(self) -> dict[str, Any]:
        """Adapt to :func:`db.upsert_raw_item`'s keyword shape.

        ``content_hash`` is renamed to ``content_hash_value`` to match the
        upsert helper, which uses the suffix to avoid clashing with the
        :func:`normalize.content_hash` callable in its module namespace.
        """
        return {
            "source_type": self.source_type,
            "dedup_key": self.dedup_key,
            "title": self.title,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "content": self.content,
            "excerpt": self.excerpt,
            "author": self.author,
            "published_at": self.published_at,
            "content_hash_value": self.content_hash,
            "metadata": dict(self.metadata),
            "raw_payload": dict(self.raw_payload),
        }


@dataclass
class FetchOutcome:
    """Per-task fetch result, recorded against the run.

    ``status`` is one of:
      - ``succeeded``: fetcher returned without raising; ``items_seen`` and
        ``items_new`` reflect what came back and what was newly stored.
      - ``failed``: fetcher raised. ``error`` carries a short reason.
      - ``skipped``: harness couldn't dispatch (no fetcher registered for
        this kind). ``error`` carries the reason. Counts as attempted in
        coverage so the run is honest about intent.
    """

    kind: str
    source_id: int | None
    source_name: str
    status: str
    items_seen: int = 0
    items_new: int = 0
    latency_ms: int = 0
    error: str | None = None


FetcherCallable = Callable[[sqlite3.Connection, ResolvedTask], Awaitable[list[FetchedItem]]]


# Module-level registry. Concrete fetcher modules register themselves at
# import time with ``FETCHER_REGISTRY[kind] = fetch_fn``. The orchestrator
# imports the desired fetcher modules and passes this dict to the runner.
# The dict is intentionally mutable — tests can substitute fakes by passing
# their own dict to :func:`run_fetch_stage` rather than mutating this one.
FETCHER_REGISTRY: dict[str, FetcherCallable] = {}
