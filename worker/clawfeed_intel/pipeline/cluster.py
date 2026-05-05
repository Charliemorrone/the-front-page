"""Dedup and clustering for the filtering lifecycle stage.

Clustering reduces the raw-item pool produced by the fetch stage into a
smaller set of *event clusters* — one cluster per logical development the
relevance filter will later judge. The architecture doc describes three
levels:

- **Level 1 — Canonical URL.** Items sharing :func:`normalize.canonicalize_url`
  collapse. Catches the same article reached through tracking-laden vs clean
  URLs. *(this module — step 7a)*
- **Level 2 — Content fingerprint.** Items sharing :func:`normalize.content_hash`
  collapse even when URLs disagree (syndicated copies). *(step 7b)*
- **Level 3 — Event clustering.** Heuristic similarity across title, entities,
  date, numeric facts. *(step 7c)*

Two-layer pattern, mirroring the fetcher modules:

- :func:`cluster_by_canonical_url` is pure — it takes raw-item rows and
  returns :class:`ClusterDraft` shapes. No DB access, fixture-testable.
- :func:`cluster_run` is the thin orchestration wrapper that loads a run's
  raw items, runs the pure pass, and persists the drafts via
  :func:`db.create_cluster`.

The relevance filter (next milestone) promotes ``status`` from ``pending`` to
``kept``/``filtered_out``. This module never sets anything other than
``pending`` — clustering is structural, not editorial.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field

from .. import db

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterDraft:
    """Pre-persistence shape produced by the pure clustering pass.

    ``cluster_key`` is the deterministic identifier the persistence layer
    feeds into ``item_clusters.cluster_key`` (subject to ``UNIQUE(run_id,
    cluster_key)``). For Level 1 every member shares the same canonical URL
    so the key is unambiguous.
    """

    cluster_key: str
    title: str
    raw_item_ids: tuple[int, ...]
    metadata: dict[str, object] = field(default_factory=dict)


def cluster_by_canonical_url(items: Iterable[sqlite3.Row]) -> list[ClusterDraft]:
    """Group raw items by their ``canonical_url`` (Level 1 dedup).

    *items* are typically rows yielded by :func:`db.iter_run_raw_items`, but
    any iterable of mappings exposing ``id``, ``canonical_url``, and
    ``title`` works — the function tolerates :class:`sqlite3.Row`, ``dict``,
    or any object with matching keys/attributes.

    Returns:
        Drafts sorted by ``cluster_key`` ascending. Within a draft,
        ``raw_item_ids`` are sorted ascending and the representative
        ``title`` comes from the smallest-id member, so re-runs against the
        same input produce the same drafts.

    Items whose ``canonical_url`` is empty after stripping fall into a single
    bucket keyed by ``""`` — fetchers always populate ``canonical_url``, so
    in practice this is the empty case. We don't drop them silently because
    that would obscure an upstream bug.
    """
    by_key: dict[str, list[tuple[int, str]]] = {}
    for item in items:
        canonical = (_get(item, "canonical_url") or "").strip()
        rid = int(_get(item, "id"))
        title = (_get(item, "title") or "").strip()
        by_key.setdefault(canonical, []).append((rid, title))

    drafts: list[ClusterDraft] = []
    for key in sorted(by_key):
        members = sorted(by_key[key], key=lambda pair: pair[0])
        ids = tuple(rid for rid, _ in members)
        # Representative title is the smallest-id member's; if blank, walk
        # forward to find the first non-blank title. A cluster with all
        # blank titles keeps "" — the relevance/summary stages will revisit.
        title = next((t for _, t in members if t), "")
        drafts.append(ClusterDraft(cluster_key=key, title=title, raw_item_ids=ids))
    return drafts


def cluster_run(conn: sqlite3.Connection, run_id: int) -> int:
    """Build Level 1 clusters for *run_id* and persist them.

    Idempotent: re-calling for the same run is safe — existing clusters keep
    their status (which the relevance filter may have already promoted) and
    new members are appended via ``INSERT OR IGNORE INTO cluster_items``.

    Returns the count of clusters now associated with the run (including
    pre-existing ones — Coverage uses this as the structural "how many
    distinct events did we observe" number).
    """
    items = list(db.iter_run_raw_items(conn, run_id))
    drafts = cluster_by_canonical_url(items)
    for draft in drafts:
        db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key=draft.cluster_key,
            title=draft.title,
            raw_item_ids=draft.raw_item_ids,
        )
    log.info(
        "run %d: clustered %d raw items into %d Level-1 clusters",
        run_id,
        len(items),
        len(drafts),
    )
    return len(drafts)


def _get(item: object, key: str) -> object:
    """Tolerant accessor for sqlite3.Row, dict, or attribute-style objects."""
    if isinstance(item, sqlite3.Row):
        return item[key] if key in item.keys() else None
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)
