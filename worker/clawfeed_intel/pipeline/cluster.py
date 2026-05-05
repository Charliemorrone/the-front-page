"""Dedup and clustering for the filtering lifecycle stage.

Clustering reduces the raw-item pool produced by the fetch stage into a
smaller set of *event clusters* — one cluster per logical development the
relevance filter will later judge. The architecture doc describes three
levels:

- **Level 1 — Canonical URL.** Items sharing :func:`normalize.canonicalize_url`
  collapse. Catches the same article reached through tracking-laden vs clean
  URLs. *(step 7a)*
- **Level 2 — Content fingerprint.** L1 drafts whose representative items
  share :func:`normalize.content_hash` fold together. Catches syndicated
  copies — same body, different URLs. *(this module — step 7b)*
- **Level 3 — Event clustering.** Heuristic similarity across title,
  entities, date, numeric facts. *(step 7c)*

Two-layer pattern, mirroring the fetcher modules:

- :func:`cluster_by_canonical_url` and :func:`fold_by_content_hash` are pure —
  they take rows / drafts and return :class:`ClusterDraft` shapes. No DB
  access, fixture-testable.
- :func:`cluster_run` is the thin orchestration wrapper that loads a run's
  raw items, chains L1 → L2, and persists the final drafts via
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
    cluster_key)``). For Level 1 it is the members' shared canonical URL;
    after Level 2 folds two L1 drafts together it becomes the smaller of
    their cluster_keys, per the architecture doc's "smallest canonical_url
    among members" rule.

    ``representative_content_hash`` is the smallest-id member's content
    hash, kept on the draft so the L2 fold can run without re-reading the
    underlying rows. ``None`` means the rep had no hash — those drafts
    cannot participate in L2 folding (we'd risk merging unrelated items).
    """

    cluster_key: str
    title: str
    raw_item_ids: tuple[int, ...]
    representative_content_hash: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def cluster_by_canonical_url(items: Iterable[sqlite3.Row]) -> list[ClusterDraft]:
    """Group raw items by their ``canonical_url`` (Level 1 dedup).

    *items* are typically rows yielded by :func:`db.iter_run_raw_items`, but
    any iterable of mappings exposing ``id``, ``canonical_url``, ``title``,
    and (optionally) ``content_hash`` works — the function tolerates
    :class:`sqlite3.Row`, ``dict``, or any object with matching keys.

    Returns:
        Drafts sorted by ``cluster_key`` ascending. Within a draft,
        ``raw_item_ids`` are sorted ascending; the representative ``title``
        and ``representative_content_hash`` come from the smallest-id
        member, so re-runs against the same input produce the same drafts.
        ``title`` falls through blank entries to the first non-blank;
        ``representative_content_hash`` does not — using a non-rep member's
        hash would make L2 folding non-deterministic across runs that see
        the same items in different orders.

    Items whose ``canonical_url`` is empty after stripping fall into a
    single bucket keyed by ``""``. Fetchers always populate
    ``canonical_url``, so this is a defensive surface; keeping the items
    visible in coverage makes upstream bugs spottable.
    """
    by_key: dict[str, list[tuple[int, str, str | None]]] = {}
    for item in items:
        canonical = (_get(item, "canonical_url") or "").strip()
        rid = int(_get(item, "id"))
        title = (_get(item, "title") or "").strip()
        rep_hash = _get(item, "content_hash")
        by_key.setdefault(canonical, []).append((rid, title, rep_hash))

    drafts: list[ClusterDraft] = []
    for key in sorted(by_key):
        members = sorted(by_key[key], key=lambda triple: triple[0])
        ids = tuple(rid for rid, _, _ in members)
        title = next((t for _, t, _ in members if t), "")
        rep_hash = members[0][2] or None
        if isinstance(rep_hash, str) and not rep_hash.strip():
            rep_hash = None
        drafts.append(
            ClusterDraft(
                cluster_key=key,
                title=title,
                raw_item_ids=ids,
                representative_content_hash=rep_hash,
            )
        )
    return drafts


def fold_by_content_hash(drafts: Iterable[ClusterDraft]) -> list[ClusterDraft]:
    """Fold L1 drafts whose representative items share a ``content_hash``.

    This is the Level 2 dedup pass. Two drafts that reach a fetcher via
    different canonical URLs but carry the same normalized title + body
    fingerprint are syndicated copies of one event, so they collapse into a
    single cluster.

    Folding rules:

    - Drafts with ``representative_content_hash is None`` (or empty after
      strip) never fold. We don't risk merging drafts that simply lack a
      fingerprint.
    - Among drafts that share a hash, the merged cluster's ``cluster_key``
      is the lexicographically smallest of the input keys (the
      "smallest canonical_url among members" rule from the architecture
      doc — a no-op for L1, load-bearing here).
    - The merged ``title`` comes from the draft whose first member id is
      smallest overall — that is the same item that would have been
      "representative" if all members were considered together at L1.
    - ``raw_item_ids`` is the sorted union of the merged drafts' members.
    - The representative content hash is preserved.

    Drafts that don't share a hash with any other draft pass through
    unchanged. The returned list is sorted by ``cluster_key`` for
    deterministic persistence ordering.
    """
    drafts_list = list(drafts)
    by_hash: dict[str, list[ClusterDraft]] = {}
    untouched: list[ClusterDraft] = []
    for draft in drafts_list:
        rep = draft.representative_content_hash
        if rep is None or not rep.strip():
            untouched.append(draft)
            continue
        by_hash.setdefault(rep, []).append(draft)

    merged: list[ClusterDraft] = []
    for rep_hash, group in by_hash.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        all_ids: set[int] = set()
        for draft in group:
            all_ids.update(draft.raw_item_ids)
        smallest_key = min(d.cluster_key for d in group)
        # Title: pick from the draft whose smallest member id is overall
        # smallest — that's the would-be representative if all members
        # were grouped at L1. Falls through blanks like cluster_by_canonical_url.
        chosen = min(group, key=lambda d: d.raw_item_ids[0])
        title = chosen.title
        if not title:
            for d in sorted(group, key=lambda d: d.raw_item_ids[0]):
                if d.title:
                    title = d.title
                    break
        merged.append(
            ClusterDraft(
                cluster_key=smallest_key,
                title=title,
                raw_item_ids=tuple(sorted(all_ids)),
                representative_content_hash=rep_hash,
            )
        )

    out = merged + untouched
    out.sort(key=lambda d: d.cluster_key)
    return out


def cluster_run(conn: sqlite3.Connection, run_id: int) -> int:
    """Build Level 1 + Level 2 clusters for *run_id* and persist them.

    Idempotent: re-calling for the same run is safe — existing clusters keep
    their status (which the relevance filter may have already promoted) and
    new members are appended via ``INSERT OR IGNORE INTO cluster_items``.

    Returns the count of clusters now associated with the run (post-fold).
    """
    items = list(db.iter_run_raw_items(conn, run_id))
    drafts = fold_by_content_hash(cluster_by_canonical_url(items))
    for draft in drafts:
        db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key=draft.cluster_key,
            title=draft.title,
            raw_item_ids=draft.raw_item_ids,
        )
    log.info(
        "run %d: clustered %d raw items into %d clusters (L1+L2)",
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
