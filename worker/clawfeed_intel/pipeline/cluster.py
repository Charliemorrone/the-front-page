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
  copies — same body, different URLs. *(step 7b)*
- **Level 3 — Event clustering (heuristic).** L2 drafts whose titles overlap
  enough, fall within a date window, and don't carry contradicting numeric
  facts fold together. Catches the "same event, independently written up by
  multiple outlets" case where neither URL nor body fingerprint matches.
  *(this module — step 7c)*

LLM-based tie-breaking on ambiguous L3 pairs is deferred to a future step
(7d, optional, gated on vMLX availability).

Two-layer pattern, mirroring the fetcher modules:

- :func:`cluster_by_canonical_url`, :func:`fold_by_content_hash`, and
  :func:`fold_by_event_similarity` are pure — they take rows / drafts and
  return :class:`ClusterDraft` shapes. No DB access, fixture-testable.
- :func:`cluster_run` is the thin orchestration wrapper that loads a run's
  raw items, chains L1 → L2 → L3, and persists the final drafts via
  :func:`db.create_cluster`.

The relevance filter (next milestone) promotes ``status`` from ``pending`` to
``kept``/``filtered_out``. This module never sets anything other than
``pending`` — clustering is structural, not editorial.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from .. import db

log = logging.getLogger(__name__)


# ── ClusterDraft ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClusterDraft:
    """Pre-persistence shape produced by the pure clustering pass.

    ``cluster_key`` is the deterministic identifier the persistence layer
    feeds into ``item_clusters.cluster_key`` (subject to ``UNIQUE(run_id,
    cluster_key)``). For Level 1 it is the members' shared canonical URL;
    after Level 2 or Level 3 folds drafts together it becomes the smallest
    of their cluster_keys, per the architecture doc's "smallest
    canonical_url among members" rule.

    ``representative_content_hash`` is the smallest-id member's content
    hash. ``None`` means the rep had no hash — those drafts cannot
    participate in L2 folding.

    ``published_at`` is the smallest-id member's published_at (falling
    through to the first non-None among members), used by L3's date-window
    check. ``None`` means no member carried a timestamp; L3 falls back to
    title similarity alone for those drafts.
    """

    cluster_key: str
    title: str
    raw_item_ids: tuple[int, ...]
    representative_content_hash: str | None = None
    published_at: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


# ── Level 1: canonical URL ────────────────────────────────────────────────────


def cluster_by_canonical_url(items: Iterable[sqlite3.Row]) -> list[ClusterDraft]:
    """Group raw items by their ``canonical_url`` (Level 1 dedup).

    *items* are typically rows yielded by :func:`db.iter_run_raw_items`, but
    any iterable of mappings exposing ``id``, ``canonical_url``, ``title``,
    and (optionally) ``content_hash``, ``published_at`` works — the function
    tolerates :class:`sqlite3.Row`, ``dict``, or any object with matching
    keys.

    Returns:
        Drafts sorted by ``cluster_key`` ascending. Within a draft,
        ``raw_item_ids`` are sorted ascending; the representative ``title``
        and ``representative_content_hash`` come from the smallest-id
        member. ``title`` and ``published_at`` fall through blank/None
        entries to the first non-blank/non-None;
        ``representative_content_hash`` does not — using a non-rep member's
        hash would make L2 folding non-deterministic across runs that see
        the same items in different orders.

    Items whose ``canonical_url`` is empty after stripping fall into a
    single bucket keyed by ``""``. Fetchers always populate
    ``canonical_url``, so this is a defensive surface; keeping the items
    visible in coverage makes upstream bugs spottable.
    """
    by_key: dict[str, list[tuple[int, str, str | None, str | None]]] = {}
    for item in items:
        canonical = (_get(item, "canonical_url") or "").strip()
        rid = int(_get(item, "id"))
        title = (_get(item, "title") or "").strip()
        rep_hash = _get(item, "content_hash")
        published_at = _get(item, "published_at")
        by_key.setdefault(canonical, []).append((rid, title, rep_hash, published_at))

    drafts: list[ClusterDraft] = []
    for key in sorted(by_key):
        members = sorted(by_key[key], key=lambda quad: quad[0])
        ids = tuple(rid for rid, _, _, _ in members)
        title = next((t for _, t, _, _ in members if t), "")
        rep_hash = members[0][2] or None
        if isinstance(rep_hash, str) and not rep_hash.strip():
            rep_hash = None
        published_at = next((p for _, _, _, p in members if p), None)
        drafts.append(
            ClusterDraft(
                cluster_key=key,
                title=title,
                raw_item_ids=ids,
                representative_content_hash=rep_hash,
                published_at=published_at,
            )
        )
    return drafts


# ── Level 2: content fingerprint ──────────────────────────────────────────────


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
    - ``published_at`` is the earliest non-None among merged drafts (news
      typically propagates outward in time, so the earliest sighting is
      most likely the original publication).
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
        merged.append(_merge_group(group, rep_hash=rep_hash))

    out = merged + untouched
    out.sort(key=lambda d: d.cluster_key)
    return out


# ── Level 3: heuristic event similarity ───────────────────────────────────────


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the",
        "is", "are", "was", "were", "be", "been", "being",
        "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "by",
        "from", "as", "into", "via", "than", "then", "so",
        "it", "its", "this", "that", "these", "those",
        "has", "have", "had", "will", "would", "can", "could", "should",
        "may", "might", "must", "do", "does", "did", "not", "no",
        "i", "you", "he", "she", "we", "they", "him", "her", "us", "them",
        "his", "hers", "ours", "theirs", "my", "your", "our", "their",
    }
)  # fmt: skip

# Wire-service / aggregator boilerplate that adds noise but no event signal.
_BOILERPLATE_TOKENS: frozenset[str] = frozenset(
    {"breaking", "watch", "live", "update", "exclusive", "report"}
)

# Tokens contain ASCII letters/digits/$ and may include hyphens or apostrophes
# inside (so "co-founder" survives as one token, "company's" as one).
_TOKEN_PATTERN = re.compile(r"[a-z0-9$]+(?:[-'’][a-z0-9]+)*")


def fold_by_event_similarity(
    drafts: Iterable[ClusterDraft],
    *,
    title_threshold: float = 0.65,
    date_window_hours: int = 48,
    min_effective_tokens: int = 3,
) -> list[ClusterDraft]:
    """Heuristic Level 3 fold: merge drafts whose titles describe the same event.

    Two drafts merge iff *all* of the following hold:

    1. Both have at least ``min_effective_tokens`` non-stopword tokens in
       their titles. Very short titles ("Update", "Watch this") would fold
       too eagerly.
    2. Their token Jaccard similarity is ``>= title_threshold``.
    3. If both carry numeric tokens (digit-bearing — "$1b", "5b", "2026")
       the sets must overlap. Two stories with different funding amounts
       are different events even if surrounding tokens look similar.
    4. If both carry parseable ``published_at``, the timestamps fall within
       ``date_window_hours``. Outside the window → different events. Either
       missing → no constraint (we don't penalize fetchers that omit
       timestamps).

    The pass uses union-find with smaller-index-as-root for transitivity:
    if A↔B and B↔C both clear the bar, A B C all merge into one cluster.
    Pair scoring is symmetric and order-independent, so re-runs with the
    same drafts produce the same merged set regardless of input order.

    Merged-draft fields follow the same rules as L2:

    - ``cluster_key`` = lexicographically smallest of the source keys.
    - ``title`` = title of the draft whose first member id is smallest
      overall (falling through blanks).
    - ``raw_item_ids`` = sorted union.
    - ``published_at`` = earliest non-None member timestamp.
    - ``representative_content_hash`` = first non-None among drafts sorted
      by smallest member id.

    Output is sorted by ``cluster_key`` for stable downstream persistence.

    The default threshold (0.65) is conservative — false positives in L3
    propagate into the relevance filter and final brief, where they would
    silence real evidence under a misleading headline. Tune via
    `title_threshold` if real-run analysis demands.
    """
    drafts_list = list(drafts)
    n = len(drafts_list)
    if n <= 1:
        return list(drafts_list)

    tokens_per_draft = [_title_tokens(d.title) for d in drafts_list]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        parent[max(rx, ry)] = min(rx, ry)

    for i in range(n):
        ti = tokens_per_draft[i]
        if len(ti) < min_effective_tokens:
            continue
        for j in range(i + 1, n):
            tj = tokens_per_draft[j]
            if len(tj) < min_effective_tokens:
                continue
            if _date_disqualifies(
                drafts_list[i].published_at,
                drafts_list[j].published_at,
                date_window_hours,
            ):
                continue
            if _numeric_disqualifies(ti, tj):
                continue
            if _jaccard(ti, tj) >= title_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[ClusterDraft] = []
    for indices in groups.values():
        if len(indices) == 1:
            merged.append(drafts_list[indices[0]])
            continue
        merged.append(_merge_group([drafts_list[idx] for idx in indices]))

    merged.sort(key=lambda d: d.cluster_key)
    return merged


def _title_tokens(title: str) -> frozenset[str]:
    if not title:
        return frozenset()
    raw = _TOKEN_PATTERN.findall(title.lower())
    return frozenset(t for t in raw if t not in _STOPWORDS and t not in _BOILERPLATE_TOKENS)


def _numeric_tokens(tokens: frozenset[str]) -> frozenset[str]:
    return frozenset(t for t in tokens if any(c.isdigit() for c in t))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _numeric_disqualifies(a: frozenset[str], b: frozenset[str]) -> bool:
    """True iff both titles carry numeric tokens AND those sets don't overlap.

    Two stories that both quote "$500M" and "$1B" with no overlap are about
    different financial events. If only one side has numerics, we can't
    judge — leave the call to title similarity.
    """
    nums_a = _numeric_tokens(a)
    nums_b = _numeric_tokens(b)
    if not nums_a or not nums_b:
        return False
    return not (nums_a & nums_b)


def _date_disqualifies(a: str | None, b: str | None, window_hours: int) -> bool:
    """True iff both timestamps parse and fall MORE than window_hours apart.

    Permissive on missing/unparseable timestamps — we'd rather rely on
    title similarity than over-disqualify legitimate matches whose
    fetchers happened to omit a date.
    """
    if a is None or b is None:
        return False
    try:
        da = datetime.fromisoformat(a)
        db_ = datetime.fromisoformat(b)
    except (TypeError, ValueError):
        return False
    if da.tzinfo is None or db_.tzinfo is None:
        return False
    delta = abs((da - db_).total_seconds()) / 3600.0
    return delta > window_hours


# ── Shared merge helper (used by L2 and L3) ──────────────────────────────────


def _merge_group(group: list[ClusterDraft], *, rep_hash: str | None = None) -> ClusterDraft:
    """Merge a group of drafts using the standard tie-break rules.

    - ``cluster_key`` = lex-smallest of group's keys.
    - ``title`` = draft with smallest first-member-id (fall through blanks).
    - ``raw_item_ids`` = sorted union of all members.
    - ``published_at`` = earliest non-None timestamp across the group.
    - ``representative_content_hash`` = caller-supplied (L2 knows the shared
      hash) or first non-None among drafts sorted by smallest member id.
    """
    sorted_drafts = sorted(group, key=lambda d: d.raw_item_ids[0])
    all_ids: set[int] = set()
    for draft in group:
        all_ids.update(draft.raw_item_ids)
    smallest_key = min(d.cluster_key for d in group)
    title = sorted_drafts[0].title
    if not title:
        for d in sorted_drafts:
            if d.title:
                title = d.title
                break
    timestamps = [d.published_at for d in group if d.published_at]
    published_at = min(timestamps) if timestamps else None
    if rep_hash is None:
        rep_hash = next(
            (d.representative_content_hash for d in sorted_drafts if d.representative_content_hash),
            None,
        )
    return ClusterDraft(
        cluster_key=smallest_key,
        title=title,
        raw_item_ids=tuple(sorted(all_ids)),
        representative_content_hash=rep_hash,
        published_at=published_at,
    )


# ── Orchestration ─────────────────────────────────────────────────────────────


def cluster_run(conn: sqlite3.Connection, run_id: int) -> int:
    """Build Level 1 + Level 2 + Level 3 clusters for *run_id* and persist them.

    Idempotent: re-calling for the same run is safe — existing clusters keep
    their status (which the relevance filter may have already promoted) and
    new members are appended via ``INSERT OR IGNORE INTO cluster_items``.

    Returns the count of clusters now associated with the run (post-fold).
    """
    items = list(db.iter_run_raw_items(conn, run_id))
    drafts = fold_by_event_similarity(fold_by_content_hash(cluster_by_canonical_url(items)))
    for draft in drafts:
        db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key=draft.cluster_key,
            title=draft.title,
            raw_item_ids=draft.raw_item_ids,
        )
    log.info(
        "run %d: clustered %d raw items into %d clusters (L1+L2+L3)",
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
