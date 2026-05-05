"""Tests for the Level 1 (canonical URL) clustering pass.

The pure function is exercised against hand-built dicts so the cluster
arithmetic can be validated without any DB or fetcher noise. The
orchestration helper is exercised against a real ``temp_db`` so we cover
the persistence + idempotency contract that the next milestone (relevance
filter) depends on.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from clawfeed_intel import db as worker_db
from clawfeed_intel.pipeline.cluster import (
    ClusterDraft,
    cluster_by_canonical_url,
    cluster_run,
    fold_by_content_hash,
    fold_by_event_similarity,
)


# ── pure: cluster_by_canonical_url ────────────────────────────────────────────


def _row(
    rid: int,
    canonical: str,
    title: str = "",
    content_hash: str | None = None,
) -> dict[str, object]:
    return {
        "id": rid,
        "canonical_url": canonical,
        "title": title,
        "content_hash": content_hash,
    }


def test_pure_empty_input_returns_empty_list():
    assert cluster_by_canonical_url([]) == []


def test_pure_single_item_produces_single_draft():
    drafts = cluster_by_canonical_url([_row(1, "https://example.com/a", "Alpha")])
    assert drafts == [
        ClusterDraft(
            cluster_key="https://example.com/a",
            title="Alpha",
            raw_item_ids=(1,),
        )
    ]


def test_pure_two_items_same_canonical_url_collapse():
    drafts = cluster_by_canonical_url(
        [
            _row(1, "https://example.com/a", "From RSS"),
            _row(2, "https://example.com/a", "From HN"),
        ]
    )
    assert len(drafts) == 1
    assert drafts[0].cluster_key == "https://example.com/a"
    assert drafts[0].raw_item_ids == (1, 2)


def test_pure_two_items_different_urls_make_two_drafts():
    drafts = cluster_by_canonical_url(
        [
            _row(1, "https://example.com/a", "Alpha"),
            _row(2, "https://example.com/b", "Beta"),
        ]
    )
    assert [d.cluster_key for d in drafts] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert [d.raw_item_ids for d in drafts] == [(1,), (2,)]


def test_pure_drafts_are_sorted_by_cluster_key():
    """Sorted output is what makes re-runs deterministic. Without sorting
    the order would depend on dict insertion order, which depends on input
    order, which is not a contract callers should have to think about."""
    drafts = cluster_by_canonical_url(
        [
            _row(1, "https://example.com/z"),
            _row(2, "https://example.com/a"),
            _row(3, "https://example.com/m"),
        ]
    )
    assert [d.cluster_key for d in drafts] == [
        "https://example.com/a",
        "https://example.com/m",
        "https://example.com/z",
    ]


def test_pure_raw_item_ids_within_a_draft_are_sorted_ascending():
    drafts = cluster_by_canonical_url(
        [
            _row(7, "https://example.com/a"),
            _row(2, "https://example.com/a"),
            _row(5, "https://example.com/a"),
        ]
    )
    assert drafts[0].raw_item_ids == (2, 5, 7)


def test_pure_title_comes_from_smallest_id_member():
    """The representative title is the first-seen member's. Stable across
    re-runs because input order doesn't matter — we sort by id."""
    drafts = cluster_by_canonical_url(
        [
            _row(9, "https://example.com/a", "Late title"),
            _row(3, "https://example.com/a", "First title"),
            _row(7, "https://example.com/a", "Mid title"),
        ]
    )
    assert drafts[0].title == "First title"


def test_pure_title_falls_through_blanks_to_first_non_blank():
    drafts = cluster_by_canonical_url(
        [
            _row(1, "https://example.com/a", ""),
            _row(2, "https://example.com/a", ""),
            _row(3, "https://example.com/a", "Recovered"),
        ]
    )
    assert drafts[0].title == "Recovered"


def test_pure_all_blank_titles_keeps_empty_string():
    drafts = cluster_by_canonical_url(
        [
            _row(1, "https://example.com/a", ""),
            _row(2, "https://example.com/a", ""),
        ]
    )
    assert drafts[0].title == ""


def test_pure_empty_canonical_url_groups_into_single_bucket():
    """Empty canonical_url is an upstream bug (every fetcher should produce
    one). We surface it as a single cluster with key="" rather than dropping
    silently — easier to spot in coverage when something's wrong."""
    drafts = cluster_by_canonical_url(
        [
            _row(1, "", "untitled-a"),
            _row(2, "", "untitled-b"),
        ]
    )
    assert len(drafts) == 1
    assert drafts[0].cluster_key == ""
    assert drafts[0].raw_item_ids == (1, 2)


def test_pure_whitespace_canonical_url_treated_as_empty():
    drafts = cluster_by_canonical_url(
        [
            _row(1, "  ", "a"),
            _row(2, "https://example.com/x", "x"),
        ]
    )
    assert {d.cluster_key for d in drafts} == {"", "https://example.com/x"}


def test_pure_accepts_sqlite_row_objects():
    """sqlite3.Row uses key-style access; the pure function must support it
    directly so callers don't need to materialize dicts."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    rows = list(
        conn.execute(
            """
            SELECT 1 AS id, 'https://example.com/a' AS canonical_url, 'Alpha' AS title
            UNION ALL
            SELECT 2, 'https://example.com/a', 'Alpha-two'
            UNION ALL
            SELECT 3, 'https://example.com/b', 'Beta'
            """
        )
    )
    drafts = cluster_by_canonical_url(rows)
    assert [(d.cluster_key, d.raw_item_ids) for d in drafts] == [
        ("https://example.com/a", (1, 2)),
        ("https://example.com/b", (3,)),
    ]


# ── pure: representative_content_hash on L1 drafts ────────────────────────────


def test_pure_l1_records_smallest_id_member_content_hash():
    drafts = cluster_by_canonical_url(
        [
            _row(7, "https://example.com/a", "Late", content_hash="hash-late"),
            _row(3, "https://example.com/a", "First", content_hash="hash-first"),
            _row(5, "https://example.com/a", "Mid", content_hash="hash-mid"),
        ]
    )
    assert drafts[0].representative_content_hash == "hash-first"


def test_pure_l1_no_content_hash_yields_none():
    drafts = cluster_by_canonical_url([_row(1, "https://example.com/a", "Alpha")])
    assert drafts[0].representative_content_hash is None


def test_pure_l1_blank_content_hash_normalized_to_none():
    """Whitespace-only hashes are upstream noise, not real fingerprints; we
    normalize them to None so they can't accidentally match each other in
    a later L2 fold."""
    drafts = cluster_by_canonical_url(
        [_row(1, "https://example.com/a", "Alpha", content_hash="   ")]
    )
    assert drafts[0].representative_content_hash is None


# ── pure: fold_by_content_hash ────────────────────────────────────────────────


def test_l2_empty_input_returns_empty_list():
    assert fold_by_content_hash([]) == []


def test_l2_single_draft_passes_through_unchanged():
    drafts = [
        ClusterDraft(
            cluster_key="https://example.com/a",
            title="Alpha",
            raw_item_ids=(1,),
            representative_content_hash="hash-a",
        )
    ]
    assert fold_by_content_hash(drafts) == drafts


def test_l2_distinct_hashes_pass_through_unchanged():
    drafts = [
        ClusterDraft(
            cluster_key="https://example.com/a",
            title="Alpha",
            raw_item_ids=(1,),
            representative_content_hash="hash-a",
        ),
        ClusterDraft(
            cluster_key="https://example.com/b",
            title="Beta",
            raw_item_ids=(2,),
            representative_content_hash="hash-b",
        ),
    ]
    folded = fold_by_content_hash(drafts)
    assert {d.cluster_key for d in folded} == {
        "https://example.com/a",
        "https://example.com/b",
    }
    assert sum(len(d.raw_item_ids) for d in folded) == 2


def test_l2_two_drafts_share_hash_fold_into_one():
    """Syndicated case: two URLs, same body. Merged cluster takes the
    smaller cluster_key (lex order on canonical URL) per the architecture
    doc's "smallest canonical_url among members" rule."""
    drafts = [
        ClusterDraft(
            cluster_key="https://yahoo.com/article",
            title="From Yahoo",
            raw_item_ids=(2,),
            representative_content_hash="hash-syndicated",
        ),
        ClusterDraft(
            cluster_key="https://example.com/article",
            title="From Original",
            raw_item_ids=(1,),
            representative_content_hash="hash-syndicated",
        ),
    ]
    folded = fold_by_content_hash(drafts)
    assert len(folded) == 1
    assert folded[0].cluster_key == "https://example.com/article"
    # Title from the smallest-overall-id member (id=1 — "From Original")
    assert folded[0].title == "From Original"
    assert folded[0].raw_item_ids == (1, 2)
    assert folded[0].representative_content_hash == "hash-syndicated"


def test_l2_three_drafts_share_hash_fold_transitively():
    drafts = [
        ClusterDraft(
            cluster_key="https://b.example.com/x",
            title="B",
            raw_item_ids=(5,),
            representative_content_hash="h",
        ),
        ClusterDraft(
            cluster_key="https://a.example.com/x",
            title="A",
            raw_item_ids=(2,),
            representative_content_hash="h",
        ),
        ClusterDraft(
            cluster_key="https://c.example.com/x",
            title="C",
            raw_item_ids=(8,),
            representative_content_hash="h",
        ),
    ]
    folded = fold_by_content_hash(drafts)
    assert len(folded) == 1
    assert folded[0].cluster_key == "https://a.example.com/x"
    assert folded[0].title == "A"
    assert folded[0].raw_item_ids == (2, 5, 8)


def test_l2_partial_fold_leaves_unmatched_drafts_alone():
    drafts = [
        ClusterDraft(
            cluster_key="https://a.example.com/x",
            title="A",
            raw_item_ids=(1,),
            representative_content_hash="shared",
        ),
        ClusterDraft(
            cluster_key="https://b.example.com/x",
            title="B",
            raw_item_ids=(2,),
            representative_content_hash="shared",
        ),
        ClusterDraft(
            cluster_key="https://c.example.com/x",
            title="C",
            raw_item_ids=(3,),
            representative_content_hash="distinct",
        ),
    ]
    folded = fold_by_content_hash(drafts)
    keys = sorted(d.cluster_key for d in folded)
    assert keys == ["https://a.example.com/x", "https://c.example.com/x"]
    a_draft = next(d for d in folded if d.cluster_key == "https://a.example.com/x")
    assert a_draft.raw_item_ids == (1, 2)
    c_draft = next(d for d in folded if d.cluster_key == "https://c.example.com/x")
    assert c_draft.raw_item_ids == (3,)


def test_l2_drafts_with_none_hash_never_fold():
    """Two drafts that both lack a content_hash must NOT fold — we'd be
    merging based on absence, which would conflate unrelated items."""
    drafts = [
        ClusterDraft(
            cluster_key="https://a.example.com/x",
            title="A",
            raw_item_ids=(1,),
            representative_content_hash=None,
        ),
        ClusterDraft(
            cluster_key="https://b.example.com/x",
            title="B",
            raw_item_ids=(2,),
            representative_content_hash=None,
        ),
    ]
    folded = fold_by_content_hash(drafts)
    assert len(folded) == 2


def test_l2_empty_string_hash_treated_as_none():
    """Same defensive rule: an empty-string hash is upstream noise, not a
    real match."""
    drafts = [
        ClusterDraft(
            cluster_key="https://a.example.com/x",
            title="A",
            raw_item_ids=(1,),
            representative_content_hash="",
        ),
        ClusterDraft(
            cluster_key="https://b.example.com/x",
            title="B",
            raw_item_ids=(2,),
            representative_content_hash="",
        ),
    ]
    folded = fold_by_content_hash(drafts)
    assert len(folded) == 2


def test_l2_mixed_hashed_and_unhashed_drafts():
    drafts = [
        ClusterDraft(
            cluster_key="https://a.example.com/x",
            title="A",
            raw_item_ids=(1,),
            representative_content_hash="h",
        ),
        ClusterDraft(
            cluster_key="https://b.example.com/x",
            title="B",
            raw_item_ids=(2,),
            representative_content_hash=None,
        ),
        ClusterDraft(
            cluster_key="https://c.example.com/x",
            title="C",
            raw_item_ids=(3,),
            representative_content_hash="h",
        ),
    ]
    folded = fold_by_content_hash(drafts)
    assert len(folded) == 2
    a_draft = next(d for d in folded if d.cluster_key == "https://a.example.com/x")
    assert a_draft.raw_item_ids == (1, 3)
    b_draft = next(d for d in folded if d.cluster_key == "https://b.example.com/x")
    assert b_draft.raw_item_ids == (2,)


def test_l2_output_sorted_by_cluster_key():
    drafts = [
        ClusterDraft(
            cluster_key="https://z.example.com/x",
            title="Z",
            raw_item_ids=(1,),
            representative_content_hash="h1",
        ),
        ClusterDraft(
            cluster_key="https://a.example.com/x",
            title="A",
            raw_item_ids=(2,),
            representative_content_hash="h2",
        ),
        ClusterDraft(
            cluster_key="https://m.example.com/x",
            title="M",
            raw_item_ids=(3,),
            representative_content_hash="h3",
        ),
    ]
    folded = fold_by_content_hash(drafts)
    assert [d.cluster_key for d in folded] == [
        "https://a.example.com/x",
        "https://m.example.com/x",
        "https://z.example.com/x",
    ]


def test_l2_merged_title_falls_through_blank():
    """If the smallest-id draft's title is blank, the merged title walks
    forward to the next draft (in id order) with a non-blank title."""
    drafts = [
        ClusterDraft(
            cluster_key="https://b.example.com/x",
            title="Recovered",
            raw_item_ids=(5,),
            representative_content_hash="h",
        ),
        ClusterDraft(
            cluster_key="https://a.example.com/x",
            title="",
            raw_item_ids=(2,),
            representative_content_hash="h",
        ),
    ]
    folded = fold_by_content_hash(drafts)
    assert len(folded) == 1
    assert folded[0].title == "Recovered"
    # cluster_key still goes to the smallest URL even if its title was blank
    assert folded[0].cluster_key == "https://a.example.com/x"


def test_l2_idempotent_when_run_twice():
    """Calling fold_by_content_hash on its own output must produce the same
    list — once folded, drafts are stable."""
    drafts = [
        ClusterDraft(
            cluster_key="https://a.example.com/x",
            title="A",
            raw_item_ids=(1,),
            representative_content_hash="h",
        ),
        ClusterDraft(
            cluster_key="https://b.example.com/x",
            title="B",
            raw_item_ids=(2,),
            representative_content_hash="h",
        ),
    ]
    once = fold_by_content_hash(drafts)
    twice = fold_by_content_hash(once)
    assert once == twice


def test_l2_propagates_earliest_published_at():
    """When two drafts fold via shared content_hash, the merged
    published_at is the earliest non-None — news propagates over time so
    the earliest sighting is the best proxy for original publication."""
    drafts = [
        ClusterDraft(
            cluster_key="https://b.example.com/x",
            title="Late",
            raw_item_ids=(5,),
            representative_content_hash="h",
            published_at="2026-05-04T18:00:00+00:00",
        ),
        ClusterDraft(
            cluster_key="https://a.example.com/x",
            title="Early",
            raw_item_ids=(2,),
            representative_content_hash="h",
            published_at="2026-05-04T09:00:00+00:00",
        ),
    ]
    folded = fold_by_content_hash(drafts)
    assert len(folded) == 1
    assert folded[0].published_at == "2026-05-04T09:00:00+00:00"


# ── pure: fold_by_event_similarity (Level 3) ──────────────────────────────────


def _draft(
    cluster_key: str,
    title: str,
    rid: int,
    *,
    published_at: str | None = None,
    rep_hash: str | None = None,
) -> ClusterDraft:
    return ClusterDraft(
        cluster_key=cluster_key,
        title=title,
        raw_item_ids=(rid,),
        representative_content_hash=rep_hash,
        published_at=published_at,
    )


def test_l3_empty_input_returns_empty():
    assert fold_by_event_similarity([]) == []


def test_l3_single_draft_passes_through():
    drafts = [_draft("https://x/a", "Anthropic raises Series F funding round", 1)]
    assert fold_by_event_similarity(drafts) == drafts


def test_l3_high_jaccard_titles_fold():
    """Two outlets cover the same event with overlapping headlines —
    Jaccard above the default 0.65 threshold → merge."""
    drafts = [
        _draft(
            "https://outlet-a.example/anthropic",
            "Anthropic raises Series F funding round",
            1,
            published_at="2026-05-04T10:00:00+00:00",
        ),
        _draft(
            "https://outlet-b.example/anthropic",
            "Anthropic raises Series F funding round today",
            2,
            published_at="2026-05-04T11:00:00+00:00",
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 1
    assert folded[0].cluster_key == "https://outlet-a.example/anthropic"
    assert folded[0].raw_item_ids == (1, 2)
    assert folded[0].title == "Anthropic raises Series F funding round"
    assert folded[0].published_at == "2026-05-04T10:00:00+00:00"


def test_l3_low_jaccard_titles_stay_distinct():
    drafts = [
        _draft("https://x/a", "Apple announces new iPhone model", 1),
        _draft("https://x/b", "Stock market closes higher today", 2),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 2


def test_l3_short_titles_skip_l3():
    """Titles below ``min_effective_tokens`` are too sparse for reliable
    Jaccard — they pass through unchanged regardless of overlap."""
    drafts = [
        _draft("https://x/a", "Update", 1),
        _draft("https://x/b", "Update", 2),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 2


def test_l3_stopwords_and_boilerplate_dropped_from_tokens():
    """Pure boilerplate ('BREAKING' / 'WATCH') doesn't contribute to
    similarity. Two stories with only the prefix in common stay distinct."""
    drafts = [
        _draft("https://x/a", "BREAKING the New York mayor resigns suddenly", 1),
        _draft("https://x/b", "BREAKING the Los Angeles mayor speaks at conference", 2),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 2


def test_l3_numeric_mismatch_disqualifies_merge():
    """Two funding stories with different amounts must not merge even if
    every other token matches."""
    drafts = [
        _draft(
            "https://x/a",
            "Anthropic raises 500m Series F funding round",
            1,
            published_at="2026-05-04T10:00:00+00:00",
        ),
        _draft(
            "https://x/b",
            "Anthropic raises 1b Series G funding round",
            2,
            published_at="2026-05-04T11:00:00+00:00",
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 2


def test_l3_overlapping_numeric_tokens_allow_merge():
    drafts = [
        _draft(
            "https://x/a",
            "Anthropic raises 1b Series F funding round",
            1,
            published_at="2026-05-04T10:00:00+00:00",
        ),
        _draft(
            "https://x/b",
            "Anthropic raises 1b Series F funding round announcement",
            2,
            published_at="2026-05-04T11:00:00+00:00",
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 1


def test_l3_one_sided_numeric_does_not_disqualify():
    """If only one title carries a number, the numeric guard can't judge —
    fall back to title similarity."""
    drafts = [
        _draft(
            "https://x/a",
            "Anthropic releases new Claude 5 model variant",
            1,
            published_at="2026-05-04T10:00:00+00:00",
        ),
        _draft(
            "https://x/b",
            "Anthropic releases new Claude model variant publicly",
            2,
            published_at="2026-05-04T11:00:00+00:00",
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 1


def test_l3_dates_outside_window_disqualify():
    """Even with identical titles, drafts more than 48h apart describe
    different events (e.g., the same headline used twice for different
    quarterly earnings)."""
    drafts = [
        _draft(
            "https://x/a",
            "Apple announces quarterly earnings beat estimates",
            1,
            published_at="2026-05-01T10:00:00+00:00",
        ),
        _draft(
            "https://x/b",
            "Apple announces quarterly earnings beat estimates",
            2,
            published_at="2026-05-04T10:00:00+00:00",
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 2


def test_l3_dates_inside_window_allow_merge():
    drafts = [
        _draft(
            "https://x/a",
            "Apple announces quarterly earnings beat estimates",
            1,
            published_at="2026-05-04T08:00:00+00:00",
        ),
        _draft(
            "https://x/b",
            "Apple announces quarterly earnings beat estimates",
            2,
            published_at="2026-05-05T18:00:00+00:00",
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 1


def test_l3_missing_date_does_not_disqualify():
    """A fetcher that omits published_at shouldn't lose L3 candidacy —
    fall back to title similarity alone."""
    drafts = [
        _draft("https://x/a", "Anthropic raises Series F funding round", 1, published_at=None),
        _draft(
            "https://x/b",
            "Anthropic raises Series F funding round announcement",
            2,
            published_at=None,
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 1


def test_l3_unparseable_date_treated_as_missing():
    drafts = [
        _draft(
            "https://x/a",
            "Anthropic raises Series F funding round",
            1,
            published_at="not-a-date",
        ),
        _draft(
            "https://x/b",
            "Anthropic raises Series F funding round announcement",
            2,
            published_at="2026-05-04T10:00:00+00:00",
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 1


def test_l3_naive_timestamp_treated_as_missing():
    """tz-naive timestamps are an upstream inconsistency. Permissive: don't
    disqualify on date when we can't compare cleanly."""
    drafts = [
        _draft(
            "https://x/a",
            "Anthropic raises Series F funding round",
            1,
            published_at="2026-05-04T10:00:00",
        ),
        _draft(
            "https://x/b",
            "Anthropic raises Series F funding round announcement",
            2,
            published_at="2026-05-04T10:00:00+00:00",
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 1


def test_l3_transitive_merge_via_chain():
    """A↔B and B↔C both qualify → A B C all merge into one cluster, even
    if A and C alone would not have triggered a direct merge."""
    drafts = [
        _draft(
            "https://x/a",
            "OpenAI announces GPT 5 model release event today",
            1,
            published_at="2026-05-04T10:00:00+00:00",
        ),
        _draft(
            "https://x/b",
            "OpenAI announces GPT 5 model release event today live",
            2,
            published_at="2026-05-04T11:00:00+00:00",
        ),
        _draft(
            "https://x/c",
            "OpenAI announces GPT 5 model release event today live coverage",
            3,
            published_at="2026-05-04T12:00:00+00:00",
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 1
    assert folded[0].raw_item_ids == (1, 2, 3)


def test_l3_partial_fold_leaves_unrelated_drafts_alone():
    drafts = [
        _draft(
            "https://x/a",
            "Anthropic raises Series F funding round",
            1,
            published_at="2026-05-04T10:00:00+00:00",
        ),
        _draft(
            "https://x/b",
            "Anthropic raises Series F funding round announcement",
            2,
            published_at="2026-05-04T11:00:00+00:00",
        ),
        _draft(
            "https://x/c",
            "Stock market closes higher today",
            3,
            published_at="2026-05-04T12:00:00+00:00",
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert len(folded) == 2
    keys = sorted(d.cluster_key for d in folded)
    assert keys == ["https://x/a", "https://x/c"]
    a_draft = next(d for d in folded if d.cluster_key == "https://x/a")
    assert a_draft.raw_item_ids == (1, 2)


def test_l3_idempotent_on_repeat():
    drafts = [
        _draft(
            "https://x/a",
            "Anthropic raises Series F funding round",
            1,
            published_at="2026-05-04T10:00:00+00:00",
        ),
        _draft(
            "https://x/b",
            "Anthropic raises Series F funding round announcement",
            2,
            published_at="2026-05-04T11:00:00+00:00",
        ),
    ]
    once = fold_by_event_similarity(drafts)
    twice = fold_by_event_similarity(once)
    assert once == twice


def test_l3_input_order_independent():
    """Two different input orderings of the same drafts produce the same
    folded output. Critical for re-runs and concurrent fetches that may
    reorder rows."""
    a = _draft(
        "https://x/a",
        "Anthropic raises Series F funding round",
        1,
        published_at="2026-05-04T10:00:00+00:00",
    )
    b = _draft(
        "https://x/b",
        "Anthropic raises Series F funding round announcement",
        2,
        published_at="2026-05-04T11:00:00+00:00",
    )
    c = _draft(
        "https://x/c",
        "Stock market closes higher today",
        3,
        published_at="2026-05-04T12:00:00+00:00",
    )
    forward = fold_by_event_similarity([a, b, c])
    reverse = fold_by_event_similarity([c, b, a])
    assert forward == reverse


def test_l3_merged_cluster_key_is_lex_smallest():
    drafts = [
        _draft(
            "https://yahoo.example/anthropic",
            "Anthropic raises Series F funding round",
            5,
            published_at="2026-05-04T10:00:00+00:00",
        ),
        _draft(
            "https://abc.example/anthropic",
            "Anthropic raises Series F funding round announcement",
            2,
            published_at="2026-05-04T11:00:00+00:00",
        ),
    ]
    folded = fold_by_event_similarity(drafts)
    assert folded[0].cluster_key == "https://abc.example/anthropic"


def test_l3_threshold_can_be_tightened():
    """Same data, higher threshold → no merge. Confirms the parameter is
    actually load-bearing and tunable."""
    drafts = [
        _draft(
            "https://x/a",
            "Anthropic raises Series F funding round",
            1,
            published_at="2026-05-04T10:00:00+00:00",
        ),
        _draft(
            "https://x/b",
            "Anthropic raises Series F funding round announcement deal",
            2,
            published_at="2026-05-04T11:00:00+00:00",
        ),
    ]
    permissive = fold_by_event_similarity(drafts, title_threshold=0.5)
    strict = fold_by_event_similarity(drafts, title_threshold=0.95)
    assert len(permissive) == 1
    assert len(strict) == 2


# ── DB helpers ────────────────────────────────────────────────────────────────


def _make_run(conn: sqlite3.Connection) -> int:
    return worker_db.create_run(
        conn,
        run_type="daily",
        window_start="2026-05-04T00:00:00+00:00",
        window_end="2026-05-05T00:00:00+00:00",
    )


def _seed_item(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    dedup_key: str,
    canonical_url: str,
    title: str = "",
    source_type: str = "rss",
    content: str = "",
    content_hash: str | None = None,
    published_at: str | None = None,
) -> int:
    raw_item_id, _ = worker_db.upsert_raw_item(
        conn,
        run_id=run_id,
        source_type=source_type,
        dedup_key=dedup_key,
        title=title,
        url=canonical_url,
        canonical_url=canonical_url,
        content=content,
        content_hash_value=content_hash,
        published_at=published_at,
    )
    return raw_item_id


def test_iter_run_raw_items_yields_only_run_scoped_items(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_a = _make_run(conn)
        run_b = _make_run(conn)
        item_a1 = _seed_item(conn, run_a, dedup_key="a-1", canonical_url="https://x/a1")
        item_a2 = _seed_item(conn, run_a, dedup_key="a-2", canonical_url="https://x/a2")
        _seed_item(conn, run_b, dedup_key="b-1", canonical_url="https://x/b1")

        rows_a = list(worker_db.iter_run_raw_items(conn, run_a))
        rows_b = list(worker_db.iter_run_raw_items(conn, run_b))

        assert sorted(r["id"] for r in rows_a) == sorted([item_a1, item_a2])
        assert len(rows_b) == 1


def test_iter_run_raw_items_returns_id_ascending(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        # Insertion order intentionally non-monotonic via different dedup keys.
        ids = [
            _seed_item(conn, run_id, dedup_key="z", canonical_url="https://x/z"),
            _seed_item(conn, run_id, dedup_key="a", canonical_url="https://x/a"),
            _seed_item(conn, run_id, dedup_key="m", canonical_url="https://x/m"),
        ]
        rows = list(worker_db.iter_run_raw_items(conn, run_id))
        assert [r["id"] for r in rows] == sorted(ids)


def test_create_cluster_inserts_row_and_attaches_items(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        rid_1 = _seed_item(conn, run_id, dedup_key="a-1", canonical_url="https://x/a")
        rid_2 = _seed_item(conn, run_id, dedup_key="a-2", canonical_url="https://x/a")

        cluster_id, was_new = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://x/a",
            title="Alpha",
            raw_item_ids=[rid_1, rid_2],
        )
        assert was_new is True

        cluster = conn.execute("SELECT * FROM item_clusters WHERE id = ?", (cluster_id,)).fetchone()
        assert cluster["run_id"] == run_id
        assert cluster["cluster_key"] == "https://x/a"
        assert cluster["title"] == "Alpha"
        assert cluster["status"] == "pending"

        members = conn.execute(
            "SELECT raw_item_id FROM cluster_items WHERE cluster_id = ? ORDER BY raw_item_id",
            (cluster_id,),
        ).fetchall()
        assert [m["raw_item_id"] for m in members] == sorted([rid_1, rid_2])


def test_create_cluster_idempotent_on_repeat(temp_db):
    """Re-calling with same (run_id, cluster_key) must not duplicate the
    cluster row, must return was_new=False, must preserve status, and must
    leave cluster_items linkages intact."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        rid = _seed_item(conn, run_id, dedup_key="a", canonical_url="https://x/a")

        cluster_id_1, first = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://x/a",
            title="Alpha",
            raw_item_ids=[rid],
        )
        # Simulate the relevance filter promoting the cluster — clustering
        # must not later overwrite this status.
        conn.execute("UPDATE item_clusters SET status='kept' WHERE id = ?", (cluster_id_1,))

        cluster_id_2, second = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://x/a",
            title="Alpha (again)",
            raw_item_ids=[rid],
        )
        assert cluster_id_1 == cluster_id_2
        assert first is True
        assert second is False

        cluster = conn.execute(
            "SELECT title, status FROM item_clusters WHERE id = ?", (cluster_id_1,)
        ).fetchone()
        assert cluster["title"] == "Alpha"
        assert cluster["status"] == "kept"

        n_clusters = conn.execute(
            "SELECT COUNT(*) AS n FROM item_clusters WHERE run_id = ?", (run_id,)
        ).fetchone()["n"]
        assert n_clusters == 1


def test_create_cluster_appends_new_members_on_repeat(temp_db):
    """A re-run that observes additional members of an existing cluster
    must record the new linkages without duplicating the existing ones."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        rid_1 = _seed_item(conn, run_id, dedup_key="a-1", canonical_url="https://x/a")
        rid_2 = _seed_item(conn, run_id, dedup_key="a-2", canonical_url="https://x/a")

        cluster_id, _ = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://x/a",
            title="Alpha",
            raw_item_ids=[rid_1],
        )
        worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://x/a",
            title="Alpha",
            raw_item_ids=[rid_1, rid_2],
        )

        members = conn.execute(
            "SELECT raw_item_id FROM cluster_items WHERE cluster_id = ? ORDER BY raw_item_id",
            (cluster_id,),
        ).fetchall()
        assert [m["raw_item_id"] for m in members] == sorted([rid_1, rid_2])


def test_create_cluster_rejects_empty_cluster_key(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        rid = _seed_item(conn, run_id, dedup_key="a", canonical_url="https://x/a")
        with pytest.raises(ValueError, match="cluster_key is required"):
            worker_db.create_cluster(
                conn,
                run_id=run_id,
                cluster_key="",
                title="x",
                raw_item_ids=[rid],
            )


def test_create_cluster_rejects_invalid_status(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        rid = _seed_item(conn, run_id, dedup_key="a", canonical_url="https://x/a")
        with pytest.raises(ValueError, match="invalid cluster status"):
            worker_db.create_cluster(
                conn,
                run_id=run_id,
                cluster_key="https://x/a",
                title="x",
                raw_item_ids=[rid],
                status="bogus",
            )


def test_create_cluster_rejects_empty_member_list(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        with pytest.raises(ValueError, match="at least one"):
            worker_db.create_cluster(
                conn,
                run_id=run_id,
                cluster_key="https://x/a",
                title="x",
                raw_item_ids=[],
            )


def test_create_cluster_rolls_back_on_member_fk_violation(temp_db):
    """If any raw_item_id is invalid, the entire transaction must roll back —
    we must not leave a phantom cluster with no attached items."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        rid_real = _seed_item(conn, run_id, dedup_key="a", canonical_url="https://x/a")

        with pytest.raises(sqlite3.IntegrityError):
            worker_db.create_cluster(
                conn,
                run_id=run_id,
                cluster_key="https://x/a",
                title="Alpha",
                raw_item_ids=[rid_real, 999_999],
            )
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM item_clusters WHERE run_id = ?", (run_id,)
        ).fetchone()["n"]
        assert n == 0


# ── orchestration: cluster_run ────────────────────────────────────────────────


def test_cluster_run_groups_run_items_by_canonical_url(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        rid_a = _seed_item(
            conn, run_id, dedup_key="rss/a", canonical_url="https://x/a", title="Alpha"
        )
        rid_a2 = _seed_item(
            conn,
            run_id,
            dedup_key="hn/a",
            canonical_url="https://x/a",
            title="Alpha (HN)",
            source_type="hn",
        )
        rid_b = _seed_item(
            conn, run_id, dedup_key="rss/b", canonical_url="https://x/b", title="Beta"
        )

        cluster_count = cluster_run(conn, run_id)
        assert cluster_count == 2

        clusters = conn.execute(
            "SELECT id, cluster_key, title, status FROM item_clusters "
            "WHERE run_id = ? ORDER BY cluster_key",
            (run_id,),
        ).fetchall()
        assert [(c["cluster_key"], c["status"]) for c in clusters] == [
            ("https://x/a", "pending"),
            ("https://x/b", "pending"),
        ]

        members_by_key = {}
        for c in clusters:
            ids = [
                m["raw_item_id"]
                for m in conn.execute(
                    "SELECT raw_item_id FROM cluster_items WHERE cluster_id = ? "
                    "ORDER BY raw_item_id",
                    (c["id"],),
                ).fetchall()
            ]
            members_by_key[c["cluster_key"]] = ids
        assert members_by_key == {
            "https://x/a": sorted([rid_a, rid_a2]),
            "https://x/b": [rid_b],
        }


def test_cluster_run_empty_run_returns_zero(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        assert cluster_run(conn, run_id) == 0
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM item_clusters WHERE run_id = ?", (run_id,)
        ).fetchone()["n"]
        assert n == 0


def test_cluster_run_all_items_distinct_produces_one_cluster_each(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        for i in range(5):
            _seed_item(
                conn,
                run_id,
                dedup_key=f"d-{i}",
                canonical_url=f"https://x/{i}",
                title=f"Title {i}",
            )
        assert cluster_run(conn, run_id) == 5


def test_cluster_run_all_items_identical_produces_one_cluster(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        for i in range(4):
            _seed_item(
                conn,
                run_id,
                dedup_key=f"src-{i}",
                canonical_url="https://x/same",
                title=f"Headline {i}",
                source_type=("rss", "hn", "reddit", "gdelt")[i],
            )
        assert cluster_run(conn, run_id) == 1
        members = conn.execute(
            """
            SELECT COUNT(*) AS n FROM cluster_items ci
              JOIN item_clusters c ON c.id = ci.cluster_id
             WHERE c.run_id = ?
            """,
            (run_id,),
        ).fetchone()["n"]
        assert members == 4


def test_cluster_run_idempotent_on_repeat(temp_db):
    """Calling cluster_run twice on the same run is a no-op for state and
    must not create duplicate clusters or relink members."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_item(conn, run_id, dedup_key="a", canonical_url="https://x/a", title="A")
        _seed_item(
            conn, run_id, dedup_key="b", canonical_url="https://x/a", title="A2", source_type="hn"
        )
        _seed_item(conn, run_id, dedup_key="c", canonical_url="https://x/c", title="C")

        first = cluster_run(conn, run_id)
        second = cluster_run(conn, run_id)
        assert first == second == 2

        cluster_count = conn.execute(
            "SELECT COUNT(*) AS n FROM item_clusters WHERE run_id = ?", (run_id,)
        ).fetchone()["n"]
        assert cluster_count == 2

        member_count = conn.execute(
            """
            SELECT COUNT(*) AS n FROM cluster_items ci
              JOIN item_clusters c ON c.id = ci.cluster_id
             WHERE c.run_id = ?
            """,
            (run_id,),
        ).fetchone()["n"]
        assert member_count == 3


def test_cluster_run_isolates_runs(temp_db):
    """Two runs that observe overlapping items still produce per-run clusters —
    item_clusters is keyed by run_id, so a topical search re-running over the
    same raw items must see fresh clusters, not the daily run's."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_a = _make_run(conn)
        run_b = _make_run(conn)
        # The same canonical_url discovered in two separate runs.
        _seed_item(conn, run_a, dedup_key="a-1", canonical_url="https://x/a", title="A")
        worker_db.upsert_raw_item(
            conn,
            run_id=run_b,
            source_type="hn",
            dedup_key="a-1",
            title="A",
            url="https://x/a",
            canonical_url="https://x/a",
            content="",
        )

        cluster_run(conn, run_a)
        cluster_run(conn, run_b)

        clusters = conn.execute(
            "SELECT run_id, cluster_key FROM item_clusters ORDER BY run_id"
        ).fetchall()
        assert [(c["run_id"], c["cluster_key"]) for c in clusters] == [
            (run_a, "https://x/a"),
            (run_b, "https://x/a"),
        ]


# ── orchestration: L2 (content-hash fold) end-to-end ──────────────────────────


def test_cluster_run_folds_syndicated_copies_into_one_cluster(temp_db):
    """Two raw items with different canonical URLs but identical content_hash
    represent the same syndicated event — L2 must fold them into a single
    cluster keyed by the smaller URL."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        rid_first = _seed_item(
            conn,
            run_id,
            dedup_key="orig",
            canonical_url="https://example.com/article",
            title="Original headline",
            content="shared body",
            content_hash="hash-syndicated",
        )
        rid_syn = _seed_item(
            conn,
            run_id,
            dedup_key="syn",
            canonical_url="https://yahoo.com/article",
            title="Yahoo's copy",
            source_type="hn",
            content="shared body",
            content_hash="hash-syndicated",
        )

        assert cluster_run(conn, run_id) == 1

        cluster = conn.execute(
            "SELECT id, cluster_key, title FROM item_clusters WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert cluster["cluster_key"] == "https://example.com/article"
        assert cluster["title"] == "Original headline"

        members = [
            row["raw_item_id"]
            for row in conn.execute(
                "SELECT raw_item_id FROM cluster_items WHERE cluster_id = ? ORDER BY raw_item_id",
                (cluster["id"],),
            ).fetchall()
        ]
        assert members == sorted([rid_first, rid_syn])


def test_cluster_run_keeps_distinct_hashes_distinct(temp_db):
    """Two URLs, two hashes — L2 should not merge them. Sanity check that
    the fold is not over-eager."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_item(
            conn,
            run_id,
            dedup_key="a",
            canonical_url="https://example.com/a",
            title="Alpha",
            content_hash="hash-a",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="b",
            canonical_url="https://example.com/b",
            title="Beta",
            content_hash="hash-b",
        )
        assert cluster_run(conn, run_id) == 2


def test_cluster_run_l1_takes_precedence_over_l2(temp_db):
    """When two items share BOTH canonical URL and content_hash, L1 has
    already collapsed them — L2 should be a no-op for that pair. Verifies
    the chained pipeline doesn't double-count."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_item(
            conn,
            run_id,
            dedup_key="a-rss",
            canonical_url="https://example.com/a",
            title="Alpha (RSS)",
            content_hash="hash-shared",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="a-hn",
            canonical_url="https://example.com/a",
            title="Alpha (HN)",
            source_type="hn",
            content_hash="hash-shared",
        )
        assert cluster_run(conn, run_id) == 1


def test_cluster_run_combines_l1_and_l2_paths(temp_db):
    """Mixed scenario: two items collapse at L1 (same URL), two more
    collapse at L2 (different URLs, same content_hash), and one stands
    alone. End count: 3 clusters."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        # L1 group: two sources point at the same URL.
        _seed_item(
            conn,
            run_id,
            dedup_key="rss/x",
            canonical_url="https://example.com/x",
            title="X (RSS)",
            content_hash="hash-x",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="hn/x",
            canonical_url="https://example.com/x",
            title="X (HN)",
            source_type="hn",
            content_hash="hash-x",
        )
        # L2 group: same content syndicated to two distinct URLs.
        _seed_item(
            conn,
            run_id,
            dedup_key="orig/y",
            canonical_url="https://news.example/y",
            title="Y original",
            content_hash="hash-syndicated",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="syn/y",
            canonical_url="https://other-news.example/y",
            title="Y syndicated",
            source_type="gdelt",
            content_hash="hash-syndicated",
        )
        # Standalone.
        _seed_item(
            conn,
            run_id,
            dedup_key="z",
            canonical_url="https://example.com/z",
            title="Z",
            content_hash="hash-z",
        )

        assert cluster_run(conn, run_id) == 3

        members = conn.execute(
            """
            SELECT c.cluster_key, COUNT(ci.raw_item_id) AS n_members
              FROM item_clusters c
              LEFT JOIN cluster_items ci ON ci.cluster_id = c.id
             WHERE c.run_id = ?
             GROUP BY c.id
             ORDER BY c.cluster_key
            """,
            (run_id,),
        ).fetchall()
        size_by_key = {row["cluster_key"]: row["n_members"] for row in members}
        assert size_by_key == {
            "https://example.com/x": 2,
            "https://example.com/z": 1,
            "https://news.example/y": 2,
        }


def test_cluster_run_does_not_fold_drafts_without_content_hash(temp_db):
    """Items missing content_hash (defensive case) must remain distinct in
    L2 — folding on absence would falsely merge unrelated items."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_item(
            conn,
            run_id,
            dedup_key="a",
            canonical_url="https://example.com/a",
            title="Alpha",
            content_hash=None,
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="b",
            canonical_url="https://example.com/b",
            title="Beta",
            content_hash=None,
        )
        assert cluster_run(conn, run_id) == 2


def test_cluster_run_l2_idempotent_on_repeat(temp_db):
    """Re-running cluster_run after L2 has already folded a syndicated pair
    is a no-op for the persisted state (no duplicate clusters, no extra
    cluster_items rows)."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_item(
            conn,
            run_id,
            dedup_key="orig",
            canonical_url="https://example.com/y",
            title="Y",
            content_hash="hash-y",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="syn",
            canonical_url="https://yahoo.com/y",
            title="Y syn",
            source_type="hn",
            content_hash="hash-y",
        )
        first = cluster_run(conn, run_id)
        second = cluster_run(conn, run_id)
        assert first == second == 1

        cluster_count = conn.execute(
            "SELECT COUNT(*) AS n FROM item_clusters WHERE run_id = ?", (run_id,)
        ).fetchone()["n"]
        assert cluster_count == 1
        member_count = conn.execute(
            """
            SELECT COUNT(*) AS n FROM cluster_items ci
              JOIN item_clusters c ON c.id = ci.cluster_id
             WHERE c.run_id = ?
            """,
            (run_id,),
        ).fetchone()["n"]
        assert member_count == 2


# ── orchestration: L3 (heuristic event similarity) end-to-end ────────────────


def test_cluster_run_folds_similar_titles_via_l3(temp_db):
    """Two stories about the same event written up by different outlets —
    different URLs, different bodies (so L1 and L2 both miss them), but
    titles overlap heavily and dates are within window. L3 should fold."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_item(
            conn,
            run_id,
            dedup_key="rss/anthropic",
            canonical_url="https://outlet-a.example/anthropic-funding",
            title="Anthropic raises Series F funding round",
            content="rss body",
            content_hash="hash-rss",
            published_at="2026-05-04T10:00:00+00:00",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="hn/anthropic",
            canonical_url="https://outlet-b.example/anthropic-funding",
            title="Anthropic raises Series F funding round announcement",
            source_type="hn",
            content="hn body",
            content_hash="hash-hn",
            published_at="2026-05-04T11:00:00+00:00",
        )
        assert cluster_run(conn, run_id) == 1

        cluster = conn.execute(
            "SELECT cluster_key, title FROM item_clusters WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert cluster["cluster_key"] == "https://outlet-a.example/anthropic-funding"
        assert cluster["title"] == "Anthropic raises Series F funding round"


def test_cluster_run_l3_preserves_distinct_funding_amounts(temp_db):
    """Two funding stories with different amounts must NOT collapse —
    L3's numeric guard is the safety net here."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_item(
            conn,
            run_id,
            dedup_key="a",
            canonical_url="https://a.example/anthropic-1",
            title="Anthropic raises 500m Series F funding round",
            content_hash="ha",
            published_at="2026-05-04T10:00:00+00:00",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="b",
            canonical_url="https://b.example/anthropic-2",
            title="Anthropic raises 1b Series G funding round",
            content_hash="hb",
            published_at="2026-05-04T11:00:00+00:00",
        )
        assert cluster_run(conn, run_id) == 2


def test_cluster_run_l3_respects_date_window(temp_db):
    """Identical headlines on dates outside the L3 window stay distinct —
    the same headline used twice for two quarterly events shouldn't
    collapse into one cluster."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_item(
            conn,
            run_id,
            dedup_key="a",
            canonical_url="https://a.example/q1",
            title="Apple announces quarterly earnings beat estimates",
            content_hash="ha",
            published_at="2026-02-01T10:00:00+00:00",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="b",
            canonical_url="https://b.example/q2",
            title="Apple announces quarterly earnings beat estimates",
            content_hash="hb",
            published_at="2026-05-01T10:00:00+00:00",
        )
        assert cluster_run(conn, run_id) == 2


def test_cluster_run_chains_l1_l2_and_l3_independently(temp_db):
    """Mixed scenario exercising all three levels in one run:

    - Two items at the same canonical URL (L1 collapse).
    - Two items at different URLs but with the same content_hash (L2 fold).
    - Two items at different URLs, different content_hashes, but similar
      titles within the date window (L3 fold).
    - One standalone.

    Expected: 4 final clusters."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        # L1 group.
        _seed_item(
            conn,
            run_id,
            dedup_key="rss/x",
            canonical_url="https://example.com/x",
            title="Event X happened today",
            content_hash="hash-x",
            published_at="2026-05-04T08:00:00+00:00",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="hn/x",
            canonical_url="https://example.com/x",
            title="Event X happened today (HN)",
            source_type="hn",
            content_hash="hash-x",
            published_at="2026-05-04T08:30:00+00:00",
        )
        # L2 group.
        _seed_item(
            conn,
            run_id,
            dedup_key="orig/y",
            canonical_url="https://news-a.example/y",
            title="Event Y syndicated copy",
            content_hash="hash-syn",
            published_at="2026-05-04T09:00:00+00:00",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="syn/y",
            canonical_url="https://news-b.example/y",
            title="Event Y syndicated copy",
            source_type="gdelt",
            content_hash="hash-syn",
            published_at="2026-05-04T09:30:00+00:00",
        )
        # L3 group.
        _seed_item(
            conn,
            run_id,
            dedup_key="z-rss",
            canonical_url="https://outlet-1.example/zeta-funding",
            title="Zeta startup raises Series A funding round",
            content_hash="hash-z1",
            published_at="2026-05-04T10:00:00+00:00",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="z-hn",
            canonical_url="https://outlet-2.example/zeta-funding",
            title="Zeta startup raises Series A funding round announcement",
            source_type="hn",
            content_hash="hash-z2",
            published_at="2026-05-04T11:00:00+00:00",
        )
        # Standalone.
        _seed_item(
            conn,
            run_id,
            dedup_key="standalone",
            canonical_url="https://example.com/unrelated",
            title="Stock market closes higher today",
            content_hash="hash-unrelated",
            published_at="2026-05-04T12:00:00+00:00",
        )

        assert cluster_run(conn, run_id) == 4

        sizes = conn.execute(
            """
            SELECT c.cluster_key, COUNT(ci.raw_item_id) AS n_members
              FROM item_clusters c
              JOIN cluster_items ci ON ci.cluster_id = c.id
             WHERE c.run_id = ?
             GROUP BY c.id
             ORDER BY c.cluster_key
            """,
            (run_id,),
        ).fetchall()
        size_by_key = {r["cluster_key"]: r["n_members"] for r in sizes}
        assert size_by_key == {
            "https://example.com/x": 2,
            "https://news-a.example/y": 2,
            "https://outlet-1.example/zeta-funding": 2,
            "https://example.com/unrelated": 1,
        }


def test_cluster_run_l3_idempotent_on_repeat(temp_db):
    """Re-running cluster_run after L3 has folded a similar pair must not
    duplicate clusters or relink members."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_item(
            conn,
            run_id,
            dedup_key="a",
            canonical_url="https://a.example/anthropic",
            title="Anthropic raises Series F funding round",
            content_hash="ha",
            published_at="2026-05-04T10:00:00+00:00",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="b",
            canonical_url="https://b.example/anthropic",
            title="Anthropic raises Series F funding round announcement",
            content_hash="hb",
            published_at="2026-05-04T11:00:00+00:00",
        )
        first = cluster_run(conn, run_id)
        second = cluster_run(conn, run_id)
        assert first == second == 1

        cluster_count = conn.execute(
            "SELECT COUNT(*) AS n FROM item_clusters WHERE run_id = ?", (run_id,)
        ).fetchone()["n"]
        assert cluster_count == 1
        member_count = conn.execute(
            """
            SELECT COUNT(*) AS n FROM cluster_items ci
              JOIN item_clusters c ON c.id = ci.cluster_id
             WHERE c.run_id = ?
            """,
            (run_id,),
        ).fetchone()["n"]
        assert member_count == 2


def test_cluster_run_l3_does_not_cross_into_l1_dup_keys(temp_db):
    """Sanity: with two L1 drafts that L3 would have folded, the persisted
    cluster_key uses the SMALLER URL — not whichever happened to fold
    second. Stable across re-runs."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_item(
            conn,
            run_id,
            dedup_key="zz",
            canonical_url="https://zz.example/anthropic",
            title="Anthropic raises Series F funding round",
            content_hash="hzz",
            published_at="2026-05-04T10:00:00+00:00",
        )
        _seed_item(
            conn,
            run_id,
            dedup_key="aa",
            canonical_url="https://aa.example/anthropic",
            title="Anthropic raises Series F funding round announcement",
            content_hash="haa",
            published_at="2026-05-04T11:00:00+00:00",
        )
        cluster_run(conn, run_id)
        cluster = conn.execute(
            "SELECT cluster_key FROM item_clusters WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert cluster["cluster_key"] == "https://aa.example/anthropic"
