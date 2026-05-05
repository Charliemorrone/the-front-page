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
)


# ── pure: cluster_by_canonical_url ────────────────────────────────────────────


def _row(rid: int, canonical: str, title: str = "") -> dict[str, object]:
    return {"id": rid, "canonical_url": canonical, "title": title}


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
