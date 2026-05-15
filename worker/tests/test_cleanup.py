"""Tests for the retention helpers and ``clawfeed-intel cleanup`` CLI.

The retention layer keeps the worker DB from growing unbounded over
indefinite daily operation — architecture-doc policy: raw items 30-90
days, llm calls 30 days. These tests pin:

- the SQL-side correctness of the prune + count helpers
- the FK cascade behavior (deleting raw_items removes the joining
  ``run_raw_items`` and ``cluster_items`` rows; ``item_clusters`` /
  ``item_summaries`` are preserved)
- the CLI's dry-run-by-default + ``--apply`` opt-in contract
- the exit-code contract cron will read
"""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from clawfeed_intel import cli, db as worker_db
from clawfeed_intel.llm.schemas import ClusterSummaryPayload


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_run(conn: sqlite3.Connection) -> int:
    return worker_db.create_run(
        conn,
        run_type="daily",
        window_start="2026-04-01T00:00:00+00:00",
        window_end="2026-04-02T00:00:00+00:00",
    )


def _seed_raw_item(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    key: str,
    fetched_at: str,
) -> int:
    """Insert a raw_item with an explicit ``fetched_at`` so retention windows
    can be exercised without freezegun / time-mocking.
    """
    raw_id, _ = worker_db.upsert_raw_item(
        conn,
        run_id=run_id,
        source_type="rss",
        dedup_key=key,
        title=key,
        url=key,
        canonical_url=key,
        content="",
    )
    conn.execute(
        "UPDATE raw_items SET fetched_at = ? WHERE id = ?",
        (fetched_at, raw_id),
    )
    conn.commit()
    return raw_id


def _seed_llm_call(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    created_at: str,
    stage: str = "relevance_filter",
) -> int:
    call_id = worker_db.record_llm_call(
        conn,
        run_id=run_id,
        stage=stage,
        provider="vmlx",
        model="stub-model",
        status="succeeded",
        latency_ms=100,
        prompt_tokens=10,
        completion_tokens=5,
    )
    conn.execute(
        "UPDATE llm_calls SET created_at = ? WHERE id = ?",
        (created_at, call_id),
    )
    conn.commit()
    return call_id


# ── cutoff_iso ────────────────────────────────────────────────────────────────


def test_cutoff_iso_subtracts_keep_days_from_now() -> None:
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
    cutoff = worker_db.cutoff_iso(now=now, keep_days=30)
    assert cutoff == "2026-04-15T12:00:00+00:00"


def test_cutoff_iso_zero_days_is_now() -> None:
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert worker_db.cutoff_iso(now=now, keep_days=0) == "2026-05-15T12:00:00+00:00"


def test_cutoff_iso_rejects_negative_keep_days() -> None:
    """Negative keep is nonsense — fail at the boundary rather than
    quietly producing a future cutoff that matches nothing.
    """
    with pytest.raises(ValueError, match=">=.*0"):
        worker_db.cutoff_iso(keep_days=-1)


def test_cutoff_iso_default_now_is_current_utc() -> None:
    """When ``now`` is omitted, the result should be close to wall time.

    Coarse window — within 5 seconds — because the test runs against
    real time. The point is to prove the default path works.
    """
    before = datetime.now(timezone.utc)
    out = worker_db.cutoff_iso(keep_days=10)
    after = datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(out)
    expected_min = before - timedelta(days=10) - timedelta(seconds=1)
    expected_max = after - timedelta(days=10) + timedelta(seconds=1)
    assert expected_min <= parsed <= expected_max


# ── prune_raw_items_before + count_raw_items_before ──────────────────────────


def test_prune_raw_items_removes_older_rows(temp_db: Path) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        old = _seed_raw_item(
            conn,
            run_id=run_id,
            key="https://old.example/",
            fetched_at="2026-01-01T00:00:00+00:00",
        )
        new = _seed_raw_item(
            conn,
            run_id=run_id,
            key="https://new.example/",
            fetched_at="2026-05-14T00:00:00+00:00",
        )

        removed = worker_db.prune_raw_items_before(conn, "2026-03-01T00:00:00+00:00")
        assert removed == 1

        survivors = {row["id"] for row in conn.execute("SELECT id FROM raw_items").fetchall()}
        assert survivors == {new}
        assert old not in survivors


def test_prune_raw_items_preserves_rows_at_or_after_cutoff(temp_db: Path) -> None:
    """``fetched_at < cutoff`` is strict — a row exactly at the cutoff
    survives. Documented behavior so callers know edge timestamps don't
    accidentally vanish.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        boundary = _seed_raw_item(
            conn,
            run_id=run_id,
            key="https://boundary.example/",
            fetched_at="2026-03-01T00:00:00+00:00",
        )
        removed = worker_db.prune_raw_items_before(conn, "2026-03-01T00:00:00+00:00")
        assert removed == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_items WHERE id = ?", (boundary,)).fetchone()[0]
            == 1
        )


def test_prune_raw_items_cascades_to_join_tables(temp_db: Path) -> None:
    """``run_raw_items`` and ``cluster_items`` declare ``ON DELETE CASCADE``
    against ``raw_items``. Verified end-to-end so a future migration
    that loosens the cascade would fail this test.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        raw_id = _seed_raw_item(
            conn,
            run_id=run_id,
            key="https://x.example/",
            fetched_at="2026-01-01T00:00:00+00:00",
        )
        cluster_id, _ = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://x.example/",
            title="t",
            raw_item_ids=[raw_id],
        )

        # Before prune: linkage rows exist.
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM run_raw_items WHERE raw_item_id = ?", (raw_id,)
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM cluster_items WHERE raw_item_id = ?", (raw_id,)
            ).fetchone()[0]
            == 1
        )

        worker_db.prune_raw_items_before(conn, "2026-03-01T00:00:00+00:00")

        # After prune: linkage rows cascaded away; cluster row survives.
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM run_raw_items WHERE raw_item_id = ?", (raw_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM cluster_items WHERE raw_item_id = ?", (raw_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM item_clusters WHERE id = ?", (cluster_id,)
            ).fetchone()[0]
            == 1
        )


def test_prune_raw_items_preserves_item_summaries(temp_db: Path) -> None:
    """Architecture-doc rule: "Runs and summaries: indefinite until
    manually pruned." A cluster's ``item_summaries`` row must survive
    even when all its raw_items aged out.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        raw_id = _seed_raw_item(
            conn,
            run_id=run_id,
            key="https://x.example/",
            fetched_at="2026-01-01T00:00:00+00:00",
        )
        cluster_id, _ = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://x.example/",
            title="t",
            raw_item_ids=[raw_id],
        )
        worker_db.update_cluster_verdict(
            conn,
            cluster_id=cluster_id,
            status="kept",
            relevance_score=0.8,
            category="x",
            event_type=None,
            filter_reason="r",
        )
        summary_id = worker_db.create_item_summary(
            conn,
            cluster_id=cluster_id,
            model="m",
            prompt_version="summary.v1",
            payload=ClusterSummaryPayload(headline="h", summary="s"),
        )

        worker_db.prune_raw_items_before(conn, "2026-03-01T00:00:00+00:00")

        row = conn.execute(
            "SELECT id, headline FROM item_summaries WHERE id = ?", (summary_id,)
        ).fetchone()
        assert row is not None
        assert row["headline"] == "h"


def test_count_raw_items_before_matches_prune(temp_db: Path) -> None:
    """The count helper is the dry-run preview; it must match what
    prune would actually remove with the same cutoff.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        for i, fetched_at in enumerate(
            [
                "2025-12-01T00:00:00+00:00",
                "2026-01-15T00:00:00+00:00",
                "2026-02-20T00:00:00+00:00",
                "2026-05-14T00:00:00+00:00",  # this one survives
            ],
        ):
            _seed_raw_item(
                conn,
                run_id=run_id,
                key=f"https://x{i}.example/",
                fetched_at=fetched_at,
            )

        cutoff = "2026-03-01T00:00:00+00:00"
        previewed = worker_db.count_raw_items_before(conn, cutoff)
        removed = worker_db.prune_raw_items_before(conn, cutoff)
        assert previewed == removed == 3


def test_prune_raw_items_empty_db_returns_zero(temp_db: Path) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        assert worker_db.prune_raw_items_before(conn, "2026-01-01T00:00:00+00:00") == 0
        assert worker_db.count_raw_items_before(conn, "2026-01-01T00:00:00+00:00") == 0


# ── prune_llm_calls_before + count_llm_calls_before ──────────────────────────


def test_prune_llm_calls_removes_older_rows(temp_db: Path) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        old = _seed_llm_call(
            conn,
            run_id=run_id,
            created_at="2026-01-01T00:00:00+00:00",
        )
        new = _seed_llm_call(
            conn,
            run_id=run_id,
            created_at="2026-05-14T00:00:00+00:00",
        )

        removed = worker_db.prune_llm_calls_before(conn, "2026-04-15T00:00:00+00:00")
        assert removed == 1

        survivors = {row["id"] for row in conn.execute("SELECT id FROM llm_calls").fetchall()}
        assert survivors == {new}
        assert old not in survivors


def test_prune_llm_calls_preserves_intel_run(temp_db: Path) -> None:
    """``llm_calls.run_id`` is ``ON DELETE SET NULL``, not cascade.
    Pruning a run's audit rows must NOT touch the run itself — runs
    are indefinite per the architecture-doc retention rule.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_llm_call(
            conn,
            run_id=run_id,
            created_at="2026-01-01T00:00:00+00:00",
        )

        removed = worker_db.prune_llm_calls_before(conn, "2026-04-15T00:00:00+00:00")
        assert removed == 1
        # The run survives.
        assert (
            conn.execute("SELECT COUNT(*) FROM intel_runs WHERE id = ?", (run_id,)).fetchone()[0]
            == 1
        )


def test_count_llm_calls_before_matches_prune(temp_db: Path) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        for created_at in [
            "2025-11-01T00:00:00+00:00",
            "2026-01-10T00:00:00+00:00",
            "2026-05-14T00:00:00+00:00",
        ]:
            _seed_llm_call(conn, run_id=run_id, created_at=created_at)
        cutoff = "2026-04-01T00:00:00+00:00"
        assert worker_db.count_llm_calls_before(conn, cutoff) == 2
        assert worker_db.prune_llm_calls_before(conn, cutoff) == 2


def test_prune_llm_calls_empty_db_returns_zero(temp_db: Path) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        assert worker_db.prune_llm_calls_before(conn, "2026-01-01T00:00:00+00:00") == 0


# ── CLI: clawfeed-intel cleanup ──────────────────────────────────────────────


def _point_cli_at_temp_db(monkeypatch: pytest.MonkeyPatch, temp_db: Path) -> None:
    """Both ``cli.DB_PATH`` and ``db.DB_PATH`` are read independently;
    a unified ``DIGEST_DB`` env-var rebind would work too but
    monkeypatching the module-level binding is what production code
    already does for tests (see :file:`test_cli_doctor.py`).
    """
    monkeypatch.setattr("clawfeed_intel.cli.DB_PATH", temp_db)
    monkeypatch.setattr("clawfeed_intel.db.DB_PATH", temp_db)


def _cleanup_args(*, apply: bool, raw_keep: int = 90, llm_keep: int = 30) -> argparse.Namespace:
    return argparse.Namespace(
        cmd="cleanup",
        raw_items_keep_days=raw_keep,
        llm_calls_keep_days=llm_keep,
        apply=apply,
    )


def test_cleanup_dry_run_does_not_delete(
    temp_db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--apply`` the CLI prints counts and exits 0; the DB
    must not change. Cron + interactive users both rely on this.
    """
    _point_cli_at_temp_db(monkeypatch, temp_db)
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_raw_item(
            conn,
            run_id=run_id,
            key="https://old.example/",
            fetched_at="2025-01-01T00:00:00+00:00",
        )
        _seed_llm_call(
            conn,
            run_id=run_id,
            created_at="2025-01-01T00:00:00+00:00",
        )

    exit_code = cli.cmd_cleanup(_cleanup_args(apply=False))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "would remove 1 raw_items" in captured.out
    assert "would remove 1 llm_calls" in captured.out
    assert "re-run with --apply" in captured.out

    # The DB is untouched.
    with closing(worker_db.connect(temp_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_items").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 1


def test_cleanup_apply_actually_deletes(
    temp_db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _point_cli_at_temp_db(monkeypatch, temp_db)
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_raw_item(
            conn,
            run_id=run_id,
            key="https://old.example/",
            fetched_at="2025-01-01T00:00:00+00:00",
        )
        _seed_raw_item(
            conn,
            run_id=run_id,
            key="https://new.example/",
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        _seed_llm_call(
            conn,
            run_id=run_id,
            created_at="2025-01-01T00:00:00+00:00",
        )

    exit_code = cli.cmd_cleanup(_cleanup_args(apply=True))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "removed 1 raw_items" in captured.out
    assert "removed 1 llm_calls" in captured.out

    with closing(worker_db.connect(temp_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_items").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 0


def test_cleanup_keep_days_args_respected(
    temp_db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 7-day window should keep more data than the 30/90-day default.

    Two items: one 60 days old, one 14 days old. With ``--raw-items-keep-days=7``
    both are old. With ``--raw-items-keep-days=30`` only the 60-day-old
    one is.
    """
    _point_cli_at_temp_db(monkeypatch, temp_db)
    now = datetime.now(timezone.utc)
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_raw_item(
            conn,
            run_id=run_id,
            key="https://60d.example/",
            fetched_at=(now - timedelta(days=60)).isoformat(timespec="seconds"),
        )
        _seed_raw_item(
            conn,
            run_id=run_id,
            key="https://14d.example/",
            fetched_at=(now - timedelta(days=14)).isoformat(timespec="seconds"),
        )

    # 7-day window: both eligible.
    cli.cmd_cleanup(_cleanup_args(apply=False, raw_keep=7, llm_keep=7))
    assert "would remove 2 raw_items" in capsys.readouterr().out

    # 30-day window: only the 60d row.
    cli.cmd_cleanup(_cleanup_args(apply=False, raw_keep=30, llm_keep=30))
    assert "would remove 1 raw_items" in capsys.readouterr().out


def test_cleanup_empty_db_reports_zero(
    temp_db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _point_cli_at_temp_db(monkeypatch, temp_db)
    exit_code = cli.cmd_cleanup(_cleanup_args(apply=False))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "would remove 0 raw_items" in captured.out
    assert "would remove 0 llm_calls" in captured.out


def test_cleanup_subcommand_wired_in_main(
    temp_db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main(["cleanup"])`` should reach :func:`cmd_cleanup` end-to-end."""
    _point_cli_at_temp_db(monkeypatch, temp_db)
    exit_code = cli.main(["cleanup"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "would remove" in captured.out
