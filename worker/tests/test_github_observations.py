"""Tests for the GitHub repository observations storage layer.

These tests pin behavior of :func:`db.record_repo_observation` and
:func:`db.get_repo_velocity` — the substrate the GitHub fetcher will sit on
in step 6.7b. The fetcher is *not* tested here; this layer is purely about
storage and velocity arithmetic.

The architecture doc's hard requirement is "real velocity from stored
observations, not Trending alone." This test file is where that requirement
lives:

- Velocity is the time-ordered delta within an explicit window.
- Day-1 (single observation) returns None — accepted in the architecture
  doc's open risks list.
- Observations outside the window are not pulled in (would conflate
  historical level with current trend).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clawfeed_intel import db


# ── helpers ───────────────────────────────────────────────────────────────────


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── record_repo_observation ───────────────────────────────────────────────────


def test_record_inserts_row_and_returns_id(temp_db):
    conn = db.connect(temp_db)
    try:
        row_id = db.record_repo_observation(
            conn,
            full_name="anthropics/claude-cookbook",
            stars=12000,
            forks=900,
            watchers=12000,
            open_issues=42,
            language="Python",
            topics=["ai", "llm", "anthropic"],
            pushed_at="2026-05-01T08:00:00+00:00",
            discovered_via="trending",
        )
        assert row_id > 0

        fetched = conn.execute(
            "SELECT * FROM github_repo_observations WHERE id = ?", (row_id,)
        ).fetchone()
        assert fetched["full_name"] == "anthropics/claude-cookbook"
        assert fetched["stars"] == 12000
        assert fetched["forks"] == 900
        assert fetched["language"] == "Python"
        assert fetched["topics"] == '["ai","llm","anthropic"]'
        assert fetched["pushed_at"] == "2026-05-01T08:00:00+00:00"
        assert fetched["discovered_via"] == "trending"
    finally:
        conn.close()


def test_record_with_minimum_fields(temp_db):
    """Every optional field is genuinely optional except discovered_via."""
    conn = db.connect(temp_db)
    try:
        row_id = db.record_repo_observation(
            conn,
            full_name="owner/min",
            stars=1,
            discovered_via="search",
        )
        fetched = conn.execute(
            "SELECT * FROM github_repo_observations WHERE id = ?", (row_id,)
        ).fetchone()
        assert fetched["forks"] is None
        assert fetched["watchers"] is None
        assert fetched["open_issues"] is None
        assert fetched["language"] is None
        assert fetched["topics"] == "[]"
        assert fetched["pushed_at"] is None
    finally:
        conn.close()


def test_record_appends_rather_than_upserting(temp_db):
    """Each call is a fresh observation — not an update. Velocity needs the
    full history of timestamped points."""
    conn = db.connect(temp_db)
    try:
        for stars in (100, 110, 130):
            db.record_repo_observation(
                conn,
                full_name="owner/repo",
                stars=stars,
                discovered_via="trending",
            )
        rows = conn.execute(
            "SELECT stars FROM github_repo_observations WHERE full_name = ? ORDER BY id",
            ("owner/repo",),
        ).fetchall()
        assert [r["stars"] for r in rows] == [100, 110, 130]
    finally:
        conn.close()


def test_record_uses_explicit_observed_at_when_provided(temp_db):
    conn = db.connect(temp_db)
    try:
        when = "2026-04-30T12:00:00+00:00"
        row_id = db.record_repo_observation(
            conn,
            full_name="owner/repo",
            stars=42,
            discovered_via="trending",
            observed_at=when,
        )
        fetched = conn.execute(
            "SELECT observed_at FROM github_repo_observations WHERE id = ?", (row_id,)
        ).fetchone()
        assert fetched["observed_at"] == when
    finally:
        conn.close()


def test_record_rejects_blank_full_name(temp_db):
    conn = db.connect(temp_db)
    try:
        with pytest.raises(ValueError, match="full_name"):
            db.record_repo_observation(conn, full_name="", stars=1, discovered_via="trending")
        with pytest.raises(ValueError, match="full_name"):
            db.record_repo_observation(conn, full_name="   ", stars=1, discovered_via="trending")
    finally:
        conn.close()


def test_record_rejects_negative_stars(temp_db):
    conn = db.connect(temp_db)
    try:
        with pytest.raises(ValueError, match="stars"):
            db.record_repo_observation(conn, full_name="o/r", stars=-1, discovered_via="trending")
    finally:
        conn.close()


def test_record_rejects_invalid_discovered_via(temp_db):
    """SQL CHECK constraint catches typos like ``Trending`` or ``manual``."""
    import sqlite3

    conn = db.connect(temp_db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db.record_repo_observation(
                conn,
                full_name="o/r",
                stars=1,
                discovered_via="manual",  # type: ignore[arg-type]
            )
    finally:
        conn.close()


def test_record_strips_whitespace_from_full_name(temp_db):
    conn = db.connect(temp_db)
    try:
        db.record_repo_observation(
            conn, full_name="  owner/repo  ", stars=10, discovered_via="trending"
        )
        rows = conn.execute("SELECT full_name FROM github_repo_observations").fetchall()
        assert rows[0]["full_name"] == "owner/repo"
    finally:
        conn.close()


# ── get_repo_velocity ─────────────────────────────────────────────────────────


def test_velocity_none_when_no_observations(temp_db):
    conn = db.connect(temp_db)
    try:
        assert db.get_repo_velocity(conn, full_name="owner/unseen") is None
    finally:
        conn.close()


def test_velocity_none_with_single_observation(temp_db):
    """Day-1 has no velocity. Architecture doc's open-risks list explicitly
    accepts this."""
    conn = db.connect(temp_db)
    try:
        db.record_repo_observation(
            conn, full_name="owner/repo", stars=100, discovered_via="trending"
        )
        assert db.get_repo_velocity(conn, full_name="owner/repo") is None
    finally:
        conn.close()


def test_velocity_computes_star_delta_in_time_order(temp_db):
    conn = db.connect(temp_db)
    try:
        now = _now()
        db.record_repo_observation(
            conn,
            full_name="owner/repo",
            stars=100,
            forks=20,
            discovered_via="trending",
            observed_at=_iso(now - timedelta(days=2)),
        )
        db.record_repo_observation(
            conn,
            full_name="owner/repo",
            stars=180,
            forks=35,
            discovered_via="trending",
            observed_at=_iso(now),
        )
        v = db.get_repo_velocity(conn, full_name="owner/repo")
        assert v is not None
        assert v.full_name == "owner/repo"
        assert v.star_delta == 80
        assert v.fork_delta == 15
        assert v.earliest_stars == 100
        assert v.latest_stars == 180
        assert v.observation_count == 2
        # 2 days observed, modulo a small float tolerance
        assert 1.9 < v.days_observed < 2.1
    finally:
        conn.close()


def test_velocity_uses_time_order_not_star_min_max(temp_db):
    """Cross-check: a temporary star dip (unstar wave) shouldn't depress the
    delta — the first vs latest comparison is what counts."""
    conn = db.connect(temp_db)
    try:
        now = _now()
        for delta_days, stars in [(3, 100), (2, 90), (1, 95), (0, 200)]:
            db.record_repo_observation(
                conn,
                full_name="owner/repo",
                stars=stars,
                discovered_via="trending",
                observed_at=_iso(now - timedelta(days=delta_days)),
            )
        v = db.get_repo_velocity(conn, full_name="owner/repo")
        assert v is not None
        # earliest = 100 (3 days ago), latest = 200 (now). Delta = 100.
        # NOT 200 - 90 = 110 (which min/max would give).
        assert v.star_delta == 100
        assert v.earliest_stars == 100
        assert v.latest_stars == 200
        assert v.observation_count == 4
    finally:
        conn.close()


def test_velocity_excludes_observations_outside_window(temp_db):
    """Observations older than window_days must not contribute. Pulling
    a 6-month-old level would conflate historical state with recent trend."""
    conn = db.connect(temp_db)
    try:
        ref = _now()
        # Way outside a 7-day window
        db.record_repo_observation(
            conn,
            full_name="owner/repo",
            stars=10,
            discovered_via="trending",
            observed_at=_iso(ref - timedelta(days=180)),
        )
        # Inside the window — only one observation
        db.record_repo_observation(
            conn,
            full_name="owner/repo",
            stars=500,
            discovered_via="trending",
            observed_at=_iso(ref - timedelta(hours=6)),
        )
        # Reference pinned so the test isn't sensitive to wall clock
        v = db.get_repo_velocity(
            conn,
            full_name="owner/repo",
            window_days=7,
            reference_at=_iso(ref),
        )
        # Only one observation in window → no velocity (NOT 500-10=490)
        assert v is None
    finally:
        conn.close()


def test_velocity_window_days_widens_the_view(temp_db):
    """Same dataset, wider window pulls the older observation in."""
    conn = db.connect(temp_db)
    try:
        ref = _now()
        db.record_repo_observation(
            conn,
            full_name="owner/repo",
            stars=10,
            discovered_via="trending",
            observed_at=_iso(ref - timedelta(days=30)),
        )
        db.record_repo_observation(
            conn,
            full_name="owner/repo",
            stars=500,
            discovered_via="trending",
            observed_at=_iso(ref - timedelta(hours=6)),
        )

        # 7-day window: only the recent observation, no velocity
        narrow = db.get_repo_velocity(
            conn, full_name="owner/repo", window_days=7, reference_at=_iso(ref)
        )
        assert narrow is None

        # 60-day window: both observations, velocity = 490
        wide = db.get_repo_velocity(
            conn, full_name="owner/repo", window_days=60, reference_at=_iso(ref)
        )
        assert wide is not None
        assert wide.star_delta == 490
    finally:
        conn.close()


def test_velocity_isolates_per_repo(temp_db):
    """Two repos in flight at the same time must not cross-contaminate."""
    conn = db.connect(temp_db)
    try:
        now = _now()
        for delta_days, stars in [(2, 100), (0, 250)]:
            db.record_repo_observation(
                conn,
                full_name="owner/alpha",
                stars=stars,
                discovered_via="trending",
                observed_at=_iso(now - timedelta(days=delta_days)),
            )
            db.record_repo_observation(
                conn,
                full_name="owner/beta",
                stars=stars * 10,
                discovered_via="search",
                observed_at=_iso(now - timedelta(days=delta_days)),
            )
        v_a = db.get_repo_velocity(conn, full_name="owner/alpha")
        v_b = db.get_repo_velocity(conn, full_name="owner/beta")
        assert v_a is not None and v_a.star_delta == 150
        assert v_b is not None and v_b.star_delta == 1500
    finally:
        conn.close()


def test_velocity_handles_missing_fork_data(temp_db):
    """Fork_delta is optional — if either endpoint is missing forks, return None."""
    conn = db.connect(temp_db)
    try:
        now = _now()
        db.record_repo_observation(
            conn,
            full_name="owner/repo",
            stars=100,
            discovered_via="trending",
            observed_at=_iso(now - timedelta(days=1)),
            # no forks recorded
        )
        db.record_repo_observation(
            conn,
            full_name="owner/repo",
            stars=200,
            forks=50,
            discovered_via="trending",
            observed_at=_iso(now),
        )
        v = db.get_repo_velocity(conn, full_name="owner/repo")
        assert v is not None
        assert v.star_delta == 100
        assert v.fork_delta is None  # one endpoint missing forks
    finally:
        conn.close()


def test_velocity_rejects_non_positive_window(temp_db):
    conn = db.connect(temp_db)
    try:
        with pytest.raises(ValueError, match="window_days"):
            db.get_repo_velocity(conn, full_name="o/r", window_days=0)
        with pytest.raises(ValueError, match="window_days"):
            db.get_repo_velocity(conn, full_name="o/r", window_days=-3)
    finally:
        conn.close()


def test_velocity_rejects_blank_full_name(temp_db):
    conn = db.connect(temp_db)
    try:
        with pytest.raises(ValueError, match="full_name"):
            db.get_repo_velocity(conn, full_name="")
        with pytest.raises(ValueError, match="full_name"):
            db.get_repo_velocity(conn, full_name="   ")
    finally:
        conn.close()


def test_velocity_rejects_naive_reference_at(temp_db):
    """Calling code must pass a tz-aware UTC timestamp; silently treating
    naive as UTC would produce subtle bugs across DST."""
    conn = db.connect(temp_db)
    try:
        with pytest.raises(ValueError, match="timezone-aware"):
            db.get_repo_velocity(conn, full_name="o/r", reference_at="2026-05-04T12:00:00")
    finally:
        conn.close()


def test_velocity_full_name_whitespace_matches_record_normalization(temp_db):
    """record_repo_observation strips whitespace; get_repo_velocity must do
    the same so callers don't have to normalize at every call site."""
    conn = db.connect(temp_db)
    try:
        now = _now()
        db.record_repo_observation(
            conn,
            full_name="owner/repo",
            stars=10,
            discovered_via="trending",
            observed_at=_iso(now - timedelta(days=1)),
        )
        db.record_repo_observation(
            conn,
            full_name="owner/repo",
            stars=20,
            discovered_via="trending",
            observed_at=_iso(now),
        )
        v = db.get_repo_velocity(conn, full_name="  owner/repo  ")
        assert v is not None
        assert v.star_delta == 10
    finally:
        conn.close()
