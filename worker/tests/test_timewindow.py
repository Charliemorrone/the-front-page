from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clawfeed_intel.timewindow import parse_window, to_iso, window_for


def test_parse_window_hours():
    assert parse_window("24h") == timedelta(hours=24)
    assert parse_window("1h") == timedelta(hours=1)


def test_parse_window_days():
    assert parse_window("7d") == timedelta(days=7)


def test_parse_window_strips_whitespace():
    assert parse_window("  24h  ") == timedelta(hours=24)


@pytest.mark.parametrize(
    "bad",
    ["", "h", "24", "24x", "24hh", "0h", "-1h", "1.5h", "24H", "7D"],
)
def test_parse_window_invalid(bad: str):
    with pytest.raises(ValueError):
        parse_window(bad)


def test_window_for_uses_now():
    fixed = datetime(2026, 5, 4, 6, 15, tzinfo=timezone.utc)
    start, end = window_for("24h", now=fixed)
    assert end == "2026-05-04T06:15:00+00:00"
    assert start == "2026-05-03T06:15:00+00:00"


def test_window_for_seven_days():
    fixed = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    start, end = window_for("7d", now=fixed)
    assert end == "2026-05-04T00:00:00+00:00"
    assert start == "2026-04-27T00:00:00+00:00"


def test_window_for_naive_now_rejected():
    with pytest.raises(ValueError):
        window_for("24h", now=datetime(2026, 5, 4, 6, 15))


def test_to_iso_naive_rejected():
    with pytest.raises(ValueError):
        to_iso(datetime(2026, 5, 4))


def test_to_iso_normalizes_to_utc():
    pacific = timezone(timedelta(hours=-7))
    dt = datetime(2026, 5, 4, 0, 0, tzinfo=pacific)
    assert to_iso(dt) == "2026-05-04T07:00:00+00:00"
