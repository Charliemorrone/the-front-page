"""Run-window parsing and ISO timestamp helpers.

The CLI accepts windows like ``24h`` or ``7d``. All persisted timestamps are
timezone-aware UTC ISO 8601 (``YYYY-MM-DDTHH:MM:SS+00:00``).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_WINDOW_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[hd])$")


def parse_window(spec: str) -> timedelta:
    """Parse a window spec like ``24h`` or ``7d`` into a :class:`timedelta`.

    Raises :class:`ValueError` for malformed input or non-positive values.
    """
    cleaned = spec.strip()
    match = _WINDOW_RE.fullmatch(cleaned)
    if match is None:
        raise ValueError(f"invalid window {spec!r}; expected forms like '24h' or '7d'")
    value = int(match.group("value"))
    if value <= 0:
        raise ValueError(f"window must be positive, got {spec!r}")
    if match.group("unit") == "h":
        return timedelta(hours=value)
    return timedelta(days=value)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    """Serialize a timezone-aware datetime as ``YYYY-MM-DDTHH:MM:SS+00:00``."""
    if dt.tzinfo is None:
        raise ValueError("refusing to serialize a naive datetime; pass a tz-aware value")
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def window_for(spec: str, *, now: datetime | None = None) -> tuple[str, str]:
    """Return ``(window_start_iso, window_end_iso)`` ending at *now* (UTC)."""
    end = now if now is not None else now_utc()
    if end.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    span = parse_window(spec)
    start = end - span
    return to_iso(start), to_iso(end)
