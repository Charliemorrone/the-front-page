"""Fetcher harness for the daily brief.

Each source family (RSS, arXiv, HN, Reddit, GDELT, SEC EDGAR, GitHub,
websites) supplies one ``FetcherCallable`` registered under its task ``kind``.
The :func:`run_fetch_stage` runner dispatches resolved tasks from the source
plan to those callables, persists items via :func:`db.upsert_raw_item`, and
records per-task outcomes against ``Coverage`` and ``source_fetch_state``.

Concrete fetcher modules land in subsequent steps; this package ships only
the contract and the runner so all eight can plug in cleanly.
"""

from .base import (
    FETCHER_REGISTRY,
    FetchedItem,
    FetcherCallable,
    FetchOutcome,
)
from .runner import run_fetch_stage

# Importing concrete fetcher modules registers them in FETCHER_REGISTRY.
# Add new fetchers here as their steps land.
from . import arxiv as _arxiv  # noqa: F401  registers kind="arxiv"
from . import gdelt as _gdelt  # noqa: F401  registers kind="gdelt"
from . import hn as _hn  # noqa: F401  registers kind="hn"
from . import rss as _rss  # noqa: F401  registers kind="rss"
from . import sec as _sec  # noqa: F401  registers kind="sec_edgar"

__all__ = [
    "FETCHER_REGISTRY",
    "FetchedItem",
    "FetcherCallable",
    "FetchOutcome",
    "run_fetch_stage",
]
