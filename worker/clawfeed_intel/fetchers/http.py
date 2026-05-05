"""Shared HTTP client config for fetchers.

Centralizes user-agent, timeouts, connection limits, and request headers so
every fetcher inherits identical, polite behaviour. Per-fetcher tuning
(e.g. SEC EDGAR's ≤10 req/s rate limit, Reddit's conservative cadence) layers
on top inside individual fetcher modules.

The contact email in the User-Agent comes from ``CLAWFEED_CONTACT_EMAIL``.
SEC EDGAR explicitly requires a contact-bearing UA, and Reddit's API guidance
asks for one too — declaring it on every fetcher keeps the policy uniform.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from .. import __version__

CONTACT_EMAIL_ENV = "CLAWFEED_CONTACT_EMAIL"
DEFAULT_CONTACT = "noreply@clawfeed.local"

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)
DEFAULT_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=8)


def user_agent() -> str:
    contact = os.environ.get(CONTACT_EMAIL_ENV, DEFAULT_CONTACT)
    return f"ClawFeed-Intel/{__version__} (+contact: {contact})"


def default_headers() -> dict[str, str]:
    return {
        "User-Agent": user_agent(),
        "Accept": (
            "application/atom+xml, application/rss+xml, "
            "application/xml;q=0.9, text/html;q=0.8, */*;q=0.5"
        ),
        "Accept-Encoding": "gzip, deflate",
    }


@asynccontextmanager
async def build_client(*, follow_redirects: bool = True) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an ``httpx.AsyncClient`` preconfigured for fetcher use.

    Each fetcher call constructs its own client. SQLite write contention is
    the project's bottleneck, not HTTP socket reuse — keeping the client
    scoped per-task makes failure-mode reasoning simpler.
    """
    async with httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        limits=DEFAULT_LIMITS,
        headers=default_headers(),
        follow_redirects=follow_redirects,
    ) as client:
        yield client
