"""Shared HTTP client config for fetchers.

Centralizes user-agent, timeouts, connection limits, and request headers so
every fetcher inherits identical, polite behaviour. Per-fetcher tuning
(e.g. SEC EDGAR's ≤10 req/s rate limit, Reddit's conservative cadence) layers
on top inside individual fetcher modules.

The contact email in the User-Agent comes from ``CLAWFEED_CONTACT_EMAIL``.
SEC EDGAR explicitly requires a contact-bearing UA, and Reddit's API guidance
asks for one too — declaring it on every fetcher keeps the policy uniform.

This module also exposes :func:`validate_safe_url`, the worker-side SSRF
guard. It mirrors the Node ``assertSafeFetchUrl`` helper in ``src/server.mjs``:
http(s)-only scheme, DNS-resolve the hostname, reject if any resolved address
is private/loopback/link-local/multicast/reserved. The genuinely-exposed
surface is the trafilatura article-fetch path inside the RSS fetcher, which
follows links extracted from third-party feed bodies; the top-level
RSS/website fetchers also call it as defense-in-depth against DNS-rebinding
or pasted-from-the-web URLs.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx

from .. import __version__

log = logging.getLogger(__name__)

CONTACT_EMAIL_ENV = "CLAWFEED_CONTACT_EMAIL"
DEFAULT_CONTACT = "noreply@clawfeed.local"

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)
DEFAULT_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=8)

ALLOWED_SCHEMES = frozenset({"http", "https"})


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


class UnsafeUrlError(ValueError):
    """Raised by :func:`validate_safe_url` for URLs that fail the SSRF guard.

    Subclasses ``ValueError`` so callers that already catch ``ValueError`` on
    URL parsing degrade naturally; isinstance-discriminated when callers want
    SSRF-specific handling (the trafilatura path logs+swallows it; top-level
    fetchers let it propagate so the runner records ``failed``).
    """


def _is_unsafe_ip(addr: str) -> bool:
    """Return True if ``addr`` is in any blocked range.

    Mirrors the Node ``isPrivateOrSpecialIp`` helper: private (RFC1918 / ULA),
    loopback, link-local, multicast, reserved, and unspecified addresses are
    all rejected. ``ipaddress`` already encodes these ranges as predicates;
    we don't reimplement the bit math.
    """
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        # Anything ``getaddrinfo`` returns that ``ipaddress`` can't parse is
        # treated as unsafe — better to fail closed than to fetch something
        # we can't classify.
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


async def validate_safe_url(url: str) -> None:
    """Raise :class:`UnsafeUrlError` if ``url`` would let us SSRF a private host.

    Behavior matches the Node ``assertSafeFetchUrl`` helper in
    ``src/server.mjs``:

    - Only ``http`` and ``https`` schemes are allowed.
    - ``localhost`` and any ``*.localhost`` hostname is rejected outright.
    - IP-literal hostnames are checked directly via :mod:`ipaddress`.
    - DNS-resolved hostnames are rejected if *any* returned address falls in
      the private/loopback/link-local/multicast/reserved ranges (so a name
      resolving to both a public and a private address is still rejected —
      the conservative bound).

    The DNS lookup is offloaded with :func:`asyncio.to_thread` because
    :func:`socket.getaddrinfo` is blocking; matches the worker's existing
    use of ``asyncio.to_thread`` for trafilatura's CPU-bound parse.
    """
    try:
        parts = urlsplit(url)
    except (TypeError, ValueError) as exc:
        raise UnsafeUrlError(f"unparseable url: {exc}") from exc

    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"disallowed url scheme: {scheme!r}")

    try:
        host = (parts.hostname or "").lower()
    except ValueError as exc:
        # urlsplit can raise on malformed IPv6 in the netloc.
        raise UnsafeUrlError(f"malformed url host: {exc}") from exc
    if not host:
        raise UnsafeUrlError("url has no host")
    if host == "localhost" or host.endswith(".localhost"):
        raise UnsafeUrlError(f"blocked host: {host!r}")

    # IP-literal short circuit — no DNS to do.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_unsafe_ip(str(literal)):
            raise UnsafeUrlError(f"blocked ip literal: {host!r}")
        return

    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None, 0, socket.SOCK_STREAM)
    except (socket.gaierror, OSError) as exc:
        raise UnsafeUrlError(f"dns lookup failed for {host!r}: {exc}") from exc

    if not infos:
        raise UnsafeUrlError(f"dns returned no addresses for {host!r}")

    addresses = {info[4][0] for info in infos if info and info[4]}
    if not addresses:
        raise UnsafeUrlError(f"dns returned no usable addresses for {host!r}")
    if any(_is_unsafe_ip(addr) for addr in addresses):
        raise UnsafeUrlError(f"blocked host {host!r}: resolves to private/special address")


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
