"""Tests for the SSRF guard in ``fetchers.http.validate_safe_url``.

The helper mirrors the Node ``assertSafeFetchUrl`` in ``src/server.mjs``:
http(s)-only scheme, DNS-resolve the hostname, reject if any resolved
address is in private/loopback/link-local/multicast/reserved ranges, and
short-circuit IP literals through the same range check.

DNS is mocked via ``monkeypatch`` on ``socket.getaddrinfo`` — no live
network in CI. Each ``_addrinfo`` helper builds the tuple shape
``getaddrinfo`` actually returns so the helper's sockaddr extraction is
exercised end-to-end.
"""

from __future__ import annotations

import socket

import pytest

from clawfeed_intel.fetchers.http import UnsafeUrlError, validate_safe_url


def _addrinfo(addr: str, *, family: int | None = None) -> tuple:
    """Build one ``getaddrinfo`` tuple: ``(family, type, proto, canon, sa)``.

    ``family`` is auto-picked from the address shape (``:`` → AF_INET6,
    else AF_INET). The sockaddr's first element is the address — that's
    all the helper inspects.
    """
    if family is None:
        family = socket.AF_INET6 if ":" in addr else socket.AF_INET
    sa: tuple = (addr, 0, 0, 0) if family == socket.AF_INET6 else (addr, 0)
    return (family, socket.SOCK_STREAM, 0, "", sa)


@pytest.fixture
def mock_dns(monkeypatch):
    """Replace ``socket.getaddrinfo`` with a function returning fixed addresses.

    Returns a setter: pass either a list of address strings (auto-wrapped
    via ``_addrinfo``), a callable for full control, or an exception
    instance (raised when the helper looks the host up).
    """

    def _set(value):
        if isinstance(value, BaseException):

            def fn(*_args, **_kwargs):
                raise value

        elif callable(value):
            fn = value
        else:
            wrapped = [_addrinfo(a) if isinstance(a, str) else a for a in value]

            def fn(*_args, **_kwargs):
                return wrapped

        monkeypatch.setattr(socket, "getaddrinfo", fn)

    return _set


# ── public host happy path ────────────────────────────────────────────────────


async def test_public_host_passes(mock_dns):
    mock_dns(["93.184.216.34"])  # example.com
    # Returns None on success — no exception is the contract.
    assert await validate_safe_url("https://example.com/path?q=1") is None


async def test_public_ipv6_host_passes(mock_dns):
    mock_dns(["2606:2800:220:1:248:1893:25c8:1946"])
    assert await validate_safe_url("https://example.com/") is None


# ── scheme rejection ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/",
        "ftp://example.com/",
        "javascript:alert(1)",
        "data:text/plain,hello",
        "ws://example.com/",
        "wss://example.com/",
        "ldap://example.com/",
    ],
)
async def test_non_http_schemes_rejected(url, mock_dns):
    # DNS shouldn't even be consulted for the wrong scheme — make sure of
    # that by raising if it is.
    mock_dns(socket.gaierror("dns should not be called"))
    with pytest.raises(UnsafeUrlError, match="scheme"):
        await validate_safe_url(url)


async def test_scheme_is_case_insensitive(mock_dns):
    mock_dns(["93.184.216.34"])
    # Standard-library urlsplit lowercases the scheme; this just pins that
    # the helper doesn't accidentally compare against the raw input.
    assert await validate_safe_url("HTTPS://example.com/") is None


# ── localhost name rejection ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://localhost:8080/admin",
        "https://something.localhost/",
        "https://api.localhost/v1",
    ],
)
async def test_localhost_names_rejected(url, mock_dns):
    # DNS is not consulted for these — verified by raising if it is.
    mock_dns(socket.gaierror("dns should not be called"))
    with pytest.raises(UnsafeUrlError, match="blocked host"):
        await validate_safe_url(url)


# ── IPv4 ranges (literal hosts, no DNS) ───────────────────────────────────────


@pytest.mark.parametrize(
    "ip",
    [
        # 0.0.0.0/8 unspecified / "this network"
        "0.0.0.0",
        "0.1.2.3",
        # 10.0.0.0/8 private
        "10.0.0.1",
        "10.255.255.254",
        # 127.0.0.0/8 loopback
        "127.0.0.1",
        "127.1.2.3",
        # 169.254.0.0/16 link-local
        "169.254.169.254",  # AWS/GCP metadata service
        # 172.16.0.0/12 private
        "172.16.0.1",
        "172.20.10.5",
        "172.31.255.254",
        # 192.168.0.0/16 private
        "192.168.0.1",
        "192.168.1.1",
        "192.168.255.254",
        # 224.0.0.0/4 multicast
        "224.0.0.1",
        "239.255.255.255",
        # 240.0.0.0/4 reserved
        "240.0.0.1",
        "255.255.255.255",
    ],
)
async def test_ipv4_literal_blocked(ip, mock_dns):
    mock_dns(socket.gaierror("dns should not be called"))
    with pytest.raises(UnsafeUrlError):
        await validate_safe_url(f"http://{ip}/")


@pytest.mark.parametrize(
    "ip",
    [
        # public ranges from the architecture doc + a couple of CDN-ish ones
        "1.1.1.1",
        "8.8.8.8",
        "93.184.216.34",
        "151.101.1.69",
    ],
)
async def test_ipv4_public_literal_passes(ip, mock_dns):
    mock_dns(socket.gaierror("dns should not be called"))
    # An IP literal short-circuits DNS entirely; it goes through directly.
    assert await validate_safe_url(f"http://{ip}/") is None


# ── 172.16/12 boundaries: nearby publics must NOT be blocked ─────────────────


async def test_ipv4_just_below_172_16_block_passes(mock_dns):
    mock_dns(socket.gaierror("dns should not be called"))
    # 172.15.x.x sits just outside the private block.
    assert await validate_safe_url("http://172.15.0.1/") is None


async def test_ipv4_just_above_172_31_block_passes(mock_dns):
    mock_dns(socket.gaierror("dns should not be called"))
    assert await validate_safe_url("http://172.32.0.1/") is None


# ── IPv6 ranges (literal hosts, no DNS) ───────────────────────────────────────


@pytest.mark.parametrize(
    "ip",
    [
        "::1",  # loopback
        "fc00::1",  # ULA / private (fc00::/7)
        "fd00::1",  # ULA / private
        "fe80::1",  # link-local (fe80::/10)
        "fe80::a00:27ff:fe1c:3a4d",
        "::ffff:127.0.0.1",  # v4-mapped loopback
        "::ffff:192.168.1.1",  # v4-mapped private
        "ff02::1",  # multicast
    ],
)
async def test_ipv6_literal_blocked(ip, mock_dns):
    mock_dns(socket.gaierror("dns should not be called"))
    with pytest.raises(UnsafeUrlError):
        await validate_safe_url(f"http://[{ip}]/")


async def test_ipv6_public_literal_passes(mock_dns):
    mock_dns(socket.gaierror("dns should not be called"))
    # 2606:2800::/32 (example.com range) is global unicast.
    assert await validate_safe_url("http://[2606:2800:220:1::1]/") is None


# ── DNS-resolved addresses ────────────────────────────────────────────────────


async def test_dns_resolves_to_loopback_blocked(mock_dns):
    mock_dns(["127.0.0.1"])
    with pytest.raises(UnsafeUrlError, match="blocked host"):
        await validate_safe_url("http://attacker-controlled.example/")


async def test_dns_resolves_to_metadata_service_blocked(mock_dns):
    mock_dns(["169.254.169.254"])
    with pytest.raises(UnsafeUrlError):
        await validate_safe_url("http://metadata.attacker.example/")


async def test_dns_returns_mixed_public_and_private_rejected(mock_dns):
    """A name resolving to both a public and a private address is rejected.

    Defense against DNS-rebinding-style attacks where a host briefly returns
    a public address (passing the check) and then a private one for the
    actual fetch. The conservative rule — reject if *any* address is unsafe
    — also matches the Node helper.
    """
    mock_dns(["93.184.216.34", "127.0.0.1"])
    with pytest.raises(UnsafeUrlError):
        await validate_safe_url("http://double-a.example/")


async def test_dns_returns_only_public_passes(mock_dns):
    mock_dns(["93.184.216.34", "151.101.1.69"])
    assert await validate_safe_url("http://multi-a.example/") is None


async def test_dns_resolves_to_ipv6_ula_blocked(mock_dns):
    mock_dns(["fc00::1"])
    with pytest.raises(UnsafeUrlError):
        await validate_safe_url("http://ula-host.example/")


async def test_dns_resolution_failure_rejected(mock_dns):
    mock_dns(socket.gaierror("Name or service not known"))
    with pytest.raises(UnsafeUrlError, match="dns lookup failed"):
        await validate_safe_url("http://nonexistent-host.example/")


async def test_dns_returns_empty_rejected(mock_dns):
    mock_dns([])
    with pytest.raises(UnsafeUrlError, match="no addresses"):
        await validate_safe_url("http://no-records.example/")


# ── malformed URLs ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not-a-url",
        "http://",
        "://example.com/",
        "http:///path-only",
    ],
)
async def test_malformed_urls_rejected(url, mock_dns):
    mock_dns(socket.gaierror("dns should not be called"))
    with pytest.raises(UnsafeUrlError):
        await validate_safe_url(url)


# ── async-safe DNS lookup ─────────────────────────────────────────────────────


async def test_dns_lookup_runs_off_event_loop(monkeypatch):
    """The blocking DNS lookup must go through ``asyncio.to_thread``.

    Pin the boundary by asserting the call happens through a
    ``run_in_executor``/``to_thread`` indirection rather than synchronously.
    The simpler load-bearing assertion: a lookup whose synchronous version
    sleeps for a beat shouldn't block other awaitables. We verify by
    arranging two concurrent calls and asserting both complete — if the
    helper called ``socket.getaddrinfo`` synchronously, the second call's
    coroutine couldn't even start until the first completed.
    """
    import asyncio
    import time

    def slow_lookup(*_args, **_kwargs):
        time.sleep(0.05)  # 50ms
        return [_addrinfo("93.184.216.34")]

    monkeypatch.setattr(socket, "getaddrinfo", slow_lookup)

    start = time.monotonic()
    await asyncio.gather(
        validate_safe_url("http://a.example/"),
        validate_safe_url("http://b.example/"),
    )
    elapsed = time.monotonic() - start

    # Two 50ms blocking lookups serialized → ~100ms; offloaded → ~50ms.
    # 90ms threshold gives generous headroom on a busy CI box.
    assert elapsed < 0.09, f"lookups appear serialized: {elapsed * 1000:.1f}ms"


# ── UnsafeUrlError shape ──────────────────────────────────────────────────────


def test_unsafe_url_error_is_a_value_error():
    """Subclassing ``ValueError`` keeps callers that broadly catch
    ``ValueError`` (URL parse failures, etc.) degrading naturally."""
    assert issubclass(UnsafeUrlError, ValueError)
