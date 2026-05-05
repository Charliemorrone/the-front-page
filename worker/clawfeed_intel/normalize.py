"""Pure utilities for URL canonicalization, content fingerprints, and per-source
dedup keys.

Functions here are deterministic and side-effect-free so fetchers can call
them freely and tests can pin behavior on inputs alone.

Two layers of dedup:

- **Level 1 (canonical URL)** — :func:`canonicalize_url` collapses tracking
  noise so the same article reached through different paths produces the same
  string. Stored in ``raw_items.canonical_url`` and used as the
  ``dedup_key`` for URL-keyed source types (RSS, websites, GDELT, …).
- **Level 2 (content fingerprint)** — :func:`content_hash` produces a stable
  SHA-256 of (title, body-prefix) so syndicated copies sharing the same body
  fold together even when their URLs disagree. Stored in
  ``raw_items.content_hash`` and used by clustering to spot duplicate coverage.

Per-source dedup-key helpers (``hn_dedup_key``, ``reddit_dedup_key``, …)
enforce a single canonical form for each source's natural id, so a fetcher
that observes the same item via two paths produces the same key both times.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# ── URL canonicalization ──────────────────────────────────────────────────────

# Exact-match tracking parameters. Lowercase comparison.
_TRACKING_PARAM_NAMES: frozenset[str] = frozenset(
    {
        # Facebook
        "fbclid",
        "fb_action_ids",
        "fb_action_types",
        "fb_ref",
        "fb_source",
        # Google / DoubleClick
        "gclid",
        "gclsrc",
        "dclid",
        "gad",
        "gad_source",
        "_ga",
        "_gl",
        # Microsoft / Bing
        "msclkid",
        # Yandex
        "yclid",
        "ymclid",
        # Mailchimp
        "mc_cid",
        "mc_eid",
        # HubSpot static names
        "__hsfp",
        "__hssc",
        "__hstc",
        # Marketo
        "mkt_tok",
        # Vero
        "vero_id",
        "vero_conv",
        # Omeda
        "oly_anon_id",
        "oly_enc_id",
        # Instagram share
        "igshid",
        # Twitter / generic referrer tags
        "ref",
        "ref_src",
        "ref_url",
        # Awin
        "awc",
        # Generic share / source tags
        "share",
        "sharer",
        "source",
        "embedded_cta",
        # Campaign IDs
        "cmpid",
        "cmp",
        "icid",
        "ncid",
        "mc_id",
        "campaign_id",
        # Other newsletter / CRM
        "nr_email_referer",
        "s_kwcid",
    }
)

# Prefix-match tracking parameters. Lowercase comparison.
_TRACKING_PARAM_PREFIXES: tuple[str, ...] = (
    "utm_",
    "mtm_",
    "pk_",
    "piwik_",
    "hsa_",
    "_hsenc",
    "_hsmi",
    "wt.",
)


def _is_tracking_param(name: str) -> bool:
    lname = name.lower()
    if lname in _TRACKING_PARAM_NAMES:
        return True
    return any(lname.startswith(prefix) for prefix in _TRACKING_PARAM_PREFIXES)


def canonicalize_url(url: str) -> str:
    """Return a canonical form of *url* suitable for cross-source dedup.

    Steps applied to ``http://`` and ``https://`` URLs:

    - lowercase scheme and host
    - strip ``www.`` and ``amp.`` host prefixes
    - drop userinfo (``user:pass@``)
    - strip default ports (80, 443)
    - drop the URL fragment
    - strip ``/amp`` (with or without trailing slash) from the path
    - strip the trailing slash from non-root paths
    - drop tracking query params (UTM, fbclid, gclid, share/ref, etc.)
    - sort remaining query params alphabetically

    Non-HTTP schemes (``mailto:``, ``tel:``, ``file:``, …) are returned with
    surrounding whitespace stripped but otherwise unchanged — we don't dedup
    them, but pass-through avoids losing the value.

    Raises:
        TypeError: if *url* is not a ``str``.
        ValueError: if *url* is empty or whitespace-only, or is HTTP-shaped
            but missing a host.
    """
    if not isinstance(url, str):
        raise TypeError(f"url must be str, got {type(url).__name__}")

    stripped = url.strip()
    if not stripped:
        raise ValueError("url is empty")

    parts = urlsplit(stripped)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        return stripped

    host = (parts.hostname or "").lower()
    if not host:
        raise ValueError(f"url missing host: {url!r}")
    if host.startswith("www."):
        host = host[4:]
    elif host.startswith("amp."):
        host = host[4:]

    netloc = host
    port = parts.port  # may raise ValueError on malformed input; that's fine
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{host}:{port}"

    path = parts.path or "/"
    if path.endswith("/amp/"):
        path = path[:-5] or "/"
    elif path.endswith("/amp"):
        path = path[:-4] or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    if parts.query:
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        kept = [(name, value) for name, value in pairs if not _is_tracking_param(name)]
        kept.sort()
        query = urlencode(kept, doseq=True)
    else:
        query = ""

    return urlunsplit((scheme, netloc, path, query, ""))


# ── Content fingerprint ───────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text_for_hash(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def content_hash(title: str | None, text: str | None, *, prefix_chars: int = 4000) -> str:
    """Stable SHA-256 fingerprint of ``(title, body-prefix)``.

    Whitespace runs collapse, leading/trailing whitespace strips, casing
    folds. Punctuation is preserved because syndicated copies typically keep
    it identical, and stripping it would conflate distinct items more often
    than it would unify true duplicates.

    Args:
        title: item title; ``None`` is treated as empty.
        text: item body; ``None`` is treated as empty.
        prefix_chars: how many characters of the body participate in the
            hash. The default (4000) matches the architecture doc and keeps
            the fingerprint stable for articles whose tails differ (appended
            share widgets, recommendation footers, comment counts).

    Raises:
        ValueError: if *prefix_chars* is negative.
    """
    if prefix_chars < 0:
        raise ValueError("prefix_chars must be >= 0")
    title_part = _normalize_text_for_hash(title or "")
    body = (text or "")[:prefix_chars]
    body_part = _normalize_text_for_hash(body)
    blob = f"{title_part}\n{body_part}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── Per-source dedup keys ─────────────────────────────────────────────────────


def hn_dedup_key(item_id: int | str) -> str:
    """Hacker News story or comment id."""
    return str(item_id).strip()


def reddit_dedup_key(fullname: str) -> str:
    """Reddit fullname like ``t3_abc123`` (post) or ``t1_def456`` (comment).

    Reddit fullnames are case-sensitive; we only strip surrounding whitespace.
    """
    return fullname.strip()


def github_dedup_key(full_name: str) -> str:
    """GitHub repo full name (``owner/repo``); folded to lowercase.

    GitHub treats owner and repo as case-insensitive for routing, so two
    sightings that differ only in casing are the same repository.
    """
    return full_name.strip().lower()


def arxiv_dedup_key(arxiv_id: str) -> str:
    """arXiv identifier (``2401.12345`` or legacy ``math.GT/0506203``).

    Identifiers are case-sensitive in their subject prefix, so we trim only.
    """
    return arxiv_id.strip()


def sec_dedup_key(accession: str) -> str:
    """SEC EDGAR accession number (``0001234567-26-000001`` style)."""
    return accession.strip()
