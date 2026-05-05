from __future__ import annotations

import pytest

from clawfeed_intel.normalize import (
    arxiv_dedup_key,
    canonicalize_url,
    content_hash,
    github_dedup_key,
    hn_dedup_key,
    reddit_dedup_key,
    sec_dedup_key,
)

# ── canonicalize_url ──────────────────────────────────────────────────────────


def test_canonicalize_lowercases_scheme_and_host_but_preserves_path_case():
    # Path is case-significant on most servers; only scheme/host fold.
    assert canonicalize_url("HTTPS://Example.COM/Foo") == "https://example.com/Foo"


def test_canonicalize_strips_www():
    assert canonicalize_url("https://www.example.com/foo") == "https://example.com/foo"


def test_canonicalize_strips_amp_subdomain():
    assert canonicalize_url("https://amp.example.com/foo") == "https://example.com/foo"


def test_canonicalize_keeps_other_subdomains():
    assert canonicalize_url("https://www2.example.com/foo") == "https://www2.example.com/foo"
    assert canonicalize_url("https://blog.example.com/foo") == "https://blog.example.com/foo"
    assert canonicalize_url("https://m.example.com/foo") == "https://m.example.com/foo"


def test_canonicalize_strips_default_ports():
    assert canonicalize_url("http://example.com:80/foo") == "http://example.com/foo"
    assert canonicalize_url("https://example.com:443/foo") == "https://example.com/foo"


def test_canonicalize_keeps_non_default_ports():
    assert canonicalize_url("https://example.com:8443/foo") == "https://example.com:8443/foo"


def test_canonicalize_strips_trailing_slash_on_non_root():
    assert canonicalize_url("https://example.com/foo/") == "https://example.com/foo"


def test_canonicalize_keeps_root_slash():
    assert canonicalize_url("https://example.com/") == "https://example.com/"


def test_canonicalize_normalizes_empty_path_to_root():
    assert canonicalize_url("https://example.com") == "https://example.com/"


def test_canonicalize_drops_fragment():
    assert canonicalize_url("https://example.com/foo#section") == "https://example.com/foo"


def test_canonicalize_strips_userinfo():
    assert canonicalize_url("https://user:pass@example.com/x") == "https://example.com/x"


def test_canonicalize_strips_outer_whitespace():
    assert canonicalize_url("   https://example.com/foo   ") == "https://example.com/foo"


def test_canonicalize_drops_utm_params():
    assert (
        canonicalize_url("https://example.com/x?utm_source=hn&utm_medium=email&id=42")
        == "https://example.com/x?id=42"
    )


@pytest.mark.parametrize("param", ["fbclid", "gclid", "msclkid", "yclid", "dclid", "_ga", "_gl"])
def test_canonicalize_drops_known_click_id_params(param: str):
    url = f"https://example.com/x?{param}=abc&id=42"
    assert canonicalize_url(url) == "https://example.com/x?id=42"


def test_canonicalize_drops_mailchimp_params():
    assert (
        canonicalize_url("https://example.com/x?mc_cid=abc&mc_eid=def&id=42")
        == "https://example.com/x?id=42"
    )


def test_canonicalize_drops_hubspot_prefixed_params():
    assert (
        canonicalize_url("https://example.com/x?_hsenc=foo&_hsmi=bar&id=42")
        == "https://example.com/x?id=42"
    )


def test_canonicalize_drops_share_and_ref():
    assert (
        canonicalize_url("https://example.com/x?ref=hn&share=t&id=42")
        == "https://example.com/x?id=42"
    )


def test_canonicalize_sorts_remaining_query():
    assert (
        canonicalize_url("https://example.com/x?b=2&a=1&c=3") == "https://example.com/x?a=1&b=2&c=3"
    )


def test_canonicalize_drops_trailing_question_mark_when_only_tracking():
    assert canonicalize_url("https://example.com/x?utm_source=hn") == "https://example.com/x"


def test_canonicalize_keeps_blank_values():
    assert (
        canonicalize_url("https://example.com/x?id=42&debug=")
        == "https://example.com/x?debug=&id=42"
    )


def test_canonicalize_strips_amp_path_suffix():
    assert canonicalize_url("https://example.com/article/amp") == "https://example.com/article"
    assert canonicalize_url("https://example.com/article/amp/") == "https://example.com/article"


def test_canonicalize_pass_through_for_non_http():
    assert canonicalize_url("mailto:hi@example.com") == "mailto:hi@example.com"
    assert canonicalize_url("tel:+15551234567") == "tel:+15551234567"
    assert canonicalize_url("file:///etc/hosts") == "file:///etc/hosts"


def test_canonicalize_rejects_empty():
    with pytest.raises(ValueError):
        canonicalize_url("")
    with pytest.raises(ValueError):
        canonicalize_url("   ")


def test_canonicalize_rejects_http_with_no_host():
    with pytest.raises(ValueError):
        canonicalize_url("https:///path-only")


def test_canonicalize_type_error_on_non_string():
    with pytest.raises(TypeError):
        canonicalize_url(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        canonicalize_url(42)  # type: ignore[arg-type]


def test_canonicalize_is_idempotent():
    url = "HTTPS://www.Example.com:443/foo/?utm_source=hn&b=2&a=1#frag"
    once = canonicalize_url(url)
    twice = canonicalize_url(once)
    assert once == twice


def test_canonicalize_folds_obvious_duplicate_forms():
    # All four URLs should canonicalize to the same string.
    base = "https://example.com/article"
    variants = [
        "https://www.example.com/article",
        "https://www.example.com/article/",
        "https://www.example.com/article#hash",
        "https://www.example.com/article?utm_source=hn",
    ]
    for v in variants:
        assert canonicalize_url(v) == base


# ── content_hash ──────────────────────────────────────────────────────────────


def test_content_hash_identical_inputs_match():
    assert content_hash("Title", "Body text") == content_hash("Title", "Body text")


def test_content_hash_whitespace_collapses():
    a = content_hash("Hello World", "first paragraph")
    b = content_hash("Hello   World", "first    paragraph")
    c = content_hash("  Hello World  ", "first paragraph\n\n")
    assert a == b == c


def test_content_hash_case_folds():
    assert content_hash("Title", "Body") == content_hash("TITLE", "body")


def test_content_hash_distinct_titles_differ():
    assert content_hash("Title A", "x") != content_hash("Title B", "x")


def test_content_hash_distinct_bodies_differ():
    assert content_hash("t", "alpha") != content_hash("t", "beta")


def test_content_hash_truncates_body_to_prefix():
    a = content_hash("t", "x" * 4000 + "tail-A")
    b = content_hash("t", "x" * 4000 + "tail-B")
    assert a == b, "bodies sharing the first 4000 chars must hash equal by default"


def test_content_hash_short_body_distinct_from_padded_long_body():
    short = content_hash("t", "abc")
    longer = content_hash("t", "abc" + "z" * 1000)
    assert short != longer


def test_content_hash_handles_none_inputs():
    assert content_hash("", "") == content_hash(None, None)


def test_content_hash_rejects_negative_prefix():
    with pytest.raises(ValueError):
        content_hash("t", "x", prefix_chars=-1)


def test_content_hash_returns_64_hex_chars():
    h = content_hash("t", "x")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ── per-source dedup keys ─────────────────────────────────────────────────────


def test_hn_dedup_key_coerces_int_to_str():
    assert hn_dedup_key(42135123) == "42135123"
    assert hn_dedup_key("42135123") == "42135123"


def test_github_dedup_key_lowercases_and_trims():
    assert github_dedup_key("Owner/Repo") == "owner/repo"
    assert github_dedup_key("  Owner/Repo  ") == "owner/repo"


def test_reddit_dedup_key_preserves_case():
    assert reddit_dedup_key("t3_abc123") == "t3_abc123"
    assert reddit_dedup_key("  t3_abc123  ") == "t3_abc123"


def test_arxiv_dedup_key_preserves_case():
    assert arxiv_dedup_key("2401.12345") == "2401.12345"
    assert arxiv_dedup_key("math.GT/0506203") == "math.GT/0506203"
    assert arxiv_dedup_key("  2401.12345  ") == "2401.12345"


def test_sec_dedup_key_preserves_format():
    assert sec_dedup_key("0001234567-26-000001") == "0001234567-26-000001"
    assert sec_dedup_key("  0001234567-26-000001  ") == "0001234567-26-000001"
