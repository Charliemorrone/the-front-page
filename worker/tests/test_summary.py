"""Tests for the cluster-summary pure layer.

Step 10a covers schemas + pure helpers + DB write. The async
``summarize_clusters`` orchestration lands in 10b and is tested against
a real :class:`LLMClient` backed by :class:`httpx.MockTransport`. The
pure layer here is fixture-testable without HTTP / DB / LLM.
"""

from __future__ import annotations

import pytest

from clawfeed_intel.llm.schemas import ClusterSummaryPayload
from clawfeed_intel.pipeline.summary import (
    PROMPT_VERSION,
    SummaryCluster,
    SummaryMember,
    build_summary_messages,
    parse_summary,
)
from clawfeed_intel.sources import CategoryPlan


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _category(
    name: str = "startup_funding",
    *,
    description: str = "Funding rounds and acquisitions.",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> CategoryPlan:
    return CategoryPlan(
        name=name,
        description=description,
        include_rules=include or ["announced funding rounds"],
        exclude_rules=exclude or ["recirculated old rounds"],
    )


def _member(
    *,
    title: str = "Anthropic raises Series E",
    canonical_url: str = "https://techcrunch.com/anthropic-series-e",
    excerpt: str = "Anthropic announced a Series E.",
    content: str = "",
    author: str = "",
    published_at: str | None = None,
) -> SummaryMember:
    return SummaryMember(
        title=title,
        canonical_url=canonical_url,
        excerpt=excerpt,
        content=content,
        author=author,
        published_at=published_at,
    )


def _cluster(
    *,
    cluster_id: int = 1,
    title: str = "Anthropic Series E",
    category: str | None = "startup_funding",
    members: tuple[SummaryMember, ...] = (),
) -> SummaryCluster:
    return SummaryCluster(
        cluster_id=cluster_id,
        title=title,
        category=category,
        members=members or (_member(),),
    )


# ── PROMPT_VERSION ────────────────────────────────────────────────────────────


def test_prompt_version_is_pinned() -> None:
    """The prompt-version slug appears in every ``item_summaries`` row.

    Bump on behavioral changes so audit rows reflect which generation
    of the prompt produced them.
    """
    assert PROMPT_VERSION == "summary.v1"


# ── Message shape ─────────────────────────────────────────────────────────────


def test_build_messages_returns_system_then_user() -> None:
    messages = build_summary_messages(_cluster(), _category())
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_messages_raises_on_empty_members() -> None:
    """A memberless cluster has nothing to summarize.

    Defensive: ``iter_kept_clusters_with_members`` (10b) will filter
    these out in SQL, but the pure helper should fail loudly rather
    than dispatch a wasted LLM call.
    """
    with pytest.raises(ValueError, match="no members"):
        build_summary_messages(
            SummaryCluster(cluster_id=42, title="empty", category="x", members=()),
            _category(),
        )


# ── System message ────────────────────────────────────────────────────────────


def test_system_message_states_grounding_rule() -> None:
    """The grounding rule is load-bearing — the prompt's job is to
    prevent the model from inventing facts the final composer would
    then propagate into the brief.
    """
    messages = build_summary_messages(_cluster(), _category())
    system = messages[0]["content"]
    assert "Grounding policy" in system
    assert "MUST be present in the supplied source items" in system
    assert "Do not invent" in system


def test_system_message_states_citation_rule() -> None:
    """``source_urls`` must come from the input — the brief's links
    section depends on this rule holding.
    """
    messages = build_summary_messages(_cluster(), _category())
    system = messages[0]["content"]
    assert "source_urls" in system
    assert "list the cluster's source URLs" in system


def test_system_message_describes_response_shape() -> None:
    messages = build_summary_messages(_cluster(), _category())
    system = messages[0]["content"]
    for field in (
        "headline",
        "summary",
        "why_it_matters",
        "entities",
        "key_facts",
        "caveats",
        "source_urls",
        "confidence",
    ):
        assert field in system, f"system message missing field {field!r}"


def test_system_message_demands_pure_json() -> None:
    """JSON-only / no-markdown-fencing wording mirrors the relevance
    filter's hardening; without it, vMLX returns markdown-wrapped
    responses under load and the bounded-repair retry fires.
    """
    messages = build_summary_messages(_cluster(), _category())
    system = messages[0]["content"]
    assert "Reply with valid JSON only" in system
    assert "no markdown fencing" in system
    assert "Begin your response with `{`" in system


def test_system_message_includes_category_context() -> None:
    messages = build_summary_messages(
        _cluster(category="startup_funding"),
        _category(
            "startup_funding",
            description="Funding rounds, new fund closes, notable acquisitions.",
            include=["announced rounds", "Form D filings"],
            exclude=["opinion posts"],
        ),
    )
    system = messages[0]["content"]
    assert "Category context:" in system
    assert "name: startup_funding" in system
    assert "Funding rounds, new fund closes" in system
    assert "announced rounds" in system
    assert "Form D filings" in system
    assert "opinion posts" in system


def test_system_message_omits_category_when_none() -> None:
    """A cluster with no category falls back to a generic framing —
    the relevance filter's permissive schema allows ``category=None``
    and we mirror that posture here. Verified end-to-end against the
    9c relaxation.
    """
    messages = build_summary_messages(
        _cluster(category=None),
        None,
    )
    system = messages[0]["content"]
    assert "Category context:" not in system
    # The grounding policy still applies regardless of category.
    assert "Grounding policy" in system


# ── User message ──────────────────────────────────────────────────────────────


def test_user_message_starts_with_cluster_header() -> None:
    messages = build_summary_messages(_cluster(title="Anthropic Series E"), _category())
    user = messages[1]["content"]
    assert user.startswith("Cluster: Anthropic Series E")


def test_user_message_falls_back_to_untitled() -> None:
    cluster = _cluster(title="", members=(_member(title="m"),))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "Cluster: (untitled)" in user


def test_user_message_states_category_when_present() -> None:
    cluster = _cluster(category="ai_research")
    messages = build_summary_messages(cluster, _category("ai_research"))
    user = messages[1]["content"]
    assert "Filed under category: ai_research" in user


def test_user_message_lists_all_members_in_order() -> None:
    members = (
        _member(title="A", canonical_url="https://a.example/post"),
        _member(title="B", canonical_url="https://b.example/post"),
        _member(title="C", canonical_url="https://c.example/post"),
    )
    cluster = _cluster(members=members)
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "[1] A" in user
    assert "[2] B" in user
    assert "[3] C" in user
    assert user.index("[1] A") < user.index("[2] B") < user.index("[3] C")


def test_user_message_surfaces_member_url_and_excerpt() -> None:
    member = _member(
        title="Article",
        canonical_url="https://example.com/post",
        excerpt="The lead paragraph mentions the round size.",
    )
    cluster = _cluster(members=(member,))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "url: https://example.com/post" in user
    assert "The lead paragraph mentions the round size." in user


def test_user_message_prefers_content_over_excerpt() -> None:
    """``content`` is the trafilatura/full-body field; ``excerpt`` is
    the fetcher's ~320-char prefix. When both are present the model
    should see the richer body.
    """
    member = _member(
        title="Article",
        canonical_url="https://example.com/post",
        excerpt="Lead paragraph excerpt.",
        content="Full extracted body with multiple sentences. Second sentence.",
    )
    cluster = _cluster(members=(member,))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "Full extracted body" in user
    assert "Lead paragraph excerpt." not in user


def test_user_message_omits_body_when_both_empty() -> None:
    """GDELT ArtList carries no body and only a title-prefix excerpt;
    the prompt should still render cleanly. Verified against the
    fetcher-contract docstring.
    """
    member = _member(
        title="Headline only",
        canonical_url="https://news.example/article",
        excerpt="",
        content="",
    )
    cluster = _cluster(members=(member,))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "Headline only" in user
    assert "body:" not in user


def test_user_message_surfaces_author_and_published_at_when_set() -> None:
    member = _member(
        title="Paper",
        canonical_url="https://arxiv.org/abs/2405.12345",
        author="A. Researcher, B. Coauthor",
        published_at="2026-05-13T00:00:00+00:00",
    )
    cluster = _cluster(members=(member,))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "author: A. Researcher, B. Coauthor" in user
    assert "published_at: 2026-05-13T00:00:00+00:00" in user


def test_user_message_omits_author_and_published_at_when_empty() -> None:
    cluster = _cluster(
        members=(_member(author="", published_at=None),),
    )
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "author:" not in user
    assert "published_at:" not in user


def test_user_message_states_member_count() -> None:
    """L2/L3-folded clusters surface every member; the count line
    nudges the model to cite cross-source corroboration in the
    summary.
    """
    members = (
        _member(title="Coverage A", canonical_url="https://a.example/"),
        _member(title="Coverage B", canonical_url="https://b.example/"),
        _member(title="Coverage C", canonical_url="https://c.example/"),
    )
    cluster = _cluster(members=members)
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "Source items (3):" in user


def test_user_message_indents_multiline_body() -> None:
    """Body lines nest under ``body:`` so the prompt stays parseable
    if a future debugger eyeballs it.
    """
    member = _member(
        content="First line.\nSecond line.\n  whitespace-only line.\n",
    )
    cluster = _cluster(members=(member,))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "      First line." in user
    assert "      Second line." in user


# ── Parse layer ───────────────────────────────────────────────────────────────


def test_parse_summary_returns_payload() -> None:
    payload = ClusterSummaryPayload(headline="Headline", summary="Summary.")
    assert parse_summary(payload) is payload


def test_parse_summary_rejects_wrong_type() -> None:
    class NotAPayload:
        headline = "h"
        summary = "s"

    with pytest.raises(TypeError, match="ClusterSummaryPayload"):
        parse_summary(NotAPayload())  # type: ignore[arg-type]
