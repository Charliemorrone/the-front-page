"""Tests for the pure layer of the relevance filter (step 9a).

Async orchestration lands in step 9b; here we only cover prompt
construction and the count-mismatch guard that protects positional
verdict assignment.
"""

from __future__ import annotations

import pytest

from clawfeed_intel.llm import RelevanceBatchResponse, RelevanceVerdict
from clawfeed_intel.pipeline.relevance import (
    PROMPT_VERSION,
    RelevanceCluster,
    build_relevance_messages,
    parse_relevance_verdicts,
)
from clawfeed_intel.sources import CategoryPlan


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _category(
    name: str,
    *,
    description: str = "",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> CategoryPlan:
    return CategoryPlan(
        name=name,
        description=description,
        include_rules=include or [],
        exclude_rules=exclude or [],
    )


def _cluster(
    cluster_id: int,
    *,
    title: str = "Sample headline",
    canonical_url: str = "https://example.com/article",
    member_urls: tuple[str, ...] = (),
    excerpt: str = "",
) -> RelevanceCluster:
    return RelevanceCluster(
        cluster_id=cluster_id,
        title=title,
        canonical_url=canonical_url,
        member_urls=member_urls,
        excerpt=excerpt,
    )


@pytest.fixture
def sample_categories() -> list[CategoryPlan]:
    return [
        _category(
            "startup_funding",
            description="Funding rounds and acquisitions.",
            include=["announced rounds", "new fund closes"],
            exclude=["recirculated old news"],
        ),
        _category(
            "ai_research",
            description="Foundation model and agentic research.",
            include=["scaling results", "alignment with practical bearing"],
        ),
    ]


# ── Prompt version ────────────────────────────────────────────────────────────


def test_prompt_version_is_relevance_v1() -> None:
    """First iteration of the relevance prompt; bump on behavior changes."""
    assert PROMPT_VERSION == "relevance.v1"


# ── build_relevance_messages — shape ──────────────────────────────────────────


def test_build_messages_returns_system_then_user(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages([_cluster(1)], sample_categories)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_build_messages_rejects_empty_batch(
    sample_categories: list[CategoryPlan],
) -> None:
    """An empty batch has no positional slots — fail loud rather than call the LLM."""
    with pytest.raises(ValueError, match="must not be empty"):
        build_relevance_messages([], sample_categories)


# ── build_relevance_messages — system message contents ────────────────────────


def test_system_message_lists_every_category_name(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages([_cluster(1)], sample_categories)
    system = msgs[0]["content"]
    assert "startup_funding" in system
    assert "ai_research" in system


def test_system_message_includes_category_descriptions_and_rules(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages([_cluster(1)], sample_categories)
    system = msgs[0]["content"]
    assert "Funding rounds and acquisitions." in system
    assert "announced rounds" in system
    assert "recirculated old news" in system


def test_system_message_states_keep_many_policy(
    sample_categories: list[CategoryPlan],
) -> None:
    """Load-bearing: architecture doc forbids a top-N selector posture."""
    system = build_relevance_messages([_cluster(1)], sample_categories)[0]["content"]
    assert "NOT a top-N selector" in system


def test_system_message_documents_response_format(
    sample_categories: list[CategoryPlan],
) -> None:
    """The prompt must spell out the JSON contract explicitly so local models
    don't drift into markdown-fenced replies under load.
    """
    system = build_relevance_messages([_cluster(1)], sample_categories)[0]["content"]
    assert '"verdicts"' in system
    assert "Begin your response with `{`" in system
    assert "no markdown fencing" in system


def test_system_message_handles_empty_categories() -> None:
    """Defensive — the resolver should always supply at least one category,
    but an empty list shouldn't crash prompt construction.
    """
    msgs = build_relevance_messages([_cluster(1)], [])
    assert "(none configured)" in msgs[0]["content"]


# ── build_relevance_messages — user message contents ──────────────────────────


def test_user_message_announces_batch_size(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages([_cluster(1), _cluster(2), _cluster(3)], sample_categories)
    user = msgs[1]["content"]
    assert "Batch of 3 clusters" in user
    assert "same order" in user


def test_user_message_lists_clusters_in_order(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages(
        [
            _cluster(1, title="First headline", canonical_url="https://a.example/"),
            _cluster(2, title="Second headline", canonical_url="https://b.example/"),
            _cluster(3, title="Third headline", canonical_url="https://c.example/"),
        ],
        sample_categories,
    )
    user = msgs[1]["content"]
    pos_1 = user.index("First headline")
    pos_2 = user.index("Second headline")
    pos_3 = user.index("Third headline")
    assert pos_1 < pos_2 < pos_3
    assert "[1]" in user and "[2]" in user and "[3]" in user


def test_user_message_includes_canonical_url(
    sample_categories: list[CategoryPlan],
) -> None:
    url = "https://techcrunch.com/anthropic-series-b"
    msgs = build_relevance_messages([_cluster(1, canonical_url=url)], sample_categories)
    assert url in msgs[1]["content"]


def test_user_message_includes_excerpt_when_present(
    sample_categories: list[CategoryPlan],
) -> None:
    excerpt = "Anthropic announced a new financing round."
    msgs = build_relevance_messages([_cluster(1, excerpt=excerpt)], sample_categories)
    assert excerpt in msgs[1]["content"]


def test_user_message_omits_excerpt_section_when_empty(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages([_cluster(1, excerpt="")], sample_categories)
    assert "excerpt:" not in msgs[1]["content"]


def test_user_message_falls_back_when_title_blank(
    sample_categories: list[CategoryPlan],
) -> None:
    """Empty title shouldn't produce a confusing blank cluster line."""
    msgs = build_relevance_messages([_cluster(1, title="")], sample_categories)
    assert "(untitled)" in msgs[1]["content"]


def test_user_message_lists_member_urls_for_folded_clusters(
    sample_categories: list[CategoryPlan],
) -> None:
    """L2/L3 folds expose multiple member URLs as cross-source evidence."""
    msgs = build_relevance_messages(
        [
            _cluster(
                1,
                canonical_url="https://npr.org/story",
                member_urls=(
                    "https://npr.org/story",
                    "https://kqed.org/syndicated",
                    "https://wbur.org/syndicated",
                ),
            )
        ],
        sample_categories,
    )
    user = msgs[1]["content"]
    assert "additional source urls" in user
    assert "kqed.org/syndicated" in user
    assert "wbur.org/syndicated" in user


def test_user_message_deduplicates_member_urls_against_canonical(
    sample_categories: list[CategoryPlan],
) -> None:
    """The canonical URL is already shown — don't repeat it in the extra list."""
    msgs = build_relevance_messages(
        [
            _cluster(
                1,
                canonical_url="https://npr.org/story",
                member_urls=(
                    "https://npr.org/story",
                    "https://kqed.org/syndicated",
                ),
            )
        ],
        sample_categories,
    )
    user = msgs[1]["content"]
    # Canonical URL appears once (in `url:` line), not duplicated in the extras section.
    assert user.count("https://npr.org/story") == 1


def test_user_message_omits_additional_urls_when_single_member(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages(
        [
            _cluster(
                1,
                canonical_url="https://example.com/a",
                member_urls=("https://example.com/a",),
            )
        ],
        sample_categories,
    )
    assert "additional source urls" not in msgs[1]["content"]


# ── parse_relevance_verdicts ──────────────────────────────────────────────────


def _verdict(keep: bool = True, category: str = "ai_research") -> RelevanceVerdict:
    return RelevanceVerdict(
        keep=keep,
        category=category,
        score=0.7,
        reason="reason",
    )


def test_parse_returns_validated_verdicts_in_order() -> None:
    response = RelevanceBatchResponse(
        verdicts=[
            _verdict(keep=True, category="ai_research"),
            _verdict(keep=False, category="startup_funding"),
        ]
    )
    verdicts = parse_relevance_verdicts(response, expected_count=2)
    assert [v.keep for v in verdicts] == [True, False]
    assert [v.category for v in verdicts] == ["ai_research", "startup_funding"]


def test_parse_raises_on_count_mismatch_too_few() -> None:
    response = RelevanceBatchResponse(verdicts=[_verdict()])
    with pytest.raises(ValueError, match="expected 3, got 1"):
        parse_relevance_verdicts(response, expected_count=3)


def test_parse_raises_on_count_mismatch_too_many() -> None:
    response = RelevanceBatchResponse(verdicts=[_verdict(), _verdict()])
    with pytest.raises(ValueError, match="expected 1, got 2"):
        parse_relevance_verdicts(response, expected_count=1)


def test_parse_raises_on_empty_verdicts_when_batch_was_nonempty() -> None:
    """An empty batch response against a real batch makes positional
    verdict assignment meaningless — load-bearing guard.
    """
    response = RelevanceBatchResponse(verdicts=[])
    with pytest.raises(ValueError, match="expected 2, got 0"):
        parse_relevance_verdicts(response, expected_count=2)
