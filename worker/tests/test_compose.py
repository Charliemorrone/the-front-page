"""Tests for the final-compose pure layer and DB loader.

Step 11a covers the deterministic layer: prompt construction,
response normalization, deterministic fallback briefs, and the
``iter_summarized_clusters_with_summary`` DB loader. The async
``compose_brief`` orchestration lands in 11b.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import Any

import httpx
import pytest

from clawfeed_intel import db as worker_db
from clawfeed_intel.llm import LLMClient, RetryConfig, RoutingConfig
from clawfeed_intel.llm.schemas import ClusterSummaryPayload
from clawfeed_intel.pipeline.compose import (
    PROMPT_VERSION,
    ComposeItem,
    ComposeOutputError,
    build_compose_messages,
    compose_brief,
    normalize_compose_output,
    render_empty_brief,
    render_fallback_brief,
)
from clawfeed_intel.runs import Coverage
from clawfeed_intel.sources import CategoryPlan, ProfileConfig, SourcePlan


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _item(
    *,
    cluster_id: int = 1,
    category: str | None = "ai_research",
    relevance_score: float | None = 0.8,
    headline: str = "Anthropic publishes scaling paper",
    summary: str = "Anthropic released a scaling-laws paper this week.",
    why_it_matters: str = "Aligns with the agentic-tooling roadmap.",
    entities: tuple[str, ...] = ("Anthropic",),
    key_facts: tuple[str, ...] = ("New 27B model size",),
    caveats: tuple[str, ...] = (),
    confidence: float | None = 0.75,
    source_urls: tuple[str, ...] = ("https://arxiv.org/abs/2405.12345",),
) -> ComposeItem:
    return ComposeItem(
        cluster_id=cluster_id,
        category=category,
        relevance_score=relevance_score,
        headline=headline,
        summary=summary,
        why_it_matters=why_it_matters,
        entities=entities,
        key_facts=key_facts,
        caveats=caveats,
        confidence=confidence,
        source_urls=source_urls,
    )


def _category(
    name: str,
    *,
    description: str = "",
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> CategoryPlan:
    return CategoryPlan(
        name=name,
        description=description,
        include_rules=list(include),
        exclude_rules=list(exclude),
    )


def _coverage(**overrides: object) -> Coverage:
    cov = Coverage()
    for key, value in overrides.items():
        setattr(cov, key, value)
    return cov


# ── PROMPT_VERSION ────────────────────────────────────────────────────────────


def test_prompt_version_is_pinned() -> None:
    """Bumped on behavioral changes — every compose call's audit row
    carries this so a prompt regression can be pinpointed.
    """
    assert PROMPT_VERSION == "compose.v1"


# ── build_compose_messages — shape ────────────────────────────────────────────


def test_build_messages_returns_system_then_user() -> None:
    messages = build_compose_messages(
        [_item()],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="2026-05-14T00:00:00+00:00",
        window_end="2026-05-15T00:00:00+00:00",
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_messages_rejects_empty_items() -> None:
    """Empty input would waste an LLM call — the orchestrator routes
    the zero-items path to :func:`render_empty_brief` directly.
    """
    with pytest.raises(ValueError, match="must not be empty"):
        build_compose_messages(
            [],
            plan_categories=[_category("ai_research")],
            coverage=_coverage(),
            window_start="x",
            window_end="y",
        )


# ── System message ────────────────────────────────────────────────────────────


def test_system_message_describes_brief_structure() -> None:
    system = build_compose_messages(
        [_item()],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="x",
        window_end="2026-05-15T00:00:00+00:00",
    )[0]["content"]
    for section in (
        "Executive Read",
        "Highest-Signal Developments",
        "Watchlist",
        "Coverage",
        "Daily Intelligence Brief",
    ):
        assert section in system, f"system message missing structural marker {section!r}"


def test_system_message_states_grounding_rule() -> None:
    """The grounding rule is load-bearing — the architecture-doc rule
    "final composer must not invent facts or add uncited claims" is
    enforced via this exact wording.
    """
    system = build_compose_messages(
        [_item()],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="x",
        window_end="y",
    )[0]["content"]
    assert "Grounding policy" in system
    assert "MUST come from the supplied cluster summaries" in system
    assert "Do not invent" in system


def test_system_message_states_citation_rule() -> None:
    system = build_compose_messages(
        [_item()],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="x",
        window_end="y",
    )[0]["content"]
    assert "[domain](url)" in system
    assert "source_urls" in system


def test_system_message_demands_pure_markdown() -> None:
    """No JSON / no code fences / begin with `# ` — mirrors the
    relevance + summary prompts' hardening for predictable parsing.
    """
    system = build_compose_messages(
        [_item()],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="x",
        window_end="y",
    )[0]["content"]
    assert "no markdown code fences" in system
    assert "Begin your response with `# `" in system


def test_system_message_lists_configured_categories() -> None:
    system = build_compose_messages(
        [_item()],
        plan_categories=[
            _category("startup_funding", description="Funding rounds and acquisitions."),
            _category("ai_research", description="Foundation model research."),
        ],
        coverage=_coverage(),
        window_start="x",
        window_end="y",
    )[0]["content"]
    assert "startup_funding" in system
    assert "Funding rounds and acquisitions." in system
    assert "ai_research" in system
    assert "Foundation model research." in system


# ── User message ──────────────────────────────────────────────────────────────


def test_user_message_states_brief_window_and_h1_date() -> None:
    user = build_compose_messages(
        [_item()],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="2026-05-14T06:00:00+00:00",
        window_end="2026-05-15T06:00:00+00:00",
    )[1]["content"]
    assert "Brief window: 2026-05-14T06:00:00+00:00 → 2026-05-15T06:00:00+00:00" in user
    assert "H1 date for the brief title: 2026-05-15" in user


def test_user_message_includes_coverage_block() -> None:
    user = build_compose_messages(
        [_item()],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(
            sources_attempted=32,
            sources_succeeded=29,
            raw_items=418,
            clusters=151,
            kept_clusters=43,
            summarized_clusters=43,
        ),
        window_start="x",
        window_end="y",
    )[1]["content"]
    assert "Sources attempted: 32" in user
    assert "Sources succeeded: 29" in user
    assert "Raw items: 418" in user
    assert "Event clusters: 151" in user
    assert "Kept clusters: 43" in user
    assert "Summarized clusters: 43" in user


def test_user_message_surfaces_failure_signals() -> None:
    """A degraded run must surface failed sources / plan warnings /
    failed batches in the prompt so the composer's Coverage section
    can explain the gaps. Architecture-doc rule: the brief "clearly
    names what degraded."
    """
    coverage = _coverage(
        failed_filter_batches=2,
        failed_summary_clusters=5,
    )
    coverage.failed_sources = ["source-a", "source-b"]
    coverage.skipped_sources = ["source-c: no fetcher"]
    coverage.plan_warnings = ["unknown kind 'lemmings'"]

    user = build_compose_messages(
        [_item()],
        plan_categories=[_category("ai_research")],
        coverage=coverage,
        window_start="x",
        window_end="y",
    )[1]["content"]
    assert "Failed sources: source-a, source-b" in user
    assert "Skipped sources: source-c: no fetcher" in user
    assert "Plan warnings: unknown kind 'lemmings'" in user
    assert "Failed filter batches: 2" in user
    assert "Failed summary clusters: 5" in user


def test_user_message_groups_items_by_configured_category_order() -> None:
    """Plan ordering controls section ordering — deterministic across
    re-runs even if the DB loader's relevance-score order interleaves
    categories.
    """
    items = [
        _item(cluster_id=1, category="ai_research", headline="AI item 1"),
        _item(cluster_id=2, category="startup_funding", headline="Funding item 1"),
        _item(cluster_id=3, category="ai_research", headline="AI item 2"),
        _item(cluster_id=4, category="startup_funding", headline="Funding item 2"),
    ]
    user = build_compose_messages(
        items,
        plan_categories=[
            _category("startup_funding"),
            _category("ai_research"),
        ],
        coverage=_coverage(),
        window_start="x",
        window_end="y",
    )[1]["content"]
    # `startup_funding` block appears before `ai_research` block.
    sf_pos = user.index("### startup_funding")
    ar_pos = user.index("### ai_research")
    assert sf_pos < ar_pos
    # Within each block items appear in input order.
    assert user.index("Funding item 1") < user.index("Funding item 2")
    assert user.index("AI item 1") < user.index("AI item 2")


def test_user_message_routes_unknown_category_to_uncategorized() -> None:
    """A cluster's category slug not in the plan → ``Uncategorized``
    bucket. The composer's prompt instructs it to Watchlist these.
    """
    items = [_item(cluster_id=1, category="mystery_genre", headline="stray item")]
    user = build_compose_messages(
        items,
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="x",
        window_end="y",
    )[1]["content"]
    assert "### Uncategorized" in user
    assert "stray item" in user


def test_user_message_renders_relevance_and_confidence_tags() -> None:
    """Both signals reach the prompt as numerical context so the
    composer can prioritize and route to Watchlist appropriately.
    """
    user = build_compose_messages(
        [_item(relevance_score=0.91, confidence=0.62)],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="x",
        window_end="y",
    )[1]["content"]
    assert "relevance=0.91" in user
    assert "confidence=0.62" in user


def test_user_message_renders_missing_score_or_confidence_as_question() -> None:
    """Null relevance / confidence shouldn't crash rendering; surface
    as ``?`` so the composer knows the value is absent.
    """
    user = build_compose_messages(
        [_item(relevance_score=None, confidence=None)],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="x",
        window_end="y",
    )[1]["content"]
    assert "relevance=?" in user
    assert "confidence=?" in user


def test_user_message_surfaces_all_payload_fields() -> None:
    user = build_compose_messages(
        [
            _item(
                headline="Headline X",
                summary="Summary X.",
                why_it_matters="Mat",
                entities=("Ent1", "Ent2"),
                key_facts=("Fact 1", "Fact 2"),
                caveats=("Caveat 1",),
                source_urls=("https://a/", "https://b/"),
            )
        ],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="x",
        window_end="y",
    )[1]["content"]
    assert "headline: Headline X" in user
    assert "summary: Summary X." in user
    assert "why_it_matters: Mat" in user
    assert "entities: Ent1, Ent2" in user
    assert "- Fact 1" in user
    assert "- Caveat 1" in user
    assert "- https://a/" in user
    assert "- https://b/" in user


def test_user_message_omits_empty_optional_sections() -> None:
    """An item with no caveats / no entities shouldn't render empty
    label lines that the composer would have to interpret.
    """
    user = build_compose_messages(
        [
            _item(
                why_it_matters="",
                entities=(),
                key_facts=(),
                caveats=(),
                source_urls=(),
            )
        ],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="x",
        window_end="y",
    )[1]["content"]
    assert "why_it_matters:" not in user
    assert "entities:" not in user
    assert "key_facts:" not in user
    assert "caveats:" not in user
    assert "source_urls:" not in user


# ── normalize_compose_output ──────────────────────────────────────────────────


def test_normalize_pass_through_clean_markdown() -> None:
    brief = "# Daily Intelligence Brief — 2026-05-15\n\n## Executive Read\n\nBody."
    assert normalize_compose_output(brief) == brief


def test_normalize_strips_surrounding_whitespace() -> None:
    brief = "  \n# Daily Intelligence Brief — 2026-05-15\n\nbody\n  "
    out = normalize_compose_output(brief)
    assert out.startswith("# Daily")
    assert out.endswith("body")


def test_normalize_strips_markdown_code_fence_wrapper() -> None:
    """Local models sometimes wrap the response in ``` ```markdown ... ``` ```
    — defense against that without re-prompting.
    """
    fenced = "```markdown\n# Daily Intelligence Brief\n\nbody\n```"
    out = normalize_compose_output(fenced)
    assert out == "# Daily Intelligence Brief\n\nbody"


def test_normalize_strips_bare_fence_wrapper() -> None:
    fenced = "```\n# Daily Intelligence Brief\n\nbody\n```"
    out = normalize_compose_output(fenced)
    assert out == "# Daily Intelligence Brief\n\nbody"


def test_normalize_drops_preamble_before_first_heading() -> None:
    """Models sometimes prepend ``Sure, here's the brief:\\n\\n# ...``."""
    polluted = "Sure, here is the brief:\n\n# Daily Intelligence Brief\n\nbody"
    out = normalize_compose_output(polluted)
    assert out.startswith("# Daily Intelligence Brief")
    assert "Sure" not in out


def test_normalize_raises_on_no_heading() -> None:
    with pytest.raises(ComposeOutputError, match="no '# ' heading"):
        normalize_compose_output("This is just prose, no heading at all.")


def test_normalize_raises_on_empty_response() -> None:
    with pytest.raises(ComposeOutputError, match="empty"):
        normalize_compose_output("")


def test_normalize_raises_on_whitespace_only_response() -> None:
    with pytest.raises(ComposeOutputError, match="empty"):
        normalize_compose_output("   \n\t  \n")


def test_normalize_raises_on_bare_fence_line() -> None:
    with pytest.raises(ComposeOutputError, match="bare fence line"):
        normalize_compose_output("```")


# ── render_empty_brief ────────────────────────────────────────────────────────


def test_empty_brief_renders_coverage_only() -> None:
    """Zero items → coverage-only brief, no LLM call needed."""
    coverage = _coverage(
        sources_attempted=5,
        sources_succeeded=3,
        raw_items=42,
        clusters=10,
        kept_clusters=0,
        summarized_clusters=0,
    )
    coverage.failed_sources = ["broken-feed"]
    out = render_empty_brief(
        coverage=coverage,
        window_start="2026-05-14T00:00:00+00:00",
        window_end="2026-05-15T00:00:00+00:00",
        run_id=42,
    )
    assert out.startswith("# Daily Intelligence Brief — 2026-05-15")
    assert "## Executive Read" in out
    assert "## Coverage" in out
    assert "Sources attempted: 5" in out
    assert "Failed sources: broken-feed" in out
    assert "_Run id: 42._" in out
    # Must satisfy the same shape as the LLM-composed brief.
    assert normalize_compose_output(out) == out.strip()


# ── render_fallback_brief ─────────────────────────────────────────────────────


def test_fallback_brief_renders_items_grouped_by_category() -> None:
    items = [
        _item(cluster_id=1, category="ai_research", headline="AI thing"),
        _item(cluster_id=2, category="startup_funding", headline="Funding thing"),
    ]
    out = render_fallback_brief(
        items,
        plan_categories=[_category("startup_funding"), _category("ai_research")],
        coverage=_coverage(),
        window_start="x",
        window_end="2026-05-15T00:00:00+00:00",
        run_id=7,
        failure_reason="vMLX read timeout",
    )
    assert out.startswith("# Daily Intelligence Brief — 2026-05-15 (degraded)")
    assert "## Startup Funding" in out
    assert "## Ai Research" in out
    assert "vMLX read timeout" in out
    assert "_Run id: 7._" in out
    assert out.index("## Startup Funding") < out.index("## Ai Research")
    assert "### Funding thing" in out
    assert "### AI thing" in out


def test_fallback_brief_humanizes_slugs_consistently() -> None:
    """``startup_funding`` → ``Startup Funding``; reproducible across
    every category slug without a special-case map.
    """
    items = [
        _item(cluster_id=1, category="github_traction", headline="repo trending"),
        _item(cluster_id=2, category="ai-coding-tools", headline="tool released"),
    ]
    out = render_fallback_brief(
        items,
        plan_categories=[
            _category("github_traction"),
            _category("ai-coding-tools"),
        ],
        coverage=_coverage(),
        window_start="x",
        window_end="2026-05-15T00:00:00+00:00",
        run_id=1,
        failure_reason="x",
    )
    assert "## Github Traction" in out
    assert "## Ai Coding Tools" in out


def test_fallback_brief_preserves_citations() -> None:
    """The architecture-doc rule "preserve citations" must hold even
    in the degraded path — that's the user's link back to the source.
    """
    out = render_fallback_brief(
        [
            _item(
                source_urls=(
                    "https://techcrunch.com/anthropic",
                    "https://www.sec.gov/x",
                )
            )
        ],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(),
        window_start="x",
        window_end="2026-05-15T00:00:00+00:00",
        run_id=1,
        failure_reason="x",
    )
    assert "https://techcrunch.com/anthropic" in out
    assert "https://www.sec.gov/x" in out


def test_fallback_brief_renders_coverage_section() -> None:
    out = render_fallback_brief(
        [_item()],
        plan_categories=[_category("ai_research")],
        coverage=_coverage(
            sources_attempted=10,
            sources_succeeded=8,
            kept_clusters=5,
            summarized_clusters=5,
            failed_summary_clusters=2,
        ),
        window_start="x",
        window_end="2026-05-15T00:00:00+00:00",
        run_id=99,
        failure_reason="x",
    )
    assert "## Coverage" in out
    assert "Sources attempted: 10" in out
    assert "Failed summary clusters: 2" in out


# ── db.iter_summarized_clusters_with_summary ──────────────────────────────────


def _seed_summarized(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    key: str,
    category: str = "ai_research",
    relevance_score: float = 0.7,
    headline: str = "h",
    summary: str = "s",
    confidence: float | None = 0.5,
    source_urls: tuple[str, ...] = ("https://example.com/a",),
) -> int:
    rep_id, _ = worker_db.upsert_raw_item(
        conn,
        run_id=run_id,
        source_type="rss",
        dedup_key=f"{key}-rep",
        title=key,
        url=key,
        canonical_url=key,
        content="",
    )
    cluster_id, _ = worker_db.create_cluster(
        conn,
        run_id=run_id,
        cluster_key=key,
        title=key,
        raw_item_ids=[rep_id],
    )
    worker_db.update_cluster_verdict(
        conn,
        cluster_id=cluster_id,
        status="kept",
        relevance_score=relevance_score,
        category=category,
        event_type=None,
        filter_reason="r",
    )
    payload = ClusterSummaryPayload(
        headline=headline,
        summary=summary,
        confidence=confidence,
        source_urls=list(source_urls),
    )
    worker_db.create_item_summary(
        conn,
        cluster_id=cluster_id,
        model="stub-model",
        prompt_version="summary.v1",
        payload=payload,
    )
    worker_db.advance_cluster_to_summarized(conn, cluster_id)
    return cluster_id


def _make_run(conn: sqlite3.Connection) -> int:
    return worker_db.create_run(
        conn,
        run_type="daily",
        window_start="2026-05-14T00:00:00+00:00",
        window_end="2026-05-15T00:00:00+00:00",
    )


def test_iter_summarized_yields_only_summarized_clusters(temp_db) -> None:
    """A pending cluster, a filtered_out cluster, and a kept (not yet
    summarized) cluster all must be filtered out by the SQL predicate.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)

        # One properly summarized cluster.
        summarized_id = _seed_summarized(conn, run_id=run_id, key="https://a.example/")

        # One cluster left at 'kept' (summary stage hasn't run for it yet).
        rep_id, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="rss",
            dedup_key="kept-rep",
            title="kept-only",
            url="https://b.example/",
            canonical_url="https://b.example/",
            content="",
        )
        kept_id, _ = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://b.example/",
            title="kept-only",
            raw_item_ids=[rep_id],
        )
        worker_db.update_cluster_verdict(
            conn,
            cluster_id=kept_id,
            status="kept",
            relevance_score=0.5,
            category="ai_research",
            event_type=None,
            filter_reason="r",
        )

        rows = list(worker_db.iter_summarized_clusters_with_summary(conn, run_id))
        assert len(rows) == 1
        assert rows[0]["cluster_id"] == summarized_id


def test_iter_summarized_orders_by_relevance_score_desc(temp_db) -> None:
    """Higher-relevance items first so the composer sees the
    highest-signal clusters at the top of the prompt.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        low = _seed_summarized(conn, run_id=run_id, key="https://low.example/", relevance_score=0.3)
        high = _seed_summarized(
            conn, run_id=run_id, key="https://high.example/", relevance_score=0.9
        )
        mid = _seed_summarized(conn, run_id=run_id, key="https://mid.example/", relevance_score=0.6)

        rows = list(worker_db.iter_summarized_clusters_with_summary(conn, run_id))
        ordered_ids = [row["cluster_id"] for row in rows]
        assert ordered_ids == [high, mid, low]


def test_iter_summarized_sorts_null_relevance_last(temp_db) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        with_score = _seed_summarized(
            conn, run_id=run_id, key="https://scored.example/", relevance_score=0.5
        )
        # Manually create a summarized cluster with NULL relevance_score
        # (a defensive corner — production won't normally produce this).
        rep_id, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="rss",
            dedup_key="null-score",
            title="null",
            url="https://null.example/",
            canonical_url="https://null.example/",
            content="",
        )
        null_id, _ = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://null.example/",
            title="null",
            raw_item_ids=[rep_id],
        )
        conn.execute(
            "UPDATE item_clusters SET status='summarized', relevance_score=NULL WHERE id=?",
            (null_id,),
        )
        conn.commit()
        worker_db.create_item_summary(
            conn,
            cluster_id=null_id,
            model="stub-model",
            prompt_version="summary.v1",
            payload=ClusterSummaryPayload(headline="h", summary="s"),
        )

        rows = list(worker_db.iter_summarized_clusters_with_summary(conn, run_id))
        assert [row["cluster_id"] for row in rows] == [with_score, null_id]


def test_iter_summarized_returns_latest_summary_per_cluster(temp_db) -> None:
    """When multiple summaries exist per cluster (e.g. prompt-version
    bump), the largest-id summary row wins.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        cluster_id = _seed_summarized(conn, run_id=run_id, key="https://a.example/")
        # Append a second summary row (different model + version).
        worker_db.create_item_summary(
            conn,
            cluster_id=cluster_id,
            model="newer-model",
            prompt_version="summary.v2",
            payload=ClusterSummaryPayload(
                headline="REPLACED headline",
                summary="REPLACED summary.",
            ),
        )

        rows = list(worker_db.iter_summarized_clusters_with_summary(conn, run_id))
        assert len(rows) == 1
        assert rows[0]["headline"] == "REPLACED headline"


def test_iter_summarized_round_trips_json_list_fields(temp_db) -> None:
    """The TEXT-JSON list fields survive the round-trip; the pipeline
    layer is responsible for ``json.loads`` when hydrating ComposeItem.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_summarized(
            conn,
            run_id=run_id,
            key="https://a.example/",
            source_urls=("https://a/", "https://b/"),
        )
        row = next(worker_db.iter_summarized_clusters_with_summary(conn, run_id))
        assert json.loads(row["source_urls"]) == ["https://a/", "https://b/"]
        assert json.loads(row["entities"]) == []
        assert json.loads(row["key_facts"]) == []


def test_iter_summarized_empty_for_run_without_summaries(temp_db) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        rows = list(worker_db.iter_summarized_clusters_with_summary(conn, run_id))
        assert rows == []


# ── Orchestration fixtures (step 11b) ─────────────────────────────────────────


@pytest.fixture
def routing() -> RoutingConfig:
    """Minimal routing config sized for final-compose tests."""
    return RoutingConfig.model_validate(
        {
            "providers": {"vmlx": {"base_url": "http://127.0.0.1:8080/v1"}},
            "stages": {
                "final_compose": {
                    "provider": "vmlx",
                    "model": "stub-compose-model",
                    "timeout_seconds": 30,
                },
            },
        }
    )


def _chat_response(*, content: str, model: str = "stub-compose-model") -> dict[str, Any]:
    return {
        "id": "chatcmpl-compose",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_client(
    routing: RoutingConfig,
    handler: Any,
    *,
    conn: sqlite3.Connection | None = None,
    run_id: int | None = None,
) -> LLMClient:
    return LLMClient(
        routing,
        transport=httpx.MockTransport(handler),
        conn=conn,
        run_id=run_id,
        retry_config=RetryConfig(max_attempts=1, wait_min_seconds=0, wait_max_seconds=0),
    )


def _make_plan(categories: list[CategoryPlan] | None = None) -> SourcePlan:
    return SourcePlan(
        profile=ProfileConfig(),
        categories=categories or [],
        dynamic_search=[],
        warnings=[],
    )


# ── compose_brief — orchestration ─────────────────────────────────────────────


async def test_compose_brief_zero_items_returns_empty_stub(temp_db, routing) -> None:
    """No summarized clusters → empty-brief stub, no LLM call."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        result = await compose_brief(
            conn,
            run_id,
            client,
            plan=_make_plan(),
            coverage=Coverage(),
            window_start="2026-05-14T00:00:00+00:00",
            window_end="2026-05-15T00:00:00+00:00",
            model="stub-compose-model",
        )

        assert calls == []
        assert result.provider_tag == "local_stub_empty"
        assert result.markdown.startswith("# Daily Intelligence Brief — 2026-05-15")
        assert "## Coverage" in result.markdown


async def test_compose_brief_happy_path_returns_normalized_markdown(temp_db, routing) -> None:
    """Items present → one LLM call → normalized markdown returned."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_response(
                content=(
                    "# Daily Intelligence Brief — 2026-05-15\n\n"
                    "## Executive Read\n\nA real-looking synthesis.\n\n"
                    "## Startup Funding\n\nSome funding paragraph.\n"
                )
            ),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_summarized(conn, run_id=run_id, key="https://a.example/")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        result = await compose_brief(
            conn,
            run_id,
            client,
            plan=_make_plan(),
            coverage=Coverage(summarized_clusters=1),
            window_start="x",
            window_end="2026-05-15T00:00:00+00:00",
            model="stub-compose-model",
        )

        assert result.provider_tag == "vmlx_fallback"
        assert result.model == "stub-compose-model"
        assert result.markdown.startswith("# Daily Intelligence Brief — 2026-05-15")
        assert "Executive Read" in result.markdown
        assert "Startup Funding" in result.markdown


async def test_compose_brief_strips_fence_wrappers(temp_db, routing) -> None:
    """The orchestrator's normalization handles ```markdown … ``` wrappers."""

    def handler(_request: httpx.Request) -> httpx.Response:
        fenced = (
            "```markdown\n# Daily Intelligence Brief — 2026-05-15\n\n"
            "## Executive Read\n\nFine.\n```"
        )
        return httpx.Response(200, json=_chat_response(content=fenced))

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_summarized(conn, run_id=run_id, key="https://a.example/")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        result = await compose_brief(
            conn,
            run_id,
            client,
            plan=_make_plan(),
            coverage=Coverage(),
            window_start="x",
            window_end="2026-05-15T00:00:00+00:00",
            model="stub-compose-model",
        )
        assert result.provider_tag == "vmlx_fallback"
        assert result.markdown.startswith("# Daily Intelligence Brief — 2026-05-15")
        assert "```" not in result.markdown


async def test_compose_brief_uses_call_site_sampling(temp_db, routing) -> None:
    """temperature=0.3 and max_tokens=8192 are pinned at the call site."""
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_chat_response(content="# Daily Intelligence Brief — 2026-05-15\n\nbody\n"),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_summarized(conn, run_id=run_id, key="https://a.example/")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        await compose_brief(
            conn,
            run_id,
            client,
            plan=_make_plan(),
            coverage=Coverage(),
            window_start="x",
            window_end="2026-05-15T00:00:00+00:00",
            model="stub-compose-model",
        )

    assert captured[0]["temperature"] == 0.3
    assert captured[0]["max_tokens"] == 8192


async def test_compose_brief_falls_back_on_llm_failure(temp_db, routing) -> None:
    """LLM 5xx → render_fallback_brief substituted; run still publishes."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_summarized(
            conn,
            run_id=run_id,
            key="https://a.example/",
            category="ai_research",
            headline="Sticking-around headline",
        )
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        result = await compose_brief(
            conn,
            run_id,
            client,
            plan=_make_plan([_category("ai_research")]),
            coverage=Coverage(summarized_clusters=1),
            window_start="x",
            window_end="2026-05-15T00:00:00+00:00",
            model="stub-compose-model",
        )

        assert result.provider_tag == "local_stub_failed"
        assert result.markdown.startswith("# Daily Intelligence Brief — 2026-05-15 (degraded)")
        # The fallback path renders the cluster's actual content from
        # the structured summary — proves citations + content survive.
        assert "Sticking-around headline" in result.markdown


async def test_compose_brief_falls_back_on_unnormalizable_response(temp_db, routing) -> None:
    """LLM returns malformed content (no `#` heading) → fallback path
    fires, exactly as it does for HTTP failure.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_response(content="here is some prose with no heading anywhere"),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_summarized(conn, run_id=run_id, key="https://a.example/")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        result = await compose_brief(
            conn,
            run_id,
            client,
            plan=_make_plan([_category("ai_research")]),
            coverage=Coverage(),
            window_start="x",
            window_end="2026-05-15T00:00:00+00:00",
            model="stub-compose-model",
        )

        assert result.provider_tag == "local_stub_failed"
        assert "(degraded)" in result.markdown


async def test_compose_brief_writes_audit_row(temp_db, routing) -> None:
    """Compose call produces exactly one ``llm_calls`` row with the
    right stage / provider / prompt_version.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_response(content="# Daily Intelligence Brief — 2026-05-15\n\nbody\n"),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_summarized(conn, run_id=run_id, key="https://a.example/")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        await compose_brief(
            conn,
            run_id,
            client,
            plan=_make_plan(),
            coverage=Coverage(),
            window_start="x",
            window_end="2026-05-15T00:00:00+00:00",
            model="stub-compose-model",
        )

        rows = conn.execute(
            "SELECT stage, provider, status, prompt_version FROM llm_calls WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["stage"] == "final_compose"
        assert rows[0]["provider"] == "vmlx"
        assert rows[0]["status"] == "succeeded"
        assert rows[0]["prompt_version"] == "compose.v1"


async def test_compose_brief_orders_items_by_relevance(temp_db, routing) -> None:
    """The prompt's user message lists items in DB-loader order
    (relevance DESC, id ASC) so higher-signal items reach the model first.
    """
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_chat_response(content="# Daily Intelligence Brief — 2026-05-15\n\nbody\n"),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_summarized(
            conn,
            run_id=run_id,
            key="https://low.example/",
            relevance_score=0.2,
            headline="LOW-PRIORITY-MARKER",
        )
        _seed_summarized(
            conn,
            run_id=run_id,
            key="https://high.example/",
            relevance_score=0.95,
            headline="HIGH-PRIORITY-MARKER",
        )
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        await compose_brief(
            conn,
            run_id,
            client,
            plan=_make_plan([_category("ai_research")]),
            coverage=Coverage(),
            window_start="x",
            window_end="2026-05-15T00:00:00+00:00",
            model="stub-compose-model",
        )

    user = captured[0]["messages"][1]["content"]
    assert user.index("HIGH-PRIORITY-MARKER") < user.index("LOW-PRIORITY-MARKER")


async def test_compose_brief_surfaces_citation_urls_in_prompt(temp_db, routing) -> None:
    """Each cluster's source_urls reach the model — the grounding +
    citation rules in the prompt depend on these landing in context.
    """
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_chat_response(content="# Daily Intelligence Brief — 2026-05-15\n\nbody\n"),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _make_run(conn)
        _seed_summarized(
            conn,
            run_id=run_id,
            key="https://a.example/",
            source_urls=("https://distinctive.example/source-1",),
        )
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        await compose_brief(
            conn,
            run_id,
            client,
            plan=_make_plan(),
            coverage=Coverage(),
            window_start="x",
            window_end="2026-05-15T00:00:00+00:00",
            model="stub-compose-model",
        )

    user = captured[0]["messages"][1]["content"]
    assert "https://distinctive.example/source-1" in user
