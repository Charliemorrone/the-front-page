"""Tests for the Phase 7b topic-query source planner pure layer.

Pure-layer focus: prompt construction, response parsing. No HTTP,
no LLM, no DB. Async orchestration wiring lands in Phase 7e and gets
its own tests at that time.
"""

from __future__ import annotations

import pytest

from clawfeed_intel.llm import SearchPlan
from clawfeed_intel.pipeline.search_planner import (
    PROMPT_VERSION,
    build_search_planner_messages,
    parse_search_plan,
)


# ── Prompt version pin ────────────────────────────────────────────────────────


def test_prompt_version_is_pinned() -> None:
    """Bump this string deliberately when the prompt's *meaning* changes
    (not on whitespace edits) so audit-log entries can be grouped.
    """
    assert PROMPT_VERSION == "search_planner.v1"


# ── Message shape ────────────────────────────────────────────────────────────


def _kinds() -> list[str]:
    return ["raw_cache", "gdelt", "hn_algolia", "reddit", "github_search", "sec_edgar", "rss"]


def test_build_messages_returns_system_then_user_in_order() -> None:
    messages = build_search_planner_messages(
        "Khosla Ventures",
        available_source_kinds=_kinds(),
        window_days=30,
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    assert len(messages) == 2


def test_build_messages_rejects_blank_query() -> None:
    """Boundary check — blank query is a caller error; we can't plan
    a search without one."""
    with pytest.raises(ValueError, match="query must not be blank"):
        build_search_planner_messages(
            "   ",
            available_source_kinds=_kinds(),
            window_days=30,
        )
    with pytest.raises(ValueError, match="query must not be blank"):
        build_search_planner_messages(
            "",
            available_source_kinds=_kinds(),
            window_days=30,
        )


def test_build_messages_rejects_empty_source_kinds() -> None:
    """Without source kinds the planner has no constrained set to
    choose from. Defensive — production callers always pass a
    non-empty list; the boundary check surfaces the misuse loudly."""
    with pytest.raises(ValueError, match="available_source_kinds must not be empty"):
        build_search_planner_messages(
            "Anthropic",
            available_source_kinds=[],
            window_days=30,
        )


# ── System message contents ──────────────────────────────────────────────────


def test_system_message_enumerates_every_available_kind() -> None:
    """The constrained-set list is load-bearing — without it the model
    would invent kinds the dispatcher can't handle."""
    messages = build_search_planner_messages(
        "Anthropic",
        available_source_kinds=_kinds(),
        window_days=14,
    )
    system = messages[0]["content"]
    for kind in _kinds():
        assert f"- {kind}" in system


def test_system_message_states_priority_ordering_policy() -> None:
    """`selected_source_kinds` order encodes priority — the prompt
    must say so explicitly. Load-bearing for the 7e dispatcher."""
    messages = build_search_planner_messages(
        "Anthropic",
        available_source_kinds=_kinds(),
        window_days=30,
    )
    system = messages[0]["content"]
    assert "priority" in system.lower()
    assert "first" in system.lower()


def test_system_message_states_variant_count_target() -> None:
    """Variant generation needs a soft target so the model doesn't
    emit one variant (defeats the purpose) or thirty (token waste)."""
    messages = build_search_planner_messages(
        "Anthropic",
        available_source_kinds=_kinds(),
        window_days=30,
    )
    system = messages[0]["content"]
    assert "3-8" in system or "3 to 8" in system


def test_system_message_states_first_variant_should_be_literal_query() -> None:
    """A predictable convention so the dispatcher can fall back to
    the first variant if the planner emits only one."""
    messages = build_search_planner_messages(
        "Anthropic",
        available_source_kinds=_kinds(),
        window_days=30,
    )
    system = messages[0]["content"]
    assert "first variant" in system.lower()
    assert "user's literal query" in system.lower() or "user query" in system.lower()


def test_system_message_states_json_only_response_format() -> None:
    """JSON-mode hardening (the 9a / 10a lesson) — local models drift
    into markdown fences and preambles under load without this."""
    messages = build_search_planner_messages(
        "Anthropic",
        available_source_kinds=_kinds(),
        window_days=30,
    )
    system = messages[0]["content"]
    assert "Reply with valid JSON only" in system
    assert "no markdown fencing" in system
    assert "Begin your response with `{`" in system


def test_system_message_lists_every_response_field() -> None:
    """Every SearchPlan field name must appear in the JSON-shape
    block so the model knows the expected keys without inference."""
    messages = build_search_planner_messages(
        "Anthropic",
        available_source_kinds=_kinds(),
        window_days=30,
    )
    system = messages[0]["content"]
    for key in (
        "selected_source_kinds",
        "query_variants",
        "required_terms",
        "excluded_terms",
        "expected_evidence_types",
        "rationale",
    ):
        assert key in system


# ── User message contents ────────────────────────────────────────────────────


def test_user_message_carries_query_and_window() -> None:
    messages = build_search_planner_messages(
        "Khosla Ventures",
        available_source_kinds=_kinds(),
        window_days=14,
    )
    user = messages[1]["content"]
    assert "Khosla Ventures" in user
    assert "14" in user
    assert "days" in user.lower()


def test_user_message_strips_query_whitespace() -> None:
    """Trim at the boundary so the prompt is clean even if the CLI
    handed us padded input."""
    messages = build_search_planner_messages(
        "  Khosla Ventures  ",
        available_source_kinds=_kinds(),
        window_days=30,
    )
    user = messages[1]["content"]
    assert "Khosla Ventures" in user
    assert "  Khosla Ventures  " not in user


def test_user_message_query_appears_before_window() -> None:
    """Query first, window context second — matches the planner's
    decision order (decide based on the topic, narrow by the window)."""
    messages = build_search_planner_messages(
        "Anthropic",
        available_source_kinds=_kinds(),
        window_days=7,
    )
    user = messages[1]["content"]
    assert user.index("Anthropic") < user.index("Time window")


# ── parse_search_plan ────────────────────────────────────────────────────────


def test_parse_returns_search_plan_unchanged() -> None:
    plan = SearchPlan(
        selected_source_kinds=["sec_edgar", "gdelt"],
        query_variants=["Khosla Ventures", "Vinod Khosla"],
    )
    out = parse_search_plan(plan)
    assert out is plan


def test_parse_rejects_wrong_type() -> None:
    """Defensive type-check for a future caller passing the raw
    response dict instead of the validated pydantic instance."""
    with pytest.raises(TypeError, match="expected SearchPlan"):
        parse_search_plan({"selected_source_kinds": ["gdelt"]})  # type: ignore[arg-type]


def test_parse_accepts_empty_plan() -> None:
    """Mirrors the schema's permissive posture — an empty plan parses
    successfully; the orchestrator surfaces the "nothing to dispatch"
    condition via coverage rather than this layer raising."""
    empty = SearchPlan()
    out = parse_search_plan(empty)
    assert out is empty
    assert out.selected_source_kinds == []
    assert out.query_variants == []
