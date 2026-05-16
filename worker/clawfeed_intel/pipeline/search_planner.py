"""Topic-query source planner for the Phase 7 topical-search flow.

Takes a user query plus the set of available source kinds and returns
a :class:`SearchPlan` — which source kinds to dispatch (in priority
order), what query variants to try, and any required/excluded terms
or expected evidence types. The result feeds the topic orchestrator
(Phase 7e), which builds a :class:`SourcePlan` from the plan and
dispatches the existing fetcher harness against it.

The architecture-doc "Prompt Responsibilities → Source Planner" spec
covers both the daily category planner and this topic-query planner.
Phase 7b uses a separate routing-stage entry (``search_planning``)
from the daily-side ``source_planning`` so the two prompt
responsibilities stay independently tunable.

Two-layer split, mirroring :mod:`pipeline.relevance` and
:mod:`pipeline.summary`:

- **Pure layer.** :func:`build_search_planner_messages`,
  :func:`parse_search_plan`, :data:`PROMPT_VERSION`. Fixture-testable
  without HTTP / DB / LLM.
- **Async orchestration** (lands in Phase 7e). Will call
  :meth:`LLMClient.chat_completion` with the :class:`SearchPlan`
  schema and hand the result to the topic-source-plan builder.

The available source kinds the prompt enumerates are passed in by the
caller, not read from :data:`sources.KNOWN_TASK_KINDS`. Phase 7b
doesn't introduce the new topic kinds (``hn_algolia`` / ``raw_cache``)
— those land in 7c — so the caller in 7e will pass the eventually-
correct list. Keeping the prompt parameterized keeps this module
independent of the fetcher registry's evolution.
"""

from __future__ import annotations

import logging

from ..llm.schemas import SearchPlan

log = logging.getLogger(__name__)

PROMPT_VERSION = "search_planner.v1"

# The planner produces structured output, so temperature is low. Not
# zero — the variant-generation task benefits from some breadth (we want
# the model to surface aliases and related phrasings, not just echo the
# query). Matches the architecture-doc Source Planner posture.
_SEARCH_PLANNER_TEMPERATURE = 0.2

# A `SearchPlan` is a small dict (~5 short string arrays). 1024 tokens is
# comfortable headroom; a single overlong "rationale" field would still
# fit. Sized at the call site (Phase 7e) — same rationale as the
# relevance + summary stages' max_tokens pins.
_SEARCH_PLANNER_MAX_TOKENS = 1024


# ── Prompt construction ───────────────────────────────────────────────────────


_SYSTEM_HEADER = (
    "You are a source planner for a personal intelligence brief system.\n"
    "\n"
    "For each user query you receive, decide which source kinds the worker "
    "should query and what query variants will produce the most relevant "
    "evidence. The user is asking a focused topical question; your plan "
    "drives the on-demand fetch + filter + summarize + compose pipeline.\n"
)

_PLANNING_POLICY = (
    "Planning policy:\n"
    "- Select source kinds from the supplied list ONLY. Do not invent kinds.\n"
    "- Order `selected_source_kinds` by priority — the first kind is the "
    "most important to query. The worker dispatches in this order and will "
    "tolerate per-source failures.\n"
    "- Generate 3-8 `query_variants` covering name aliases (e.g. firm name + "
    "founder name), related phrasings (e.g. 'X led round', 'X portfolio'), "
    "and source-tuned variants (e.g. Form D for SEC). The first variant "
    "SHOULD be the user's literal query.\n"
    "- `required_terms` (optional): post-fetch filter terms every kept "
    "item should contain (e.g. for 'Khosla Ventures' news, require "
    '"khosla"). Use sparingly — overly strict required terms collapse '
    "coverage.\n"
    "- `excluded_terms` (optional): terms whose presence disqualifies an "
    "item (e.g. exclude 'cricket' for 'Anthropic' queries to avoid the "
    "unrelated namesake).\n"
    "- `expected_evidence_types` (optional): short tags naming the kinds "
    "of items you expect to find (e.g. funding_round, regulatory_filing, "
    "github_repo, news_article). Drives downstream relevance judgment.\n"
    "- `rationale` (optional): one short sentence explaining your source "
    "+ variant choices. Audit-facing — keep it terse.\n"
)

_RESPONSE_FORMAT = (
    "Response format:\n"
    "Reply with a single JSON object. Shape:\n"
    "{\n"
    '  "selected_source_kinds": ["<kind>", "..."],\n'
    '  "query_variants": ["<variant>", "..."],\n'
    '  "required_terms": ["<term>", "..."],\n'
    '  "excluded_terms": ["<term>", "..."],\n'
    '  "expected_evidence_types": ["<type>", "..."],\n'
    '  "rationale": "<one short sentence>"\n'
    "}\n"
    "\n"
    "Reply with valid JSON only — no markdown fencing, no commentary, no "
    "preamble. Begin your response with `{`."
)


def build_search_planner_messages(
    query: str,
    *,
    available_source_kinds: list[str],
    window_days: int,
) -> list[dict[str, str]]:
    """Construct the OpenAI-style messages list for one planner call.

    Deterministic and fixture-testable — no HTTP, no LLM, no DB. The
    output is the exact ``messages`` argument the orchestrator hands
    to :meth:`LLMClient.chat_completion`.

    ``available_source_kinds`` is enumerated in the system prompt so
    the model knows the constrained set it must choose from. The
    caller is responsible for the actual list — Phase 7b doesn't
    introduce the new topic kinds, so 7e (which wires the planner
    into the orchestrator) will pass the eventually-correct set.

    Raises:
        ValueError: blank ``query`` or empty ``available_source_kinds``.
            Both are caller errors — without a query there's nothing
            to plan, and without source kinds there's no plan to make.
    """
    stripped = query.strip()
    if not stripped:
        raise ValueError("query must not be blank")
    if not available_source_kinds:
        raise ValueError("available_source_kinds must not be empty")

    system = _render_system_message(available_source_kinds)
    user = _render_user_message(stripped, window_days=window_days)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _render_system_message(available_source_kinds: list[str]) -> str:
    parts: list[str] = [
        _SYSTEM_HEADER,
        "",
        "Available source kinds (choose from these only):",
    ]
    for kind in available_source_kinds:
        parts.append(f"- {kind}")
    parts.extend(["", _PLANNING_POLICY, "", _RESPONSE_FORMAT])
    return "\n".join(parts)


def _render_user_message(query: str, *, window_days: int) -> str:
    return f"Query: {query}\nTime window: last {window_days} days.\n\nPlan the search."


# ── Response parsing ──────────────────────────────────────────────────────────


def parse_search_plan(parsed: SearchPlan) -> SearchPlan:
    """Type-guard pass-through for the planner response.

    The pydantic schema already validated the response shape at
    :meth:`LLMClient.chat_completion` time. This helper exists for
    symmetry with :func:`pipeline.relevance.parse_relevance_verdicts`
    and as the natural home for any future cross-field invariants the
    schema can't express (e.g. "every required_term must appear in
    at least one query_variant" — not enforced today, but the place
    it would live).

    Raises:
        TypeError: ``parsed`` is not a :class:`SearchPlan` instance.
            Defensive against a future caller passing the raw response
            dict; the type-check makes the contract explicit.
    """
    if not isinstance(parsed, SearchPlan):
        raise TypeError(f"expected SearchPlan, got {type(parsed).__name__}")
    return parsed


__all__ = (
    "PROMPT_VERSION",
    "build_search_planner_messages",
    "parse_search_plan",
)
