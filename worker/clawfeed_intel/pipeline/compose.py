"""Final composition for the composing lifecycle stage.

Takes every cluster at ``status='summarized'`` and produces one
Markdown daily brief. Distinct from the per-cluster ``summarize``
stage in two ways:

- **One LLM call per run, not per cluster.** Final composition is
  where prose quality matters, and the architecture doc reserves one
  or two calls for it. The call sees every summary in one prompt.
- **Free-form Markdown output, no JSON schema.** Wrapping a long
  brief in a JSON string is the perfect setup for escape-quoting
  failures the bounded-repair retry can't recover from; the prompt
  enforces the shape ("begin with `# `, no markdown fencing") and a
  lightweight :func:`normalize_compose_output` strips off the
  predictable model artifacts (preamble, fence wrappers).

Phase 1 reality: the architecture-doc-target plan routes
``final_compose`` to OpenClaw / ``gpt-5.3-codex`` with a local
``Qwen3.5-122B-A10B-4bit`` fallback. Until OpenClaw's wire protocol
is integrated AND the flagship is on disk, the configured stage is
``mlx-community/Qwen3.5-27B-4bit`` via vmlx — the orchestrator stamps
``metadata.composition_provider = 'vmlx_fallback'`` to mark the
degraded mode honestly in the digest record.

Two-layer split, mirroring :mod:`pipeline.summary` and
:mod:`pipeline.relevance`:

- **Pure layer.** :class:`ComposeItem`, :data:`PROMPT_VERSION`,
  :func:`build_compose_messages`, :func:`normalize_compose_output`,
  :func:`render_empty_brief`, :func:`render_fallback_brief`.
  Fixture-testable without HTTP / DB / LLM.
- **Async orchestration.** :func:`compose_brief` (lands in 11b)
  loads summarized items, dispatches one LLM call against the
  ``final_compose`` stage, normalizes the response, and returns the
  Markdown string the orchestrator stores in ``digests.content``.
  Per-call failure routes to :func:`render_fallback_brief` so the
  run still publishes a useful brief — the architecture-doc rule
  "failed sources degrade coverage; they do not fail the run"
  extended to the compose stage.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass

from .. import db
from ..llm import LLMClient, StageConfig
from ..llm.routing import FallbackConfig
from ..runs import Coverage
from ..sources import CategoryPlan, SourcePlan

log = logging.getLogger(__name__)

PROMPT_VERSION = "compose.v1"

# Brief prose; higher than the 0.1 used for structured-output stages
# so the composer can write idiomatic English. The grounding rule is
# enforced by the prompt + the orchestrator's normalization, not by
# pinning temperature low — a 0.1 here produces wooden output without
# meaningfully improving accuracy because the model is conditioning on
# already-grounded summaries.
_COMPOSE_TEMPERATURE = 0.3

# Generous: a brief with 200+ kept clusters can run several thousand
# output tokens. vMLX servers cap server-side, so this is a hint, not
# a guarantee. If the response is truncated, the orchestrator's
# normalizer still produces a usable (if short) brief — that's the
# architecture-doc "degraded mode" posture.
_COMPOSE_MAX_TOKENS = 8192


# ── Errors ────────────────────────────────────────────────────────────────────


class ComposeOutputError(ValueError):
    """Final-compose response failed lightweight structure checks.

    Raised when the response is empty after fence/preamble stripping
    or has no ``#`` heading. The orchestrator catches this and routes
    to :func:`render_fallback_brief` rather than failing the run —
    consistent with the architecture-doc "always publish a useful
    brief" rule.
    """


# ── Input shape ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ComposeItem:
    """One cluster's brief-input shape, hydrated from ``item_summaries``.

    The brief composer consumes :class:`ComposeItem` instances; the
    DB loader (:func:`db.iter_summarized_clusters_with_summary`) is
    responsible for hydrating these from the JSON-string columns in
    ``item_summaries`` so the pure helpers never see raw SQLite rows.

    ``category`` is the slug written by the relevance filter (may be
    ``None`` for clusters whose filter verdict left it empty —
    permissive per the 9c lesson). ``relevance_score`` is the LLM's
    rank from the filter pass; higher means the model judged the
    cluster more brief-worthy. The composer uses this signal for the
    "Highest-Signal Developments" section.
    """

    cluster_id: int
    category: str | None
    relevance_score: float | None
    headline: str
    summary: str
    why_it_matters: str = ""
    entities: tuple[str, ...] = ()
    key_facts: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    confidence: float | None = None
    source_urls: tuple[str, ...] = ()


# ── Prompt construction ───────────────────────────────────────────────────────


_SYSTEM_HEADER = (
    "You are the final composer for a personal daily intelligence brief.\n"
    "\n"
    "You receive structured summaries of every kept cluster from today's "
    "run plus a coverage report. Your job is to weave them into one "
    "publishable Markdown brief.\n"
)

_BRIEF_STRUCTURE = (
    "Brief structure (use this exact section order — skip empty sections):\n"
    "1. `# Daily Intelligence Brief — <date>` (H1 with the run's window-end date)\n"
    "2. `## Executive Read` — 1-3 short paragraphs synthesizing the day "
    "across all categories. Frame what mattered and why.\n"
    "3. `## Highest-Signal Developments` — the 3-7 items that most "
    "deserve attention across all categories. Use the relevance score "
    "and confidence as hints. Each item is a short bullet with a "
    "headline and the most important fact, ending in citation links.\n"
    "4. One `## <Category>` section per editorial category that has "
    "items (e.g. `## Startup Funding`, `## AI Research`). Humanize "
    "category slugs (snake_case → Title Case). Inside each section, "
    "include EVERY cluster filed under that category — this brief is "
    "NOT a top-N teaser. Each cluster gets its own paragraph or "
    "bullet group ending in citation links.\n"
    "5. `## Watchlist` — lower-confidence or developing items "
    "(items with `confidence` ≤ 0.4, or items the cluster summary "
    "flagged with caveats). One short bullet per item.\n"
    "6. `## Coverage` — the coverage stats supplied below, rendered "
    "as a short bulleted list. Name any failed sources, plan "
    "warnings, failed filter batches, or failed summary clusters.\n"
)

_GROUNDING_POLICY = (
    "Grounding policy (load-bearing):\n"
    "- Every fact, name, number, and claim MUST come from the supplied "
    "cluster summaries below. Do not invent details that aren't in the "
    "input. Do not invent clusters.\n"
    "- Preserve citations: end each item with one or more "
    "`[domain](url)` markdown links drawn from the cluster's "
    "`source_urls` list. Do not invent URLs.\n"
    "- If sources disagree inside a cluster's caveats, reflect that "
    "in the prose rather than picking a side.\n"
    "- Do not add editorializing or speculation. The brief is a "
    "factual condensation.\n"
)

_RESPONSE_FORMAT = (
    "Output format:\n"
    "Reply with pure Markdown — no JSON, no markdown code fences, "
    "no `Here is your brief` preamble. Begin your response with `# `.\n"
)


def build_compose_messages(
    items: list[ComposeItem],
    *,
    plan_categories: list[CategoryPlan],
    coverage: Coverage,
    window_start: str,
    window_end: str,
) -> list[dict[str, str]]:
    """Construct the OpenAI-style messages list for one compose call.

    Deterministic and fixture-testable — no HTTP, no LLM, no DB. The
    output is the exact ``messages`` argument the orchestrator hands
    to :meth:`LLMClient.chat_completion`.

    ``plan_categories`` carries the editorial config so the model
    knows the canonical section ordering. Clusters whose
    ``ComposeItem.category`` doesn't match a configured plan slug
    are listed under an explicit "Uncategorized" group so the model
    can route them to the Watchlist section.

    Raises:
        ValueError: empty ``items``. The orchestrator routes the
            zero-items case to :func:`render_empty_brief` instead so
            the LLM call is skipped entirely.
    """
    if not items:
        raise ValueError("items must not be empty (use render_empty_brief instead)")

    system = _render_system_message(plan_categories)
    user = _render_user_message(
        items,
        plan_categories=plan_categories,
        coverage=coverage,
        window_start=window_start,
        window_end=window_end,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _render_system_message(plan_categories: list[CategoryPlan]) -> str:
    parts: list[str] = [_SYSTEM_HEADER, "", _BRIEF_STRUCTURE]
    if plan_categories:
        parts.append("")
        parts.append("Editorial categories configured for this run:")
        for cat in plan_categories:
            parts.append(_render_category(cat))
    parts.extend(["", _GROUNDING_POLICY, "", _RESPONSE_FORMAT])
    return "\n".join(parts)


def _render_category(category: CategoryPlan) -> str:
    line = f"- `{category.name}`"
    if category.description:
        line += f": {category.description}"
    return line


def _render_user_message(
    items: list[ComposeItem],
    *,
    plan_categories: list[CategoryPlan],
    coverage: Coverage,
    window_start: str,
    window_end: str,
) -> str:
    grouped = _group_by_category(items, plan_categories)
    lines: list[str] = [
        f"Brief window: {window_start} → {window_end}.",
        f"H1 date for the brief title: {window_end[:10]}.",
        "",
        "Coverage:",
        f"- Sources attempted: {coverage.sources_attempted}",
        f"- Sources succeeded: {coverage.sources_succeeded}",
        f"- Raw items: {coverage.raw_items}",
        f"- Event clusters: {coverage.clusters}",
        f"- Kept clusters: {coverage.kept_clusters}",
        f"- Summarized clusters: {coverage.summarized_clusters}",
    ]
    if coverage.failed_sources:
        lines.append(f"- Failed sources: {', '.join(coverage.failed_sources)}")
    if coverage.skipped_sources:
        lines.append(f"- Skipped sources: {'; '.join(coverage.skipped_sources)}")
    if coverage.plan_warnings:
        lines.append(f"- Plan warnings: {'; '.join(coverage.plan_warnings)}")
    if coverage.failed_filter_batches:
        lines.append(f"- Failed filter batches: {coverage.failed_filter_batches}")
    if coverage.failed_summary_clusters:
        lines.append(f"- Failed summary clusters: {coverage.failed_summary_clusters}")

    lines.extend(["", f"Cluster summaries ({len(items)} total, grouped by category):"])
    for group_name, group_items in grouped:
        if not group_items:
            continue
        lines.extend(["", f"### {group_name}"])
        for idx, item in enumerate(group_items, start=1):
            lines.append(_render_item(idx, item))
    return "\n".join(lines).rstrip() + "\n"


def _group_by_category(
    items: list[ComposeItem],
    plan_categories: list[CategoryPlan],
) -> list[tuple[str, list[ComposeItem]]]:
    """Group items by category slug, in the plan's declared order.

    Plan-ordered first (deterministic section ordering across re-
    runs), then a final ``Uncategorized`` bucket for items whose
    category slug doesn't appear in the plan. Within each group the
    caller's input order is preserved — by convention the DB loader
    sorts by ``relevance_score DESC, cluster_id ASC`` so each group
    is already in highest-signal-first order.
    """
    declared_order = [c.name for c in plan_categories]
    buckets: dict[str | None, list[ComposeItem]] = {name: [] for name in declared_order}
    uncategorized: list[ComposeItem] = []
    for item in items:
        if item.category and item.category in buckets:
            buckets[item.category].append(item)
        else:
            uncategorized.append(item)
    grouped: list[tuple[str, list[ComposeItem]]] = [
        (name, buckets[name]) for name in declared_order if buckets[name]
    ]
    if uncategorized:
        grouped.append(("Uncategorized", uncategorized))
    return grouped


def _render_item(idx: int, item: ComposeItem) -> str:
    parts: list[str] = []
    score_tag = (
        f"relevance={item.relevance_score:.2f}"
        if item.relevance_score is not None
        else "relevance=?"
    )
    conf_tag = (
        f"confidence={item.confidence:.2f}" if item.confidence is not None else "confidence=?"
    )
    parts.append(f"[{idx}] ({score_tag}, {conf_tag}) cluster_id={item.cluster_id}")
    parts.append(f"    headline: {item.headline}")
    parts.append(f"    summary: {item.summary}")
    if item.why_it_matters:
        parts.append(f"    why_it_matters: {item.why_it_matters}")
    if item.entities:
        parts.append(f"    entities: {', '.join(item.entities)}")
    if item.key_facts:
        parts.append("    key_facts:")
        for fact in item.key_facts:
            parts.append(f"      - {fact}")
    if item.caveats:
        parts.append("    caveats:")
        for caveat in item.caveats:
            parts.append(f"      - {caveat}")
    if item.source_urls:
        parts.append("    source_urls:")
        for url in item.source_urls:
            parts.append(f"      - {url}")
    return "\n".join(parts)


# ── Response normalization ────────────────────────────────────────────────────


def normalize_compose_output(content: str) -> str:
    """Strip predictable model artifacts from the compose response.

    Local models drift in two predictable ways under load:
    1. Prepending preamble like ``Here is the brief:\\n\\n# ...``
    2. Wrapping the brief in ```` ```markdown ... ``` ```` fences

    Both are addressed without prompt-engineering the model harder:
    fences are sliced off, and content before the first ``# ``
    heading is dropped. Anything after the last fence is dropped too.

    The non-empty + has-H1 invariants are the only structural checks
    — beyond those, brief quality is judged by humans reading it.

    Raises:
        ComposeOutputError: response empty after normalization, or
            no ``# `` heading found.
    """
    s = content.strip()
    if not s:
        raise ComposeOutputError("compose response was empty")

    if s.startswith("```"):
        newline_idx = s.find("\n")
        if newline_idx == -1:
            raise ComposeOutputError("compose response was a bare fence line")
        s = s[newline_idx + 1 :].rstrip()
        if s.endswith("```"):
            s = s[:-3].rstrip()
        if not s:
            raise ComposeOutputError("compose response was empty after fence stripping")

    if not s.startswith("# "):
        marker = s.find("\n# ")
        if marker == -1:
            raise ComposeOutputError("compose response has no '# ' heading")
        s = s[marker + 1 :]

    return s.strip()


# ── Deterministic fallbacks (no LLM call) ─────────────────────────────────────


def render_empty_brief(
    *,
    coverage: Coverage,
    window_start: str,
    window_end: str,
    run_id: int,
    brief_kind: str = "daily",
    query: str | None = None,
) -> str:
    """Render a deterministic 'no items today' brief without an LLM call.

    Triggered by :func:`compose_brief` when the summary stage produced
    zero rows — fetch fully degraded, filter rejected everything, or
    every cluster's summary call failed. Calling the LLM with no items
    would waste tokens; emitting nothing would lose the coverage
    audit trail. The architecture-doc rule "always publish a useful
    brief" is satisfied by a coverage-only document.

    ``brief_kind`` parameterizes the H1 line: ``"daily"`` produces
    ``"# Daily Intelligence Brief — <date>"``; ``"topic"`` produces
    ``"# Topic Brief: <query> — <date>"``. The empty-brief shape is
    identical otherwise — the topic-flavored prompt + section layout
    lands in Phase 7e once we have real items to compose with.
    """
    lines: list[str] = [
        _render_brief_h1(brief_kind=brief_kind, query=query, window_end=window_end),
        "",
        "## Executive Read",
        "",
        "No cluster summaries reached the composition stage for this run. "
        "The pipeline executed end-to-end; the coverage block below "
        "explains what was attempted.",
        "",
        "## Coverage",
        "",
        f"- Window: {window_start} → {window_end}",
        f"- Sources attempted: {coverage.sources_attempted}",
        f"- Sources succeeded: {coverage.sources_succeeded}",
        f"- Raw items: {coverage.raw_items}",
        f"- Event clusters: {coverage.clusters}",
        f"- Kept clusters: {coverage.kept_clusters}",
        f"- Summarized clusters: {coverage.summarized_clusters}",
    ]
    if coverage.failed_sources:
        lines.append(f"- Failed sources: {', '.join(coverage.failed_sources)}")
    if coverage.skipped_sources:
        lines.append(f"- Skipped sources: {'; '.join(coverage.skipped_sources)}")
    if coverage.plan_warnings:
        lines.append(f"- Plan warnings: {'; '.join(coverage.plan_warnings)}")
    if coverage.failed_filter_batches:
        lines.append(f"- Failed filter batches: {coverage.failed_filter_batches}")
    if coverage.failed_summary_clusters:
        lines.append(f"- Failed summary clusters: {coverage.failed_summary_clusters}")
    lines.extend(["", f"_Run id: {run_id}._"])
    return "\n".join(lines) + "\n"


def render_fallback_brief(
    items: list[ComposeItem],
    *,
    plan_categories: list[CategoryPlan],
    coverage: Coverage,
    window_start: str,
    window_end: str,
    run_id: int,
    failure_reason: str,
    brief_kind: str = "daily",
    query: str | None = None,
) -> str:
    """Render a degraded brief directly from cluster summaries (no LLM call).

    Triggered by :func:`compose_brief` when the final-compose LLM call
    fails after retries, or when its response fails normalization. The
    architecture-doc rule "failed sources degrade coverage; they do
    not fail the run" applies here too: a vMLX outage at the compose
    stage shouldn't lose every upstream stage's work. This emits a
    deterministic Markdown brief built from the structured
    ``item_summaries`` rows — less polished than an LLM-composed
    brief, but still grounded and citation-preserving.

    ``brief_kind`` + ``query`` parameterize the H1 line the same way
    :func:`render_empty_brief` does.
    """
    grouped = _group_by_category(items, plan_categories)
    h1 = _render_brief_h1(brief_kind=brief_kind, query=query, window_end=window_end)
    lines: list[str] = [
        f"{h1} (degraded)",
        "",
        "_Final composition failed; this brief was rendered directly "
        "from the structured cluster summaries without prose synthesis. "
        f"Reason: {failure_reason}._",
        "",
    ]
    for group_name, group_items in grouped:
        section_title = _humanize_category(group_name)
        lines.extend([f"## {section_title}", ""])
        for item in group_items:
            lines.append(f"### {item.headline}")
            lines.append("")
            lines.append(item.summary)
            if item.why_it_matters:
                lines.extend(["", f"_Why it matters:_ {item.why_it_matters}"])
            if item.key_facts:
                lines.append("")
                for fact in item.key_facts:
                    lines.append(f"- {fact}")
            if item.caveats:
                lines.append("")
                lines.append("_Caveats:_")
                for caveat in item.caveats:
                    lines.append(f"- {caveat}")
            if item.source_urls:
                lines.append("")
                lines.append("Sources: " + ", ".join(item.source_urls))
            lines.append("")

    lines.extend(["## Coverage", ""])
    lines.append(f"- Window: {window_start} → {window_end}")
    lines.append(f"- Sources attempted: {coverage.sources_attempted}")
    lines.append(f"- Sources succeeded: {coverage.sources_succeeded}")
    lines.append(f"- Raw items: {coverage.raw_items}")
    lines.append(f"- Event clusters: {coverage.clusters}")
    lines.append(f"- Kept clusters: {coverage.kept_clusters}")
    lines.append(f"- Summarized clusters: {coverage.summarized_clusters}")
    if coverage.failed_sources:
        lines.append(f"- Failed sources: {', '.join(coverage.failed_sources)}")
    if coverage.skipped_sources:
        lines.append(f"- Skipped sources: {'; '.join(coverage.skipped_sources)}")
    if coverage.plan_warnings:
        lines.append(f"- Plan warnings: {'; '.join(coverage.plan_warnings)}")
    if coverage.failed_filter_batches:
        lines.append(f"- Failed filter batches: {coverage.failed_filter_batches}")
    if coverage.failed_summary_clusters:
        lines.append(f"- Failed summary clusters: {coverage.failed_summary_clusters}")
    lines.extend(["", f"_Run id: {run_id}._"])
    return "\n".join(lines) + "\n"


def _render_brief_h1(*, brief_kind: str, query: str | None, window_end: str) -> str:
    """Build the H1 line for the deterministic brief paths.

    Daily briefs are date-keyed; topic briefs lead with the query so
    the operator can scan a list of past topic runs without opening
    each one. The window-end date trails the topic title so the brief
    is still time-attributable. Empty / whitespace-only query falls
    through to the date-only form to keep the brief publishable even
    if the CLI somehow handed us a blank query string.
    """
    date = window_end[:10]
    if brief_kind == "topic" and query and query.strip():
        return f"# Topic Brief: {query.strip()} — {date}"
    return f"# Daily Intelligence Brief — {date}"


def _humanize_category(slug: str) -> str:
    """Convert ``startup_funding`` → ``Startup Funding`` for section titles.

    The fallback path bypasses the LLM, so the title-casing the model
    would normally do happens here deterministically. Pure helper —
    no special-case maps; that keeps the door open for any new
    category slug to render automatically.
    """
    if not slug or slug == "Uncategorized":
        return "Uncategorized"
    return " ".join(part.capitalize() for part in slug.replace("-", "_").split("_"))


# ── Async orchestration ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ComposeResult:
    """Outcome of one compose run, surfaced to the orchestrator.

    ``provider_tag`` is what gets stamped onto
    ``metadata.composition_provider``:

    - ``"gemini_cli"`` — Tier 1 frontier compose succeeded
      (architecture-doc target after the 2026-05-15 amendment).
    - ``"vmlx_fallback"`` — Either Tier 1 routed to vMLX directly
      (no gemini_cli configured in the YAML), or Tier 2 fell back
      to vMLX after a Tier 1 (gemini_cli) failure. The label is
      shared because the operator-facing meaning is the same: a
      local model produced the brief.
    - ``"local_stub_failed"`` — Tier 3 deterministic
      ``render_fallback_brief`` ran because both Tier 1 and Tier 2
      (when configured) failed.
    - ``"local_stub_empty"`` — Zero summarized clusters reached
      compose; deterministic empty-brief stub was emitted, no LLM
      call. (Unchanged from Phase 1.)

    Persistent ``"vmlx_fallback"`` days after Step 12b indicates the
    Gemini CLI Tier 1 is failing repeatedly — a signal worth
    investigating from the dashboard rather than letting it drift
    into a quiet-degraded daily routine.
    """

    markdown: str
    provider_tag: str
    model: str


async def compose_brief(
    conn: sqlite3.Connection,
    run_id: int,
    llm_client: LLMClient,
    *,
    plan: SourcePlan,
    coverage: Coverage,
    window_start: str,
    window_end: str,
    model: str,
    prompt_version: str = PROMPT_VERSION,
    brief_kind: str = "daily",
    query: str | None = None,
) -> ComposeResult:
    """Produce one Markdown daily brief for *run_id*.

    Three-tier resilience chain (Step 12b):

    1. **Empty short-circuit.** Zero items → :func:`render_empty_brief`
       directly, no LLM call. Tag: ``"local_stub_empty"``.
    2. **Tier 1** — primary stage (``final_compose``). With the
       2026-05-15 amendment this routes through ``gemini_cli`` to
       Gemini 3 Pro, but a vmlx-only deployment is still supported.
       On success: tag ``"gemini_cli"`` or ``"vmlx_fallback"``
       depending on the routed provider.
    3. **Tier 2** — fallback stage if the primary failed AND a
       ``fallback`` block is declared on the ``final_compose`` stage
       in ``model-routing.yaml``. The architecture doc routes this
       to the strongest cached local vMLX model. On success: tag
       ``"vmlx_fallback"``. Skipped entirely when no fallback is
       declared (Tier 1 failure goes straight to Tier 3).
    4. **Tier 3** — deterministic :func:`render_fallback_brief`
       built directly from the cluster summaries. No LLM call.
       Tag: ``"local_stub_failed"``. The architecture-doc rule
       "always publish a useful brief" lives here.

    Returns a :class:`ComposeResult` carrying the Markdown plus the
    provider tag the orchestrator stamps onto metadata.
    """
    items = _load_compose_items(conn, run_id)

    if not items:
        log.info("run %d: compose has 0 items; emitting empty-brief stub", run_id)
        markdown = render_empty_brief(
            coverage=coverage,
            window_start=window_start,
            window_end=window_end,
            run_id=run_id,
            brief_kind=brief_kind,
            query=query,
        )
        return ComposeResult(markdown=markdown, provider_tag="local_stub_empty", model=model)

    plan_categories = list(plan.categories)
    primary_stage = llm_client.routing.resolve("final_compose")
    # Capture the primary failure outside the except block so the
    # else-branch below can build the Tier-3 failure_reason from it.
    # Python clears exception variables at except-scope exit.
    primary_failure: BaseException | None = None

    # Tier 1: primary stage.
    try:
        markdown = await _compose_via_llm(
            llm_client,
            items=items,
            plan_categories=plan_categories,
            coverage=coverage,
            window_start=window_start,
            window_end=window_end,
            prompt_version=prompt_version,
            stage_config_override=None,
        )
        tier1_tag = _provider_tag_for(primary_stage)
        return ComposeResult(
            markdown=markdown,
            provider_tag=tier1_tag,
            model=primary_stage.model,
        )
    except Exception as primary_exc:
        primary_failure = primary_exc
        log.warning(
            "run %d: Tier 1 final-compose (%s) failed (%s); trying fallback",
            run_id,
            primary_stage.provider,
            primary_exc,
        )

    # Tier 2: fallback stage if configured.
    if primary_stage.fallback is not None:
        fallback_stage = _stage_from_fallback(primary_stage.fallback, primary_stage)
        try:
            markdown = await _compose_via_llm(
                llm_client,
                items=items,
                plan_categories=plan_categories,
                coverage=coverage,
                window_start=window_start,
                window_end=window_end,
                prompt_version=prompt_version,
                stage_config_override=fallback_stage,
            )
            return ComposeResult(
                markdown=markdown,
                provider_tag="vmlx_fallback",
                model=fallback_stage.model,
            )
        except Exception as fallback_exc:
            log.warning(
                "run %d: Tier 2 fallback compose (%s) also failed (%s); rendering deterministic brief",
                run_id,
                fallback_stage.provider,
                fallback_exc,
            )
            failure_reason = _short_failure(fallback_exc)
    else:
        assert primary_failure is not None
        failure_reason = _short_failure(primary_failure)

    # Tier 3: deterministic fallback brief.
    markdown = render_fallback_brief(
        items,
        plan_categories=plan_categories,
        coverage=coverage,
        window_start=window_start,
        window_end=window_end,
        run_id=run_id,
        failure_reason=failure_reason,
        brief_kind=brief_kind,
        query=query,
    )
    return ComposeResult(
        markdown=markdown,
        provider_tag="local_stub_failed",
        model=model,
    )


def _provider_tag_for(stage_config: StageConfig) -> str:
    """Map a stage's provider to the metadata tag the dashboard reads.

    ``gemini_cli`` → ``"gemini_cli"`` (frontier compose ran).
    ``vmlx`` → ``"vmlx_fallback"`` (local model ran; either by config
    choice or because frontier was unconfigured in this deployment).
    The shared ``"vmlx_fallback"`` tag for both "vmlx as primary" and
    "vmlx as Tier 2 fallback" is intentional: operationally an
    operator reading the dashboard cares whether frontier was used,
    not which tier produced a non-frontier result.
    """
    if stage_config.provider == "gemini_cli":
        return "gemini_cli"
    return "vmlx_fallback"


def _stage_from_fallback(
    fallback: FallbackConfig,
    parent: StageConfig,
) -> StageConfig:
    """Build a synthetic ``StageConfig`` from a parent's fallback block.

    ``timeout_seconds`` falls through to the parent's value when the
    fallback doesn't override it. Other parent-only fields
    (``batch_size``, ``retries``, the recursive ``fallback``) are
    deliberately dropped — Tier 2 doesn't recurse, doesn't batch, and
    uses the provider's default retry policy.
    """
    return StageConfig(
        provider=fallback.provider,
        model=fallback.model,
        timeout_seconds=fallback.timeout_seconds or parent.timeout_seconds,
    )


async def _compose_via_llm(
    llm_client: LLMClient,
    *,
    items: list[ComposeItem],
    plan_categories: list[CategoryPlan],
    coverage: Coverage,
    window_start: str,
    window_end: str,
    prompt_version: str,
    stage_config_override: StageConfig | None,
) -> str:
    """Run one compose call and normalize the response.

    ``stage_config_override`` is forwarded into
    :meth:`LLMClient.chat_completion` so a Tier 2 call dispatches
    against the fallback provider/model without needing a separate
    YAML stage entry. Raises on any LLM or normalization failure —
    the caller decides whether to try the next tier.
    """
    messages = build_compose_messages(
        items,
        plan_categories=plan_categories,
        coverage=coverage,
        window_start=window_start,
        window_end=window_end,
    )
    result = await llm_client.chat_completion(
        stage="final_compose",
        messages=messages,
        prompt_version=prompt_version,
        temperature=_COMPOSE_TEMPERATURE,
        max_tokens=_COMPOSE_MAX_TOKENS,
        stage_config_override=stage_config_override,
    )
    return normalize_compose_output(result.content)


def _load_compose_items(conn: sqlite3.Connection, run_id: int) -> list[ComposeItem]:
    """Hydrate :class:`ComposeItem` instances from the DB loader rows.

    The JSON-list TEXT columns are deserialized here so the pure
    helpers always see typed Python tuples. Defensive: a malformed
    JSON column (e.g. legacy row written by an older code path)
    is logged and treated as an empty list — better to surface a
    partial cluster than to abort the whole brief.
    """
    items: list[ComposeItem] = []
    for row in db.iter_summarized_clusters_with_summary(conn, run_id):
        items.append(
            ComposeItem(
                cluster_id=int(row["cluster_id"]),
                category=row["category"],
                relevance_score=row["relevance_score"],
                headline=row["headline"] or "",
                summary=row["summary"] or "",
                why_it_matters=row["why_it_matters"] or "",
                entities=tuple(_load_json_list(row["entities"])),
                key_facts=tuple(_load_json_list(row["key_facts"])),
                caveats=tuple(_load_json_list(row["caveats"])),
                confidence=row["confidence"],
                source_urls=tuple(_load_json_list(row["source_urls"])),
            )
        )
    return items


def _load_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("compose: invalid JSON-list column %r; treating as empty", raw[:60])
        return []
    if not isinstance(decoded, list):
        log.warning(
            "compose: JSON-list column was %s, not list; treating as empty",
            type(decoded).__name__,
        )
        return []
    return [str(x) for x in decoded]


def _short_failure(exc: BaseException) -> str:
    """One-line description of an exception for the degraded-brief notice."""
    name = type(exc).__name__
    msg = str(exc)
    if not msg:
        return name
    if len(msg) > 200:
        msg = msg[:197] + "..."
    return f"{name}: {msg}"


__all__ = (
    "PROMPT_VERSION",
    "ComposeItem",
    "ComposeOutputError",
    "ComposeResult",
    "build_compose_messages",
    "compose_brief",
    "normalize_compose_output",
    "render_empty_brief",
    "render_fallback_brief",
)
