"""Relevance filter for the filtering lifecycle stage.

Promotes clusters from ``status='pending'`` to ``'kept'`` (with a
``category``, ``relevance_score``, ``event_type``, ``filter_reason``) or
to ``'filtered_out'``. Batched LLM call per ``StageConfig.batch_size``
(default 12 per the architecture doc). The model chooses categories
from the editorial list in :file:`config/intel-sources.yaml`.

Two-layer split, mirroring the fetcher and clustering modules:

- **Pure layer.** :class:`RelevanceCluster`,
  :func:`build_relevance_messages`, :func:`parse_relevance_verdicts`,
  and :data:`PROMPT_VERSION`. Fixture-testable without HTTP / DB / LLM.
- **Async orchestration.** :func:`filter_clusters` loads pending
  clusters, batches them, calls :meth:`LLMClient.chat_completion` with
  the :class:`RelevanceBatchResponse` schema, applies verdicts via
  :func:`db.update_cluster_verdict`. Per-batch LLM failures degrade
  coverage rather than crashing the run — the architecture-doc
  "failed sources degrade coverage; they do not fail the run" rule
  extended to LLM stages.

The architecture-doc rule the prompt is designed around: **this stage
must be allowed to keep many items. It is not a top-N selector.** If
the filter rejects most clusters from a typical run, the prompt is
wrong, not the threshold.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

from .. import db
from ..llm import LLMClient
from ..llm.schemas import RelevanceBatchResponse, RelevanceVerdict
from ..runs import Coverage
from ..sources import CategoryPlan

log = logging.getLogger(__name__)

PROMPT_VERSION = "relevance.v1"

# Structured outputs need a low temperature to keep JSON well-formed under
# load. Pinned at the call site (not in routing YAML) because it's a
# property of the *prompt* — the relevance prompt is JSON-mode; a future
# free-form planner prompt against the same model wouldn't want this.
_RELEVANCE_TEMPERATURE = 0.1

# Local MLX servers default to ~1024 completion tokens, which truncates
# batched verdict responses mid-JSON. A 12-cluster batch with verbose
# `reason` fields can run ~200 tokens/verdict — budget generously to
# leave headroom and avoid the schema-repair retry path under normal
# load. ~250 tokens/verdict × 12 verdicts = 3000, plus envelope: 4096
# gives comfortable margin without imposing a memory burden on the
# server.
_RELEVANCE_MAX_TOKENS_PER_VERDICT = 320


# ── Input shape ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RelevanceCluster:
    """One cluster's prompt-relevant shape.

    Populated by the orchestrator from the ``item_clusters →
    cluster_items → raw_items`` join. The pure-layer helpers consume
    this; they do not touch SQLite.

    ``canonical_url`` is the representative member's URL (smallest-id
    member, matching the clustering layer's tie-break rule).
    ``member_urls`` are every distinct URL in the cluster — for an L2 or
    L3 fold this exposes the cross-source evidence the LLM can use to
    decide whether the cluster is over- or under-folded.

    ``excerpt`` is the smallest-id member's excerpt. Fetchers already
    cap excerpts at ~320 chars, so this is bounded in practice. For L2/
    L3 clusters that span multiple distinct excerpts, only one shows
    here; if 9b's smoke turns up clusters being filtered ambiguously,
    concatenating a few member excerpts is the cheapest next step.
    """

    cluster_id: int
    title: str
    canonical_url: str
    member_urls: tuple[str, ...] = ()
    excerpt: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


# ── Prompt construction ───────────────────────────────────────────────────────


_SYSTEM_HEADER = (
    "You are a relevance filter for a personal daily intelligence brief.\n"
    "\n"
    "For each cluster in the batch you receive, decide whether it belongs in "
    "today's brief (`keep=true`) or should be rejected as feed noise "
    "(`keep=false`). Each cluster represents one event with one or more "
    "articles describing it.\n"
)

_KEEP_POLICY = (
    "Keep policy:\n"
    "- Keep every cluster that meaningfully fits any configured category. "
    "This is NOT a top-N selector. If 40 clusters fit, keep 40.\n"
    "- The final brief includes every relevant kept cluster; ranking affects "
    "ordering, not inclusion.\n"
    "- Reject only clear feed noise: SEO listicles, opinion posts with no new "
    "information, recirculated old news, off-topic content.\n"
)

_RESPONSE_FORMAT = (
    "Response format:\n"
    "Reply with a single JSON object containing a `verdicts` array. Each "
    "verdict MUST appear in the same order as the input clusters and the "
    "array length MUST equal the input batch size. Verdict shape:\n"
    "{\n"
    '  "verdicts": [\n'
    "    {\n"
    '      "keep": true,\n'
    '      "category": "<one of the configured category names>",\n'
    '      "score": 0.0,\n'
    '      "event_type": "<short identifier or null>",\n'
    '      "reason": "<one sentence>",\n'
    '      "entities": ["<company/product/person>", "..."],\n'
    '      "evidence_urls": ["<url>", "..."],\n'
    '      "uncertainty": 0.0\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "Field rules:\n"
    "- `category`: one of the configured category names below. Use the exact "
    "name as written.\n"
    "- `score`: confidence the cluster belongs in the brief, 0.0 - 1.0.\n"
    "- `uncertainty`: 0.0 - 1.0, optional (null when not borderline).\n"
    "- `event_type`: short snake_case identifier (e.g. funding_round, "
    "model_release, repo_release) or null.\n"
    "- `entities`: notable companies, products, or people in the cluster.\n"
    "- `evidence_urls`: up to 3 URLs drawn from the input that best evidence "
    "the verdict.\n"
    "\n"
    "Reply with valid JSON only — no markdown fencing, no commentary, no "
    "preamble. Begin your response with `{`."
)


def build_relevance_messages(
    clusters_batch: list[RelevanceCluster],
    categories: list[CategoryPlan],
) -> list[dict[str, str]]:
    """Construct the OpenAI-style messages list for one batch.

    Deterministic and fixture-testable — no HTTP, no LLM, no DB. The
    output is the exact ``messages`` argument the orchestrator hands
    to :meth:`LLMClient.chat_completion`.

    Raises:
        ValueError: empty batch (the LLM call would have nothing to
            decide, and the count-check in
            :func:`parse_relevance_verdicts` would fail anyway).
    """
    if not clusters_batch:
        raise ValueError("clusters_batch must not be empty")

    system = _render_system_message(categories)
    user = _render_user_message(clusters_batch)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _render_system_message(categories: list[CategoryPlan]) -> str:
    parts: list[str] = [_SYSTEM_HEADER, "", "Editorial categories:"]
    if not categories:
        # Should never happen in production — the resolver always
        # materializes at least one category. But fall back cleanly so
        # tests can stress the empty-categories edge case.
        parts.append("- (none configured)")
    else:
        for cat in categories:
            parts.append(_render_category(cat))
    parts.extend(["", _KEEP_POLICY, "", _RESPONSE_FORMAT])
    return "\n".join(parts)


def _render_category(category: CategoryPlan) -> str:
    lines = [f"- {category.name}"]
    if category.description:
        lines.append(f"  description: {category.description}")
    if category.include_rules:
        lines.append("  include:")
        for rule in category.include_rules:
            lines.append(f"    - {rule}")
    if category.exclude_rules:
        lines.append("  exclude:")
        for rule in category.exclude_rules:
            lines.append(f"    - {rule}")
    return "\n".join(lines)


def _render_user_message(clusters: list[RelevanceCluster]) -> str:
    lines = [
        f"Batch of {len(clusters)} clusters. Return verdicts in the same order.",
        "",
    ]
    for idx, cluster in enumerate(clusters, start=1):
        lines.append(f"[{idx}] {cluster.title or '(untitled)'}")
        if cluster.canonical_url:
            lines.append(f"    url: {cluster.canonical_url}")
        if cluster.excerpt:
            lines.append(f"    excerpt: {cluster.excerpt}")
        extra_urls = _extra_member_urls(cluster)
        if extra_urls:
            lines.append("    additional source urls:")
            for url in extra_urls:
                lines.append(f"      - {url}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _extra_member_urls(cluster: RelevanceCluster) -> list[str]:
    """Member URLs other than the representative, deduped, order preserved.

    Surfacing these in the prompt lets the LLM see cross-source
    corroboration on L2/L3-folded clusters without inflating the
    payload — most clusters have one member, so the section is empty.
    """
    seen: set[str] = {cluster.canonical_url} if cluster.canonical_url else set()
    out: list[str] = []
    for url in cluster.member_urls:
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


# ── Response parsing ──────────────────────────────────────────────────────────


def parse_relevance_verdicts(
    parsed: RelevanceBatchResponse,
    *,
    expected_count: int,
) -> list[RelevanceVerdict]:
    """Unpack verdicts; assert the count matches the batch.

    The pydantic schema already validated each verdict's shape and
    numeric bounds at :meth:`LLMClient.chat_completion` time, so this
    helper only enforces the structural invariant pydantic can't
    express: the LLM must emit one verdict per input cluster, in order.
    A count mismatch makes positional verdict assignment meaningless,
    so we raise rather than silently apply wrong verdicts.

    Raises:
        ValueError: ``len(parsed.verdicts) != expected_count``.
    """
    verdicts = list(parsed.verdicts)
    if len(verdicts) != expected_count:
        raise ValueError(f"verdict count mismatch: expected {expected_count}, got {len(verdicts)}")
    return verdicts


# ── Async orchestration ──────────────────────────────────────────────────────


async def filter_clusters(
    conn: sqlite3.Connection,
    run_id: int,
    llm_client: LLMClient,
    coverage: Coverage,
    *,
    categories: list[CategoryPlan],
    batch_size: int = 12,
    prompt_version: str = PROMPT_VERSION,
) -> int:
    """Apply relevance verdicts to every pending cluster in *run_id*.

    Walks pending clusters in id order, batches by *batch_size*, and
    judges each batch via one :meth:`LLMClient.chat_completion` call.
    Verdicts are applied immediately so a mid-stage crash leaves a
    coherent partial result rather than an all-or-nothing rollback.

    Per-batch failure model (the architecture-doc "failed sources
    degrade coverage" rule extended to LLM stages): when an LLM call
    raises after retries / repair, the batch's clusters stay at
    ``status='pending'``, :meth:`Coverage.record_failed_filter_batch`
    increments, and the run continues. Those clusters won't reach
    summary or compose — they simply don't make it into today's brief.

    Per-cluster failure model: if applying a verdict raises (concurrent
    delete, schema drift, etc.), the cluster is skipped with a logged
    warning. Sibling clusters in the batch are unaffected.

    Returns the number of clusters promoted to ``'kept'``.
    """
    clusters = _load_pending_clusters(conn, run_id)
    if not clusters:
        return 0

    kept = 0
    for batch in _batched(clusters, batch_size):
        try:
            verdicts = await _judge_batch(
                llm_client, batch, categories, prompt_version=prompt_version
            )
        except Exception as exc:
            log.warning(
                "run %d: relevance batch failed (%d clusters left pending): %s",
                run_id,
                len(batch),
                exc,
            )
            coverage.record_failed_filter_batch()
            continue

        kept += _apply_verdicts(conn, batch, verdicts)

    return kept


async def _judge_batch(
    llm_client: LLMClient,
    batch: list[RelevanceCluster],
    categories: list[CategoryPlan],
    *,
    prompt_version: str,
) -> list[RelevanceVerdict]:
    """Run one batch through the LLM and unpack the verdict list."""
    messages = build_relevance_messages(batch, categories)
    # Size the completion budget to the batch — small batches don't need
    # the full ceiling; large batches absolutely do.
    max_tokens = _RELEVANCE_MAX_TOKENS_PER_VERDICT * len(batch) + 256
    result = await llm_client.chat_completion(
        stage="relevance_filter",
        messages=messages,
        response_schema=RelevanceBatchResponse,
        prompt_version=prompt_version,
        temperature=_RELEVANCE_TEMPERATURE,
        max_tokens=max_tokens,
    )
    # ``response_schema`` populates ``parsed`` — guaranteed non-None
    # here, but a defensive cast keeps the type-checker happy.
    parsed = result.parsed
    if not isinstance(parsed, RelevanceBatchResponse):
        raise RuntimeError(
            f"LLMClient returned parsed={type(parsed).__name__}; expected RelevanceBatchResponse"
        )
    return parse_relevance_verdicts(parsed, expected_count=len(batch))


def _apply_verdicts(
    conn: sqlite3.Connection,
    batch: list[RelevanceCluster],
    verdicts: list[RelevanceVerdict],
) -> int:
    """Write verdicts back to ``item_clusters``; return kept count."""
    kept = 0
    for cluster, verdict in zip(batch, verdicts, strict=True):
        status = "kept" if verdict.keep else "filtered_out"
        try:
            db.update_cluster_verdict(
                conn,
                cluster_id=cluster.cluster_id,
                status=status,
                relevance_score=verdict.score,
                category=verdict.category,
                event_type=verdict.event_type,
                filter_reason=verdict.reason,
            )
        except Exception as exc:
            log.warning(
                "failed to apply verdict to cluster %d: %s",
                cluster.cluster_id,
                exc,
            )
            continue
        if verdict.keep:
            kept += 1
    return kept


def _load_pending_clusters(
    conn: sqlite3.Connection,
    run_id: int,
) -> list[RelevanceCluster]:
    """Hydrate :class:`RelevanceCluster` instances for every pending cluster.

    The smallest-id member is the representative: its canonical_url and
    excerpt seed the cluster's prompt fields. All member canonical URLs
    flow into ``member_urls`` so L2/L3 folds can be inspected as
    cross-source evidence by the LLM.
    """
    out: list[RelevanceCluster] = []
    for cluster_id, title, members in db.iter_pending_clusters_with_members(conn, run_id):
        if not members:
            continue
        representative = members[0]
        member_urls = tuple((m["canonical_url"] or "") for m in members if m["canonical_url"])
        out.append(
            RelevanceCluster(
                cluster_id=cluster_id,
                title=title,
                canonical_url=representative["canonical_url"] or "",
                excerpt=representative["excerpt"] or "",
                member_urls=member_urls,
            )
        )
    return out


def _batched(items: list[RelevanceCluster], size: int) -> list[list[RelevanceCluster]]:
    """Split *items* into contiguous batches of at most *size* elements.

    Done inline rather than via :func:`itertools.batched` for Python
    3.11 compatibility — the project pins 3.12 but the worker pyproject
    declares ``requires-python = ">=3.12"`` only because the type
    annotations use 3.10+ syntax, and the rest of the codebase avoids
    3.12-only stdlib niceties.
    """
    if size <= 0:
        raise ValueError(f"batch size must be positive, got {size}")
    return [items[i : i + size] for i in range(0, len(items), size)]


__all__ = (
    "PROMPT_VERSION",
    "RelevanceCluster",
    "build_relevance_messages",
    "filter_clusters",
    "parse_relevance_verdicts",
)
