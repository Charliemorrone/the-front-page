"""Relevance filter for the filtering lifecycle stage.

Promotes clusters from ``status='pending'`` to ``'kept'`` (with a
``category``, ``relevance_score``, ``event_type``, ``filter_reason``) or
to ``'filtered_out'``. Batched LLM call per ``StageConfig.batch_size``
(default 12 per the architecture doc). The model chooses categories
from the editorial list in :file:`config/intel-sources.yaml`.

This module ships the **pure layer** (step 9a):

- :class:`RelevanceCluster` — the input shape the orchestrator hands
  the pure helper after joining ``item_clusters → cluster_items →
  raw_items``.
- :func:`build_relevance_messages` — deterministic OpenAI-style
  messages list. System message lays out the categories and the
  JSON contract; user message enumerates the batch in order.
- :func:`parse_relevance_verdicts` — unpacks the pydantic-validated
  :class:`RelevanceBatchResponse` and asserts the verdict count
  matches the input batch (positional verdict assignment is meaningless
  otherwise — fail loud rather than mis-apply).

The async orchestration layer (``filter_clusters``) and the
``Coverage.failed_filter_batches`` counter land in step 9b.

The architecture-doc rule the prompt is designed around: **this stage
must be allowed to keep many items. It is not a top-N selector.** If
the filter rejects most clusters from a typical run, the prompt is
wrong, not the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.schemas import RelevanceBatchResponse, RelevanceVerdict
from ..sources import CategoryPlan

PROMPT_VERSION = "relevance.v1"


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


__all__ = (
    "PROMPT_VERSION",
    "RelevanceCluster",
    "build_relevance_messages",
    "parse_relevance_verdicts",
)
