"""Cluster summary for the summarizing lifecycle stage.

Promotes clusters from ``status='kept'`` to ``'summarized'`` by writing
one ``item_summaries`` row per cluster: a grounded, citation-aware
condensation that the final composer (step 11) weaves into the brief.

Two-layer split, mirroring :mod:`pipeline.relevance`:

- **Pure layer.** :class:`SummaryMember`, :class:`SummaryCluster`,
  :func:`build_summary_messages`, :func:`parse_summary`, and
  :data:`PROMPT_VERSION`. Fixture-testable without HTTP / DB / LLM.
- **Async orchestration.** Lands in step 10b. The pure layer is shaped
  so the orchestration is a thin loop: one ``chat_completion`` per
  cluster, write the row, advance status.

Architecture-doc rules baked into the prompt:

- The summary must be **grounded in the cluster's source items**. The
  system message forbids invented facts and asks the model to preserve
  member URLs as citations.
- One call per cluster, not one call per batch. Cluster summaries don't
  share well across batches because each model call needs the cluster's
  full member content as context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.schemas import ClusterSummaryPayload
from ..sources import CategoryPlan

PROMPT_VERSION = "summary.v1"


# ── Input shape ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SummaryMember:
    """One raw item attached to a cluster, as visible to the summary prompt.

    Populated by the orchestrator from the ``item_clusters →
    cluster_items → raw_items`` join. The pure-layer helpers consume
    these directly so they don't need a SQLite connection.

    ``content`` is the full extracted body when available (RSS
    trafilatura output, HN selftext, Reddit selftext, arXiv abstract,
    website body); ``excerpt`` is the fetcher-supplied prefix
    (~320 chars). For sources where ``content`` is empty (GDELT
    ArtList, GitHub repo entries), only ``excerpt`` and ``title`` carry
    signal — and that's enough for a kept-cluster summary.
    """

    title: str
    canonical_url: str
    excerpt: str = ""
    content: str = ""
    author: str = ""
    published_at: str | None = None


@dataclass(frozen=True)
class SummaryCluster:
    """One cluster's prompt-relevant shape for the summary stage.

    The first element of ``members`` is the representative (smallest-id
    member, matching the clustering layer's tie-break rule). For an L2
    or L3 fold ``members`` carries every member so the summary can
    surface cross-source evidence and citations.

    ``category`` is the slug written by the relevance filter (e.g.
    ``"ai_research"``); ``None`` is permitted defensively so a cluster
    that lost its category column for any reason doesn't blow up the
    summary call — the prompt falls back to a generic framing.
    """

    cluster_id: int
    title: str
    category: str | None
    members: tuple[SummaryMember, ...] = field(default_factory=tuple)


# ── Prompt construction ───────────────────────────────────────────────────────


_SYSTEM_HEADER = (
    "You are a summarizer for a personal daily intelligence brief.\n"
    "\n"
    "You will receive ONE event cluster with its supporting source items "
    "(titles, URLs, excerpts, and extracted body text). Your job is to "
    "produce a single grounded summary the final composer can weave into "
    "today's brief.\n"
)

_GROUNDING_POLICY = (
    "Grounding policy (load-bearing):\n"
    "- Every fact, name, number, and claim in your summary MUST be present "
    "in the supplied source items. Do not invent details that aren't "
    "supported by the input.\n"
    "- Preserve citations: list the cluster's source URLs in `source_urls` "
    "so the final brief can link back to evidence.\n"
    "- If the sources disagree, capture the disagreement in `caveats` "
    "rather than picking a side.\n"
    "- If the cluster is thin (one source, sparse text), keep the summary "
    "short and surface that thinness in `confidence` and `caveats`.\n"
)

_RESPONSE_FORMAT = (
    "Response format:\n"
    "Reply with a single JSON object with this shape:\n"
    "{\n"
    '  "headline": "<one-line factual headline, no editorial spin>",\n'
    '  "summary": "<2-4 sentence factual condensation of the event>",\n'
    '  "why_it_matters": "<one sentence on relevance to the configured '
    'category, or empty string>",\n'
    '  "entities": ["<company/product/person>", "..."],\n'
    '  "key_facts": ["<specific fact with number, date, or named entity>", "..."],\n'
    '  "caveats": ["<uncertainty, contradiction, missing context>", "..."],\n'
    '  "source_urls": ["<url>", "..."],\n'
    '  "confidence": 0.0\n'
    "}\n"
    "\n"
    "Field rules:\n"
    "- `headline` and `summary` are required. The rest may be empty arrays "
    "or empty strings when the cluster doesn't support them. `confidence` "
    "may be null if you can't estimate it.\n"
    "- `confidence`: 0.0 - 1.0, how well-grounded the summary is in the "
    "supplied sources.\n"
    "- `source_urls`: draw from the URLs in the cluster's source items. "
    "Don't invent URLs.\n"
    "\n"
    "Reply with valid JSON only — no markdown fencing, no commentary, no "
    "preamble. Begin your response with `{`."
)


def build_summary_messages(
    cluster: SummaryCluster,
    category: CategoryPlan | None = None,
) -> list[dict[str, str]]:
    """Construct the OpenAI-style messages list for one cluster.

    Deterministic and fixture-testable — no HTTP, no LLM, no DB. The
    output is the exact ``messages`` argument the orchestrator hands to
    :meth:`LLMClient.chat_completion`. ``category`` is the editorial
    :class:`CategoryPlan` matching :attr:`SummaryCluster.category`; pass
    ``None`` when the cluster has no category or when the resolver
    couldn't materialize it (the prompt falls back to a generic framing
    rather than failing the call).

    Raises:
        ValueError: cluster has no members. A clusterless cluster has
            nothing to summarize and the LLM call would be wasted.
    """
    if not cluster.members:
        raise ValueError(f"cluster {cluster.cluster_id} has no members")

    system = _render_system_message(category)
    user = _render_user_message(cluster)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _render_system_message(category: CategoryPlan | None) -> str:
    parts: list[str] = [_SYSTEM_HEADER]
    if category is not None:
        parts.extend(["", "Category context:", _render_category(category)])
    parts.extend(["", _GROUNDING_POLICY, "", _RESPONSE_FORMAT])
    return "\n".join(parts)


def _render_category(category: CategoryPlan) -> str:
    lines = [f"- name: {category.name}"]
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


def _render_user_message(cluster: SummaryCluster) -> str:
    header = f"Cluster: {cluster.title or '(untitled)'}"
    lines: list[str] = [header]
    if cluster.category:
        lines.append(f"Filed under category: {cluster.category}")
    lines.extend(["", f"Source items ({len(cluster.members)}):", ""])
    for idx, member in enumerate(cluster.members, start=1):
        lines.append(f"[{idx}] {member.title or '(untitled)'}")
        if member.canonical_url:
            lines.append(f"    url: {member.canonical_url}")
        if member.author:
            lines.append(f"    author: {member.author}")
        if member.published_at:
            lines.append(f"    published_at: {member.published_at}")
        body = member.content or member.excerpt
        if body:
            lines.append("    body:")
            lines.append(_indent_body(body))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _indent_body(body: str) -> str:
    """Indent each line of *body* by six spaces so it nests under ``body:``.

    Whitespace-collapsed inside lines so a malformed extractor output
    (e.g. a giant single line with embedded newlines) still renders
    readably. We don't truncate here — the LLM call's ``max_tokens``
    budget governs how much body the model can spend tokens digesting,
    not the prompt construction.
    """
    out: list[str] = []
    for raw in body.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        out.append(f"      {stripped}")
    return "\n".join(out) if out else "      "


# ── Response parsing ──────────────────────────────────────────────────────────


def parse_summary(parsed: ClusterSummaryPayload) -> ClusterSummaryPayload:
    """Type-assert and return the schema-validated payload.

    The pydantic schema already validated shape and bounds at
    :meth:`LLMClient.chat_completion` time, so this helper is a thin
    type-guard rather than a real parser. Kept as a named function for
    symmetry with :func:`pipeline.relevance.parse_relevance_verdicts`
    and to give step 10b's orchestration a single point of post-
    validation enforcement if a future invariant (e.g. ``source_urls``
    must be a subset of the cluster's member URLs) needs adding.

    Raises:
        TypeError: ``parsed`` is not a :class:`ClusterSummaryPayload`.
    """
    if not isinstance(parsed, ClusterSummaryPayload):
        raise TypeError(f"expected ClusterSummaryPayload, got {type(parsed).__name__}")
    return parsed


__all__ = (
    "PROMPT_VERSION",
    "SummaryCluster",
    "SummaryMember",
    "build_summary_messages",
    "parse_summary",
)
