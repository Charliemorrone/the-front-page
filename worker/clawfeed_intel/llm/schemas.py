"""Pydantic schemas for LLM-stage structured outputs.

Centralized so the LLM client (:mod:`llm.client`) and the pipeline
stages (:mod:`pipeline.relevance` and onward) import the same
definition. Keeping the contracts here makes them easy to evolve
together when a prompt iteration adds a new field.

Validation posture mirrors :mod:`llm.routing`: ``extra="forbid"``
everywhere so a hallucinated extra field surfaces as a schema-validation
error, fires the bounded repair retry, and only then bubbles up as
:class:`LLMSchemaError`. Numeric fields are bounded with ``ge=0, le=1``
so the schema rejects out-of-range scores without the pipeline having
to second-guess the model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _SchemaBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RelevanceVerdict(_SchemaBase):
    """One cluster's keep/reject decision from the relevance filter.

    Field shape from the architecture doc's "Prompt Responsibilities →
    Relevance Filter" section. The LLM emits one of these per cluster
    in the batch, in the same order it received them — verdict
    assignment is positional, so order preservation is load-bearing.

    ``category`` and ``reason`` are intentionally permissive
    (``str | None``): local models reliably emit ``null`` for these
    fields on rejected verdicts (no category to assign, no narrative
    needed) — discovered live during the first end-to-end smoke run
    against Qwen3.5-27B-4bit. Strict requirement would force the
    schema-repair retry on every batch with any rejection and still
    likely fail; we'd rather accept the verdict and surface the
    judgement than reject the whole batch.
    """

    keep: bool
    category: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    event_type: str | None = None
    reason: str | None = None
    entities: list[str] = Field(default_factory=list)
    evidence_urls: list[str] = Field(default_factory=list)
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)


class RelevanceBatchResponse(_SchemaBase):
    """Wrapper around the per-batch verdict list.

    Wrapper-shaped (not a bare top-level array) for two reasons:
    local models reliably begin object responses with ``{`` when
    instructed, while bare-array prompts invite markdown fences and
    trailing commentary the JSON-repair retry then has to clean up; and
    a wrapper leaves room for a future batch-level field (e.g. a
    batch-confidence summary) without bumping the contract.
    """

    verdicts: list[RelevanceVerdict]


class SearchPlan(_SchemaBase):
    """Planner-recommended search strategy for one topic query (Phase 7b).

    Field shape from the architecture doc's "Prompt Responsibilities →
    Source Planner" section. The planner takes a query + the available
    source kinds + the time window and returns this plan; the topic
    orchestration layer (7e) dispatches fetchers per the plan.

    All fields are permissive per the 9c lesson: local models reliably
    emit empty arrays / ``null`` for narrative additions under load,
    and an empty plan is itself a meaningful signal (downstream
    coverage stays honest — "planner found no usable sources" produces
    a 0-item brief, same shape as "fetchers all failed"). Strict
    ``min_length=1`` here would force the bounded-repair retry on
    every borderline case and still likely fail.

    ``selected_source_kinds`` order IS the priority order — the first
    kind is the planner's highest-priority dispatch target. This
    matches the architecture-doc "Priority order" field without
    bloating the schema with per-source priority integers.

    Source-kind values are NOT constrained at the schema layer (no
    ``Literal[...]``) — the caller sanity-checks against the actual
    fetcher inventory at dispatch time. Constraining here would force a
    schema bump every time a fetcher is added or renamed, which is the
    wrong coupling.
    """

    selected_source_kinds: list[str] = Field(default_factory=list)
    query_variants: list[str] = Field(default_factory=list)
    required_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)
    expected_evidence_types: list[str] = Field(default_factory=list)
    rationale: str = ""


class ClusterSummaryPayload(_SchemaBase):
    """One cluster's grounded summary from the cluster-summary stage.

    Field shape from the architecture doc's "Prompt Responsibilities →
    Cluster Summary" section. Only ``headline`` and ``summary`` are
    required: these are the load-bearing signals the final composer
    needs to render the cluster in the brief at all. Everything else
    (``why_it_matters``, ``entities``, ``key_facts``, ``caveats``,
    ``source_urls``, ``confidence``) is permissive — local models
    routinely emit ``null`` or empty arrays for narrative additions
    under load, and tightening the schema would force the bounded
    repair retry on every cluster and still likely fail. The 9c lesson
    applied at the start: the architecture-doc field list specifies
    *what* the schema covers, not *which fields are required*. Tighten
    later if the architecture-doc-target flagship (``Qwen3.5-122B-
    A10B-4bit``) reliably emits richer payloads.
    """

    headline: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    why_it_matters: str = ""
    entities: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


__all__ = (
    "ClusterSummaryPayload",
    "RelevanceBatchResponse",
    "RelevanceVerdict",
    "SearchPlan",
)
