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


__all__ = (
    "RelevanceBatchResponse",
    "RelevanceVerdict",
)
