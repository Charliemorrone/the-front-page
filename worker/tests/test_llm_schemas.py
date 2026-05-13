"""Tests for the LLM-stage pydantic schemas.

The schemas defend the pipeline against malformed LLM output: each one
declares ``extra="forbid"`` so hallucinated keys fail validation, and
numeric fields are bounded so out-of-range scores can't slip through
into the database.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from clawfeed_intel.llm import RelevanceBatchResponse, RelevanceVerdict


# ── RelevanceVerdict ──────────────────────────────────────────────────────────


def _valid_verdict_payload() -> dict[str, object]:
    return {
        "keep": True,
        "category": "startup_funding",
        "score": 0.82,
        "event_type": "funding_round",
        "reason": "Series B closed; substantive financing event.",
        "entities": ["Anthropic"],
        "evidence_urls": ["https://techcrunch.com/anthropic-series-b"],
        "uncertainty": 0.1,
    }


def test_verdict_accepts_full_payload() -> None:
    verdict = RelevanceVerdict.model_validate(_valid_verdict_payload())
    assert verdict.keep is True
    assert verdict.category == "startup_funding"
    assert verdict.score == 0.82
    assert verdict.event_type == "funding_round"
    assert verdict.entities == ["Anthropic"]
    assert verdict.uncertainty == 0.1


def test_verdict_accepts_minimum_fields() -> None:
    """Optional fields default cleanly so a terse LLM reply still validates."""
    verdict = RelevanceVerdict.model_validate(
        {
            "keep": False,
            "score": 0.1,
        }
    )
    assert verdict.category is None
    assert verdict.event_type is None
    assert verdict.reason is None
    assert verdict.entities == []
    assert verdict.evidence_urls == []
    assert verdict.uncertainty is None


def test_verdict_accepts_null_category_and_reason() -> None:
    """Local models reliably emit ``null`` for category and reason on
    rejected verdicts. Caught live during the first end-to-end smoke
    against Qwen3.5-27B-4bit (run 1 of the 2026-05-12 smoke). Strict
    requirement would force the repair retry on every batch with any
    rejection and still likely fail. The schema accepts ``None``; the
    DB columns are nullable.
    """
    verdict = RelevanceVerdict.model_validate(
        {
            "keep": False,
            "category": None,
            "score": 0.05,
            "event_type": None,
            "reason": None,
        }
    )
    assert verdict.keep is False
    assert verdict.category is None
    assert verdict.reason is None


def test_verdict_score_lower_bound() -> None:
    payload = _valid_verdict_payload()
    payload["score"] = -0.1
    with pytest.raises(ValidationError):
        RelevanceVerdict.model_validate(payload)


def test_verdict_score_upper_bound() -> None:
    payload = _valid_verdict_payload()
    payload["score"] = 1.5
    with pytest.raises(ValidationError):
        RelevanceVerdict.model_validate(payload)


def test_verdict_uncertainty_bounds() -> None:
    payload = _valid_verdict_payload()
    payload["uncertainty"] = 1.5
    with pytest.raises(ValidationError):
        RelevanceVerdict.model_validate(payload)


def test_verdict_rejects_extra_fields() -> None:
    """``extra='forbid'`` so hallucinated keys are a schema-validation failure."""
    payload = _valid_verdict_payload()
    payload["confidence_level"] = "high"
    with pytest.raises(ValidationError):
        RelevanceVerdict.model_validate(payload)


def test_verdict_requires_keep() -> None:
    payload = _valid_verdict_payload()
    del payload["keep"]
    with pytest.raises(ValidationError):
        RelevanceVerdict.model_validate(payload)


def test_verdict_requires_score() -> None:
    """``score`` stays required — it's how downstream stages rank borderline
    verdicts. Unlike ``reason``, the model reliably emits it.
    """
    payload = _valid_verdict_payload()
    del payload["score"]
    with pytest.raises(ValidationError):
        RelevanceVerdict.model_validate(payload)


def test_verdict_strips_string_whitespace() -> None:
    """``str_strip_whitespace`` defends against trailing newlines from the LLM."""
    payload = _valid_verdict_payload()
    payload["category"] = "  startup_funding  "
    verdict = RelevanceVerdict.model_validate(payload)
    assert verdict.category == "startup_funding"


# ── RelevanceBatchResponse ────────────────────────────────────────────────────


def test_batch_response_round_trip_via_json() -> None:
    """The LLM client validates via ``model_validate_json`` — exercise that path."""
    raw = (
        '{"verdicts": ['
        '{"keep": true, "category": "ai_research", "score": 0.9, '
        '"reason": "Strong new scaling result."}, '
        '{"keep": false, "category": "ai_research", "score": 0.2, '
        '"reason": "Marginal benchmark delta."}'
        "]}"
    )
    parsed = RelevanceBatchResponse.model_validate_json(raw)
    assert len(parsed.verdicts) == 2
    assert parsed.verdicts[0].keep is True
    assert parsed.verdicts[1].keep is False


def test_batch_response_accepts_empty_verdicts() -> None:
    """An empty array is legal at the schema layer; the count-check in
    :func:`pipeline.relevance.parse_relevance_verdicts` is what surfaces a
    count mismatch when the batch wasn't empty.
    """
    parsed = RelevanceBatchResponse.model_validate({"verdicts": []})
    assert parsed.verdicts == []


def test_batch_response_requires_verdicts_key() -> None:
    with pytest.raises(ValidationError):
        RelevanceBatchResponse.model_validate({"results": []})


def test_batch_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RelevanceBatchResponse.model_validate({"verdicts": [], "batch_confidence": 0.5})


def test_batch_response_propagates_verdict_validation() -> None:
    """A bad verdict inside a valid wrapper still fails — single error type."""
    with pytest.raises(ValidationError):
        RelevanceBatchResponse.model_validate(
            {
                "verdicts": [
                    {
                        "keep": True,
                        "category": "ai_research",
                        "score": 2.0,  # out of range
                        "reason": "x",
                    }
                ]
            }
        )
