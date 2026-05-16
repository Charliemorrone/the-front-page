"""Tests for the LLM-stage pydantic schemas.

The schemas defend the pipeline against malformed LLM output: each one
declares ``extra="forbid"`` so hallucinated keys fail validation, and
numeric fields are bounded so out-of-range scores can't slip through
into the database.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from clawfeed_intel.llm import (
    ClusterSummaryPayload,
    RelevanceBatchResponse,
    RelevanceVerdict,
    SearchPlan,
)


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


# ── ClusterSummaryPayload ─────────────────────────────────────────────────────


def _valid_summary_payload() -> dict[str, object]:
    return {
        "headline": "Anthropic closes $500M Series E",
        "summary": (
            "Anthropic announced a $500M Series E led by GeneralCo. "
            "Existing investors participated. Proceeds fund model training."
        ),
        "why_it_matters": "Largest AI-lab financing of the week.",
        "entities": ["Anthropic", "GeneralCo"],
        "key_facts": ["$500M raise", "Series E", "Led by GeneralCo"],
        "caveats": ["Valuation not disclosed."],
        "source_urls": [
            "https://techcrunch.com/anthropic-series-e",
            "https://www.sec.gov/Archives/edgar/data/.../primary_doc.xml",
        ],
        "confidence": 0.85,
    }


def test_summary_accepts_full_payload() -> None:
    payload = ClusterSummaryPayload.model_validate(_valid_summary_payload())
    assert payload.headline.startswith("Anthropic")
    assert payload.confidence == 0.85
    assert payload.entities == ["Anthropic", "GeneralCo"]
    assert len(payload.source_urls) == 2


def test_summary_accepts_minimum_fields() -> None:
    """Only ``headline`` and ``summary`` are required; everything else defaults."""
    payload = ClusterSummaryPayload.model_validate(
        {
            "headline": "Headline only",
            "summary": "One factual sentence.",
        }
    )
    assert payload.why_it_matters == ""
    assert payload.entities == []
    assert payload.key_facts == []
    assert payload.caveats == []
    assert payload.source_urls == []
    assert payload.confidence is None


def test_summary_accepts_null_narrative_fields() -> None:
    """The 9c-lesson regression guard for step 10.

    Local 27B models routinely emit ``null`` for narrative fields under
    load (no entities to surface for a sparse cluster, no caveats, no
    confidence estimate). Tightening the schema would force the
    repair retry on most clusters and still likely fail. Only
    ``headline`` and ``summary`` are load-bearing; everything else
    must accept reasonable absences.
    """
    payload = ClusterSummaryPayload.model_validate(
        {
            "headline": "Headline",
            "summary": "Summary.",
            "why_it_matters": "",
            "entities": [],
            "key_facts": [],
            "caveats": [],
            "source_urls": [],
            "confidence": None,
        }
    )
    assert payload.confidence is None
    assert payload.entities == []


def test_summary_confidence_lower_bound() -> None:
    payload = _valid_summary_payload()
    payload["confidence"] = -0.1
    with pytest.raises(ValidationError):
        ClusterSummaryPayload.model_validate(payload)


def test_summary_confidence_upper_bound() -> None:
    payload = _valid_summary_payload()
    payload["confidence"] = 1.5
    with pytest.raises(ValidationError):
        ClusterSummaryPayload.model_validate(payload)


def test_summary_requires_headline() -> None:
    payload = _valid_summary_payload()
    del payload["headline"]
    with pytest.raises(ValidationError):
        ClusterSummaryPayload.model_validate(payload)


def test_summary_requires_summary() -> None:
    payload = _valid_summary_payload()
    del payload["summary"]
    with pytest.raises(ValidationError):
        ClusterSummaryPayload.model_validate(payload)


def test_summary_rejects_blank_headline() -> None:
    """``min_length=1`` combined with ``str_strip_whitespace=True``: a
    whitespace-only headline fails validation. The brief can't render a
    cluster without a headline, so this is enforced at the schema
    boundary rather than relying on downstream presentation logic.
    """
    payload = _valid_summary_payload()
    payload["headline"] = "   "
    with pytest.raises(ValidationError):
        ClusterSummaryPayload.model_validate(payload)


def test_summary_rejects_blank_summary() -> None:
    payload = _valid_summary_payload()
    payload["summary"] = "\n\t"
    with pytest.raises(ValidationError):
        ClusterSummaryPayload.model_validate(payload)


def test_summary_rejects_extra_fields() -> None:
    payload = _valid_summary_payload()
    payload["takeaways"] = ["hallucinated key"]
    with pytest.raises(ValidationError):
        ClusterSummaryPayload.model_validate(payload)


def test_summary_strips_string_whitespace() -> None:
    payload = _valid_summary_payload()
    payload["headline"] = "  Anthropic closes $500M Series E  "
    parsed = ClusterSummaryPayload.model_validate(payload)
    assert parsed.headline == "Anthropic closes $500M Series E"


def test_summary_round_trip_via_json() -> None:
    """The LLM client validates via ``model_validate_json`` — exercise that path."""
    raw = (
        '{"headline": "Quiet release", "summary": "A small repo gained '
        'attention.", "entities": ["acme/awesome"], "confidence": null}'
    )
    parsed = ClusterSummaryPayload.model_validate_json(raw)
    assert parsed.headline == "Quiet release"
    assert parsed.confidence is None
    assert parsed.entities == ["acme/awesome"]


# ── SearchPlan (Phase 7b) ─────────────────────────────────────────────────────


def _valid_search_plan_payload() -> dict[str, object]:
    return {
        "selected_source_kinds": ["sec_edgar", "gdelt", "github_search"],
        "query_variants": [
            "Khosla Ventures",
            "Vinod Khosla",
            "Khosla led round",
            "Khosla Ventures Form D",
        ],
        "required_terms": ["khosla"],
        "excluded_terms": [],
        "expected_evidence_types": ["funding_round", "regulatory_filing"],
        "rationale": "Investor-focused query; prioritize filings + news + repo signals.",
    }


def test_search_plan_accepts_full_payload() -> None:
    plan = SearchPlan.model_validate(_valid_search_plan_payload())
    assert plan.selected_source_kinds == ["sec_edgar", "gdelt", "github_search"]
    assert plan.query_variants[0] == "Khosla Ventures"
    assert plan.required_terms == ["khosla"]
    assert plan.expected_evidence_types == ["funding_round", "regulatory_filing"]


def test_search_plan_accepts_empty_payload() -> None:
    """Permissive per the 9c lesson — an empty plan is a meaningful
    signal ("planner found no usable sources"), not a schema error.
    Downstream the orchestrator surfaces it via coverage and produces
    a 0-item brief, same shape as "fetchers all failed".
    """
    plan = SearchPlan.model_validate({})
    assert plan.selected_source_kinds == []
    assert plan.query_variants == []
    assert plan.required_terms == []
    assert plan.excluded_terms == []
    assert plan.expected_evidence_types == []
    assert plan.rationale == ""


def test_search_plan_preserves_source_kind_order() -> None:
    """``selected_source_kinds`` order IS the priority order — load-
    bearing for the orchestrator's dispatch loop."""
    payload = _valid_search_plan_payload()
    payload["selected_source_kinds"] = ["github_search", "raw_cache", "hn_algolia"]
    plan = SearchPlan.model_validate(payload)
    assert plan.selected_source_kinds == ["github_search", "raw_cache", "hn_algolia"]


def test_search_plan_rejects_extra_fields() -> None:
    payload = _valid_search_plan_payload()
    payload["priority_weights"] = {"sec_edgar": 1.0}  # hallucinated field
    with pytest.raises(ValidationError):
        SearchPlan.model_validate(payload)


def test_search_plan_strips_string_whitespace() -> None:
    payload = _valid_search_plan_payload()
    payload["rationale"] = "  Trim me.  "
    plan = SearchPlan.model_validate(payload)
    assert plan.rationale == "Trim me."


def test_search_plan_round_trip_via_json() -> None:
    """The LLM client validates via ``model_validate_json`` — exercise that path."""
    raw = (
        '{"selected_source_kinds": ["gdelt", "reddit"], '
        '"query_variants": ["Anthropic", "Claude AI"], '
        '"required_terms": [], '
        '"excluded_terms": ["cricket"], '
        '"expected_evidence_types": ["news_article"], '
        '"rationale": "AI company query; exclude unrelated cricket namesake."}'
    )
    plan = SearchPlan.model_validate_json(raw)
    assert plan.selected_source_kinds == ["gdelt", "reddit"]
    assert plan.excluded_terms == ["cricket"]


def test_search_plan_does_not_constrain_source_kind_values() -> None:
    """Schema deliberately does NOT use ``Literal[...]`` — caller
    sanity-checks against the actual fetcher inventory. This test
    pins that decision: a "future" or "unknown" kind passes schema
    validation. Catching it is the dispatcher's job, not the schema's.
    """
    payload = _valid_search_plan_payload()
    payload["selected_source_kinds"] = ["some_future_fetcher_kind"]
    plan = SearchPlan.model_validate(payload)
    assert plan.selected_source_kinds == ["some_future_fetcher_kind"]
