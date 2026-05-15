"""Tests for the cluster-summary pure layer and orchestration.

Step 10a covers schemas + pure helpers + DB write. Step 10b adds the
async orchestration tests against a real :class:`LLMClient` backed by
:class:`httpx.MockTransport` — no live vMLX in CI.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import Any

import httpx
import pytest

from clawfeed_intel import db as worker_db
from clawfeed_intel.llm import LLMClient, RetryConfig, RoutingConfig
from clawfeed_intel.llm.schemas import ClusterSummaryPayload
from clawfeed_intel.pipeline.summary import (
    PROMPT_VERSION,
    SummaryCluster,
    SummaryMember,
    build_summary_messages,
    parse_summary,
    summarize_clusters,
)
from clawfeed_intel.runs import Coverage
from clawfeed_intel.sources import CategoryPlan, ProfileConfig, SourcePlan


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _category(
    name: str = "startup_funding",
    *,
    description: str = "Funding rounds and acquisitions.",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> CategoryPlan:
    return CategoryPlan(
        name=name,
        description=description,
        include_rules=include or ["announced funding rounds"],
        exclude_rules=exclude or ["recirculated old rounds"],
    )


def _member(
    *,
    title: str = "Anthropic raises Series E",
    canonical_url: str = "https://techcrunch.com/anthropic-series-e",
    excerpt: str = "Anthropic announced a Series E.",
    content: str = "",
    author: str = "",
    published_at: str | None = None,
) -> SummaryMember:
    return SummaryMember(
        title=title,
        canonical_url=canonical_url,
        excerpt=excerpt,
        content=content,
        author=author,
        published_at=published_at,
    )


def _cluster(
    *,
    cluster_id: int = 1,
    title: str = "Anthropic Series E",
    category: str | None = "startup_funding",
    members: tuple[SummaryMember, ...] = (),
) -> SummaryCluster:
    return SummaryCluster(
        cluster_id=cluster_id,
        title=title,
        category=category,
        members=members or (_member(),),
    )


# ── PROMPT_VERSION ────────────────────────────────────────────────────────────


def test_prompt_version_is_pinned() -> None:
    """The prompt-version slug appears in every ``item_summaries`` row.

    Bump on behavioral changes so audit rows reflect which generation
    of the prompt produced them.
    """
    assert PROMPT_VERSION == "summary.v1"


# ── Message shape ─────────────────────────────────────────────────────────────


def test_build_messages_returns_system_then_user() -> None:
    messages = build_summary_messages(_cluster(), _category())
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_messages_raises_on_empty_members() -> None:
    """A memberless cluster has nothing to summarize.

    Defensive: ``iter_kept_clusters_with_members`` (10b) will filter
    these out in SQL, but the pure helper should fail loudly rather
    than dispatch a wasted LLM call.
    """
    with pytest.raises(ValueError, match="no members"):
        build_summary_messages(
            SummaryCluster(cluster_id=42, title="empty", category="x", members=()),
            _category(),
        )


# ── System message ────────────────────────────────────────────────────────────


def test_system_message_states_grounding_rule() -> None:
    """The grounding rule is load-bearing — the prompt's job is to
    prevent the model from inventing facts the final composer would
    then propagate into the brief.
    """
    messages = build_summary_messages(_cluster(), _category())
    system = messages[0]["content"]
    assert "Grounding policy" in system
    assert "MUST be present in the supplied source items" in system
    assert "Do not invent" in system


def test_system_message_states_citation_rule() -> None:
    """``source_urls`` must come from the input — the brief's links
    section depends on this rule holding.
    """
    messages = build_summary_messages(_cluster(), _category())
    system = messages[0]["content"]
    assert "source_urls" in system
    assert "list the cluster's source URLs" in system


def test_system_message_describes_response_shape() -> None:
    messages = build_summary_messages(_cluster(), _category())
    system = messages[0]["content"]
    for field in (
        "headline",
        "summary",
        "why_it_matters",
        "entities",
        "key_facts",
        "caveats",
        "source_urls",
        "confidence",
    ):
        assert field in system, f"system message missing field {field!r}"


def test_system_message_demands_pure_json() -> None:
    """JSON-only / no-markdown-fencing wording mirrors the relevance
    filter's hardening; without it, vMLX returns markdown-wrapped
    responses under load and the bounded-repair retry fires.
    """
    messages = build_summary_messages(_cluster(), _category())
    system = messages[0]["content"]
    assert "Reply with valid JSON only" in system
    assert "no markdown fencing" in system
    assert "Begin your response with `{`" in system


def test_system_message_includes_category_context() -> None:
    messages = build_summary_messages(
        _cluster(category="startup_funding"),
        _category(
            "startup_funding",
            description="Funding rounds, new fund closes, notable acquisitions.",
            include=["announced rounds", "Form D filings"],
            exclude=["opinion posts"],
        ),
    )
    system = messages[0]["content"]
    assert "Category context:" in system
    assert "name: startup_funding" in system
    assert "Funding rounds, new fund closes" in system
    assert "announced rounds" in system
    assert "Form D filings" in system
    assert "opinion posts" in system


def test_system_message_omits_category_when_none() -> None:
    """A cluster with no category falls back to a generic framing —
    the relevance filter's permissive schema allows ``category=None``
    and we mirror that posture here. Verified end-to-end against the
    9c relaxation.
    """
    messages = build_summary_messages(
        _cluster(category=None),
        None,
    )
    system = messages[0]["content"]
    assert "Category context:" not in system
    # The grounding policy still applies regardless of category.
    assert "Grounding policy" in system


# ── User message ──────────────────────────────────────────────────────────────


def test_user_message_starts_with_cluster_header() -> None:
    messages = build_summary_messages(_cluster(title="Anthropic Series E"), _category())
    user = messages[1]["content"]
    assert user.startswith("Cluster: Anthropic Series E")


def test_user_message_falls_back_to_untitled() -> None:
    cluster = _cluster(title="", members=(_member(title="m"),))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "Cluster: (untitled)" in user


def test_user_message_states_category_when_present() -> None:
    cluster = _cluster(category="ai_research")
    messages = build_summary_messages(cluster, _category("ai_research"))
    user = messages[1]["content"]
    assert "Filed under category: ai_research" in user


def test_user_message_lists_all_members_in_order() -> None:
    members = (
        _member(title="A", canonical_url="https://a.example/post"),
        _member(title="B", canonical_url="https://b.example/post"),
        _member(title="C", canonical_url="https://c.example/post"),
    )
    cluster = _cluster(members=members)
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "[1] A" in user
    assert "[2] B" in user
    assert "[3] C" in user
    assert user.index("[1] A") < user.index("[2] B") < user.index("[3] C")


def test_user_message_surfaces_member_url_and_excerpt() -> None:
    member = _member(
        title="Article",
        canonical_url="https://example.com/post",
        excerpt="The lead paragraph mentions the round size.",
    )
    cluster = _cluster(members=(member,))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "url: https://example.com/post" in user
    assert "The lead paragraph mentions the round size." in user


def test_user_message_prefers_content_over_excerpt() -> None:
    """``content`` is the trafilatura/full-body field; ``excerpt`` is
    the fetcher's ~320-char prefix. When both are present the model
    should see the richer body.
    """
    member = _member(
        title="Article",
        canonical_url="https://example.com/post",
        excerpt="Lead paragraph excerpt.",
        content="Full extracted body with multiple sentences. Second sentence.",
    )
    cluster = _cluster(members=(member,))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "Full extracted body" in user
    assert "Lead paragraph excerpt." not in user


def test_user_message_omits_body_when_both_empty() -> None:
    """GDELT ArtList carries no body and only a title-prefix excerpt;
    the prompt should still render cleanly. Verified against the
    fetcher-contract docstring.
    """
    member = _member(
        title="Headline only",
        canonical_url="https://news.example/article",
        excerpt="",
        content="",
    )
    cluster = _cluster(members=(member,))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "Headline only" in user
    assert "body:" not in user


def test_user_message_surfaces_author_and_published_at_when_set() -> None:
    member = _member(
        title="Paper",
        canonical_url="https://arxiv.org/abs/2405.12345",
        author="A. Researcher, B. Coauthor",
        published_at="2026-05-13T00:00:00+00:00",
    )
    cluster = _cluster(members=(member,))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "author: A. Researcher, B. Coauthor" in user
    assert "published_at: 2026-05-13T00:00:00+00:00" in user


def test_user_message_omits_author_and_published_at_when_empty() -> None:
    cluster = _cluster(
        members=(_member(author="", published_at=None),),
    )
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "author:" not in user
    assert "published_at:" not in user


def test_user_message_states_member_count() -> None:
    """L2/L3-folded clusters surface every member; the count line
    nudges the model to cite cross-source corroboration in the
    summary.
    """
    members = (
        _member(title="Coverage A", canonical_url="https://a.example/"),
        _member(title="Coverage B", canonical_url="https://b.example/"),
        _member(title="Coverage C", canonical_url="https://c.example/"),
    )
    cluster = _cluster(members=members)
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "Source items (3):" in user


def test_user_message_indents_multiline_body() -> None:
    """Body lines nest under ``body:`` so the prompt stays parseable
    if a future debugger eyeballs it.
    """
    member = _member(
        content="First line.\nSecond line.\n  whitespace-only line.\n",
    )
    cluster = _cluster(members=(member,))
    messages = build_summary_messages(cluster, _category())
    user = messages[1]["content"]
    assert "      First line." in user
    assert "      Second line." in user


# ── Parse layer ───────────────────────────────────────────────────────────────


def test_parse_summary_returns_payload() -> None:
    payload = ClusterSummaryPayload(headline="Headline", summary="Summary.")
    assert parse_summary(payload) is payload


def test_parse_summary_rejects_wrong_type() -> None:
    class NotAPayload:
        headline = "h"
        summary = "s"

    with pytest.raises(TypeError, match="ClusterSummaryPayload"):
        parse_summary(NotAPayload())  # type: ignore[arg-type]


# ── Orchestration fixtures (step 10b) ─────────────────────────────────────────


@pytest.fixture
def routing() -> RoutingConfig:
    """Minimal routing config sized for cluster-summary tests."""
    return RoutingConfig.model_validate(
        {
            "providers": {
                "vmlx": {"base_url": "http://127.0.0.1:8080/v1"},
            },
            "stages": {
                "cluster_summary": {
                    "provider": "vmlx",
                    "model": "stub-summary-model",
                    "timeout_seconds": 30,
                },
            },
        }
    )


def _summary_payload_content(**overrides: object) -> str:
    base: dict[str, object] = {
        "headline": "Stub headline",
        "summary": "Stub summary sentence.",
        "why_it_matters": "",
        "entities": [],
        "key_facts": [],
        "caveats": [],
        "source_urls": [],
        "confidence": None,
    }
    base.update(overrides)
    return json.dumps(base)


def _chat_response(*, content: str, model: str = "stub-summary-model") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_client(
    routing: RoutingConfig,
    handler: Any,
    *,
    conn: sqlite3.Connection | None = None,
    run_id: int | None = None,
) -> LLMClient:
    return LLMClient(
        routing,
        transport=httpx.MockTransport(handler),
        conn=conn,
        run_id=run_id,
        retry_config=RetryConfig(max_attempts=1, wait_min_seconds=0, wait_max_seconds=0),
    )


def _make_plan(categories: list[CategoryPlan] | None = None) -> SourcePlan:
    return SourcePlan(
        profile=ProfileConfig(),
        categories=categories or [],
        dynamic_search=[],
        warnings=[],
    )


def _seed_run(conn: sqlite3.Connection) -> int:
    return worker_db.create_run(
        conn,
        run_type="daily",
        window_start="2026-05-13T00:00:00+00:00",
        window_end="2026-05-14T00:00:00+00:00",
    )


def _seed_kept_cluster(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    key: str,
    title: str = "",
    category: str = "ai_research",
    content: str = "Body for the LLM to summarize.",
    excerpt: str = "Short excerpt.",
) -> int:
    rep_id, _ = worker_db.upsert_raw_item(
        conn,
        run_id=run_id,
        source_type="rss",
        dedup_key=f"{key}-rep",
        title=title or key,
        url=key,
        canonical_url=key,
        excerpt=excerpt,
        content=content,
    )
    cluster_id, _ = worker_db.create_cluster(
        conn,
        run_id=run_id,
        cluster_key=key,
        title=title or key,
        raw_item_ids=[rep_id],
    )
    worker_db.update_cluster_verdict(
        conn,
        cluster_id=cluster_id,
        status="kept",
        relevance_score=0.8,
        category=category,
        event_type=None,
        filter_reason="Substantive event.",
    )
    return cluster_id


# ── db.iter_kept_clusters_with_members ────────────────────────────────────────


def test_iter_kept_only_yields_kept_clusters(temp_db) -> None:
    """Pending and filtered-out clusters must be skipped."""
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        kept_id = _seed_kept_cluster(conn, run_id, key="https://a.example/")

        # A second cluster left at pending (no verdict).
        pending_rep, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="rss",
            dedup_key="pending-rep",
            title="Pending",
            url="https://b.example/",
            canonical_url="https://b.example/",
            content="",
        )
        worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://b.example/",
            title="Pending",
            raw_item_ids=[pending_rep],
        )

        # A third cluster explicitly rejected.
        rejected_rep, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="rss",
            dedup_key="rej-rep",
            title="Rejected",
            url="https://c.example/",
            canonical_url="https://c.example/",
            content="",
        )
        rejected_id, _ = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://c.example/",
            title="Rejected",
            raw_item_ids=[rejected_rep],
        )
        worker_db.update_cluster_verdict(
            conn,
            cluster_id=rejected_id,
            status="filtered_out",
            relevance_score=0.1,
            category=None,
            event_type=None,
            filter_reason=None,
        )

        yielded = list(worker_db.iter_kept_clusters_with_members(conn, run_id))
        assert len(yielded) == 1
        cluster_id, _title, category, _members = yielded[0]
        assert cluster_id == kept_id
        assert category == "ai_research"


def test_iter_kept_groups_members_in_id_order(temp_db) -> None:
    """The smallest-id member of each cluster comes first in the
    yielded list — mirrors the relevance loader's representative rule.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        rep_id, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="rss",
            dedup_key="rep",
            title="Rep",
            url="https://a.example/",
            canonical_url="https://a.example/",
            content="rep body",
        )
        extra_id, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="rss",
            dedup_key="extra",
            title="Extra",
            url="https://b.example/",
            canonical_url="https://b.example/",
            content="extra body",
        )
        cluster_id, _ = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://a.example/",
            title="Combined",
            raw_item_ids=[extra_id, rep_id],  # passed out of order on purpose
        )
        worker_db.update_cluster_verdict(
            conn,
            cluster_id=cluster_id,
            status="kept",
            relevance_score=0.7,
            category="ai_research",
            event_type=None,
            filter_reason="r",
        )

        yielded = list(worker_db.iter_kept_clusters_with_members(conn, run_id))
        assert len(yielded) == 1
        _id, _title, _category, members = yielded[0]
        assert [m["canonical_url"] for m in members] == [
            "https://a.example/",  # smaller id first
            "https://b.example/",
        ]


# ── db.advance_cluster_to_summarized ──────────────────────────────────────────


def test_advance_cluster_to_summarized_happy_path(temp_db) -> None:
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        cluster_id = _seed_kept_cluster(conn, run_id, key="https://a.example/")

        worker_db.advance_cluster_to_summarized(conn, cluster_id)

        status = conn.execute(
            "SELECT status FROM item_clusters WHERE id = ?", (cluster_id,)
        ).fetchone()["status"]
        assert status == "summarized"


def test_advance_refuses_pending_cluster(temp_db) -> None:
    """A pending cluster (no verdict yet) must not skip the filter stage —
    the SQL precondition ``status='kept'`` makes this impossible.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        rep_id, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="rss",
            dedup_key="rep",
            title="Pending",
            url="https://a.example/",
            canonical_url="https://a.example/",
            content="",
        )
        cluster_id, _ = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key="https://a.example/",
            title="Pending",
            raw_item_ids=[rep_id],
        )
        with pytest.raises(LookupError):
            worker_db.advance_cluster_to_summarized(conn, cluster_id)


def test_advance_is_idempotent_against_already_summarized(temp_db) -> None:
    """Re-applying the advance after the row is already summarized
    raises LookupError — the orchestration's replay-safe loader makes
    this a defensive surface, not an expected path.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        cluster_id = _seed_kept_cluster(conn, run_id, key="https://a.example/")
        worker_db.advance_cluster_to_summarized(conn, cluster_id)
        with pytest.raises(LookupError):
            worker_db.advance_cluster_to_summarized(conn, cluster_id)


# ── summarize_clusters — orchestration ────────────────────────────────────────


async def test_summarize_clusters_no_kept_returns_zero(temp_db, routing) -> None:
    """No kept clusters → return 0 and never call the LLM."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        coverage = Coverage()
        summarized = await summarize_clusters(
            conn,
            run_id,
            client,
            coverage,
            plan=_make_plan(),
            model="stub-summary-model",
        )
        assert summarized == 0
        assert calls == []
        assert coverage.failed_summary_clusters == 0


async def test_summarize_clusters_happy_path_writes_row_and_advances_status(
    temp_db, routing
) -> None:
    """One kept cluster → one ``item_summaries`` row, status='summarized'."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_response(
                content=_summary_payload_content(
                    headline="Concrete headline",
                    summary="Concrete summary sentence.",
                    entities=["Anthropic"],
                    source_urls=["https://a.example/"],
                    confidence=0.7,
                )
            ),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        cluster_id = _seed_kept_cluster(conn, run_id, key="https://a.example/")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        coverage = Coverage()

        summarized = await summarize_clusters(
            conn,
            run_id,
            client,
            coverage,
            plan=_make_plan(),
            model="stub-summary-model",
        )

        assert summarized == 1
        assert coverage.failed_summary_clusters == 0

        row = conn.execute(
            "SELECT * FROM item_summaries WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()
        assert row["headline"] == "Concrete headline"
        assert row["model"] == "stub-summary-model"
        assert row["prompt_version"] == "summary.v1"
        assert row["confidence"] == 0.7
        assert json.loads(row["entities"]) == ["Anthropic"]
        assert json.loads(row["source_urls"]) == ["https://a.example/"]

        status = conn.execute(
            "SELECT status FROM item_clusters WHERE id = ?", (cluster_id,)
        ).fetchone()["status"]
        assert status == "summarized"


async def test_summarize_clusters_one_call_per_cluster(temp_db, routing) -> None:
    """Three kept clusters → three HTTP calls (not batched)."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_chat_response(content=_summary_payload_content()))

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        _seed_kept_cluster(conn, run_id, key="https://a.example/")
        _seed_kept_cluster(conn, run_id, key="https://b.example/")
        _seed_kept_cluster(conn, run_id, key="https://c.example/")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        coverage = Coverage()

        summarized = await summarize_clusters(
            conn,
            run_id,
            client,
            coverage,
            plan=_make_plan(),
            model="stub-summary-model",
        )

        assert summarized == 3
        assert len(captured) == 3


async def test_summarize_clusters_uses_call_site_sampling(temp_db, routing) -> None:
    """temperature=0.1 and max_tokens=2048 are pinned at the call site."""
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_response(content=_summary_payload_content()))

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        _seed_kept_cluster(conn, run_id, key="https://a.example/")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        await summarize_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            plan=_make_plan(),
            model="stub-summary-model",
        )

    assert captured[0]["temperature"] == 0.1
    assert captured[0]["max_tokens"] == 2048


async def test_summarize_clusters_surfaces_category_context(temp_db, routing) -> None:
    """The category description + include/exclude rules land in the prompt."""
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_response(content=_summary_payload_content()))

    plan = _make_plan(
        [
            CategoryPlan(
                name="ai_research",
                description="Foundation model and agentic research.",
                include_rules=["scaling results"],
                exclude_rules=["unsubstantiated demos"],
                tasks=[],
            ),
        ]
    )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        _seed_kept_cluster(conn, run_id, key="https://a.example/", category="ai_research")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        await summarize_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            plan=plan,
            model="stub-summary-model",
        )

    system = captured[0]["messages"][0]["content"]
    assert "ai_research" in system
    assert "Foundation model and agentic research." in system
    assert "scaling results" in system
    assert "unsubstantiated demos" in system


async def test_summarize_clusters_unknown_category_falls_back(temp_db, routing) -> None:
    """A cluster's category slug not matching any configured plan
    entry → the prompt skips the category-context block rather than
    raising. Mirrors the 9c posture: prefer accepting reasonable
    absences over failing.
    """
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_response(content=_summary_payload_content()))

    plan = _make_plan([CategoryPlan(name="other_category", description="x", tasks=[])])

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        _seed_kept_cluster(conn, run_id, key="https://a.example/", category="missing")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        summarized = await summarize_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            plan=plan,
            model="stub-summary-model",
        )

    assert summarized == 1
    system = captured[0]["messages"][0]["content"]
    assert "Category context:" not in system


async def test_summarize_clusters_per_cluster_failure_degrades_coverage(temp_db, routing) -> None:
    """LLM call raises on cluster A; cluster B's call succeeds. A stays
    at 'kept', B advances to 'summarized'. Coverage counter increments
    by exactly one. The run does NOT abort.
    """
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        # First cluster (A) returns 500 → fails after retries=1.
        # Second cluster (B) succeeds.
        if len(requests_seen) == 1:
            return httpx.Response(500)
        return httpx.Response(200, json=_chat_response(content=_summary_payload_content()))

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        cluster_a = _seed_kept_cluster(conn, run_id, key="https://a.example/")
        cluster_b = _seed_kept_cluster(conn, run_id, key="https://b.example/")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        coverage = Coverage()

        summarized = await summarize_clusters(
            conn,
            run_id,
            client,
            coverage,
            plan=_make_plan(),
            model="stub-summary-model",
        )

        assert summarized == 1
        assert coverage.failed_summary_clusters == 1

        status_a = conn.execute(
            "SELECT status FROM item_clusters WHERE id = ?", (cluster_a,)
        ).fetchone()["status"]
        status_b = conn.execute(
            "SELECT status FROM item_clusters WHERE id = ?", (cluster_b,)
        ).fetchone()["status"]
        assert status_a == "kept"
        assert status_b == "summarized"

        # The failed cluster has no item_summaries row.
        a_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM item_summaries WHERE cluster_id = ?", (cluster_a,)
        ).fetchone()["n"]
        b_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM item_summaries WHERE cluster_id = ?", (cluster_b,)
        ).fetchone()["n"]
        assert a_rows == 0
        assert b_rows == 1


async def test_summarize_clusters_is_replay_safe(temp_db, routing) -> None:
    """A cluster already at 'summarized' is not reprocessed on a re-run.

    The pending-only filter at the SQL layer is what makes this true —
    documented load-bearing behavior. Verified by running summarize
    twice against the same DB and asserting only one call fired.
    """
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_chat_response(content=_summary_payload_content()))

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        cluster_id = _seed_kept_cluster(conn, run_id, key="https://a.example/")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        first = await summarize_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            plan=_make_plan(),
            model="stub-summary-model",
        )
        assert first == 1

        # Second sweep over the same run — cluster is now 'summarized'.
        second = await summarize_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            plan=_make_plan(),
            model="stub-summary-model",
        )
        assert second == 0
        assert calls == 1

        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM item_summaries WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()["n"]
        assert rows == 1  # Not doubled.


async def test_summarize_clusters_writes_audit_row(temp_db, routing) -> None:
    """Each successful cluster_summary call writes one ``llm_calls`` row
    with ``stage='cluster_summary'`` and ``prompt_version='summary.v1'``.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(content=_summary_payload_content()))

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        _seed_kept_cluster(conn, run_id, key="https://a.example/")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        await summarize_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            plan=_make_plan(),
            model="stub-summary-model",
        )

        rows = conn.execute(
            "SELECT stage, provider, status, prompt_version FROM llm_calls WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["stage"] == "cluster_summary"
        assert rows[0]["provider"] == "vmlx"
        assert rows[0]["status"] == "succeeded"
        assert rows[0]["prompt_version"] == "summary.v1"


async def test_summarize_clusters_surfaces_member_body_in_prompt(temp_db, routing) -> None:
    """The cluster's full member content (trafilatura body) reaches the
    LLM via the user message — the brief's grounding depends on this.
    """
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_response(content=_summary_payload_content()))

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        _seed_kept_cluster(
            conn,
            run_id,
            key="https://a.example/",
            content="Distinctive trafilatura body content for grounding.",
        )
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        await summarize_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            plan=_make_plan(),
            model="stub-summary-model",
        )

    user = captured[0]["messages"][1]["content"]
    assert "Distinctive trafilatura body content" in user
