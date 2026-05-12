"""Tests for the relevance filter.

Step 9a covers the pure layer (prompt construction, parse helpers, count
guards). Step 9b adds async orchestration tests against a real
:class:`LLMClient` backed by :class:`httpx.MockTransport` — no live vMLX
in CI.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import Any

import httpx
import pytest

from clawfeed_intel import db as worker_db
from clawfeed_intel.llm import (
    LLMClient,
    RelevanceBatchResponse,
    RelevanceVerdict,
    RetryConfig,
    RoutingConfig,
)
from clawfeed_intel.pipeline.relevance import (
    PROMPT_VERSION,
    RelevanceCluster,
    build_relevance_messages,
    filter_clusters,
    parse_relevance_verdicts,
)
from clawfeed_intel.runs import Coverage
from clawfeed_intel.sources import CategoryPlan


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _category(
    name: str,
    *,
    description: str = "",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> CategoryPlan:
    return CategoryPlan(
        name=name,
        description=description,
        include_rules=include or [],
        exclude_rules=exclude or [],
    )


def _cluster(
    cluster_id: int,
    *,
    title: str = "Sample headline",
    canonical_url: str = "https://example.com/article",
    member_urls: tuple[str, ...] = (),
    excerpt: str = "",
) -> RelevanceCluster:
    return RelevanceCluster(
        cluster_id=cluster_id,
        title=title,
        canonical_url=canonical_url,
        member_urls=member_urls,
        excerpt=excerpt,
    )


@pytest.fixture
def sample_categories() -> list[CategoryPlan]:
    return [
        _category(
            "startup_funding",
            description="Funding rounds and acquisitions.",
            include=["announced rounds", "new fund closes"],
            exclude=["recirculated old news"],
        ),
        _category(
            "ai_research",
            description="Foundation model and agentic research.",
            include=["scaling results", "alignment with practical bearing"],
        ),
    ]


# ── Prompt version ────────────────────────────────────────────────────────────


def test_prompt_version_is_relevance_v1() -> None:
    """First iteration of the relevance prompt; bump on behavior changes."""
    assert PROMPT_VERSION == "relevance.v1"


# ── build_relevance_messages — shape ──────────────────────────────────────────


def test_build_messages_returns_system_then_user(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages([_cluster(1)], sample_categories)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_build_messages_rejects_empty_batch(
    sample_categories: list[CategoryPlan],
) -> None:
    """An empty batch has no positional slots — fail loud rather than call the LLM."""
    with pytest.raises(ValueError, match="must not be empty"):
        build_relevance_messages([], sample_categories)


# ── build_relevance_messages — system message contents ────────────────────────


def test_system_message_lists_every_category_name(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages([_cluster(1)], sample_categories)
    system = msgs[0]["content"]
    assert "startup_funding" in system
    assert "ai_research" in system


def test_system_message_includes_category_descriptions_and_rules(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages([_cluster(1)], sample_categories)
    system = msgs[0]["content"]
    assert "Funding rounds and acquisitions." in system
    assert "announced rounds" in system
    assert "recirculated old news" in system


def test_system_message_states_keep_many_policy(
    sample_categories: list[CategoryPlan],
) -> None:
    """Load-bearing: architecture doc forbids a top-N selector posture."""
    system = build_relevance_messages([_cluster(1)], sample_categories)[0]["content"]
    assert "NOT a top-N selector" in system


def test_system_message_documents_response_format(
    sample_categories: list[CategoryPlan],
) -> None:
    """The prompt must spell out the JSON contract explicitly so local models
    don't drift into markdown-fenced replies under load.
    """
    system = build_relevance_messages([_cluster(1)], sample_categories)[0]["content"]
    assert '"verdicts"' in system
    assert "Begin your response with `{`" in system
    assert "no markdown fencing" in system


def test_system_message_handles_empty_categories() -> None:
    """Defensive — the resolver should always supply at least one category,
    but an empty list shouldn't crash prompt construction.
    """
    msgs = build_relevance_messages([_cluster(1)], [])
    assert "(none configured)" in msgs[0]["content"]


# ── build_relevance_messages — user message contents ──────────────────────────


def test_user_message_announces_batch_size(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages([_cluster(1), _cluster(2), _cluster(3)], sample_categories)
    user = msgs[1]["content"]
    assert "Batch of 3 clusters" in user
    assert "same order" in user


def test_user_message_lists_clusters_in_order(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages(
        [
            _cluster(1, title="First headline", canonical_url="https://a.example/"),
            _cluster(2, title="Second headline", canonical_url="https://b.example/"),
            _cluster(3, title="Third headline", canonical_url="https://c.example/"),
        ],
        sample_categories,
    )
    user = msgs[1]["content"]
    pos_1 = user.index("First headline")
    pos_2 = user.index("Second headline")
    pos_3 = user.index("Third headline")
    assert pos_1 < pos_2 < pos_3
    assert "[1]" in user and "[2]" in user and "[3]" in user


def test_user_message_includes_canonical_url(
    sample_categories: list[CategoryPlan],
) -> None:
    url = "https://techcrunch.com/anthropic-series-b"
    msgs = build_relevance_messages([_cluster(1, canonical_url=url)], sample_categories)
    assert url in msgs[1]["content"]


def test_user_message_includes_excerpt_when_present(
    sample_categories: list[CategoryPlan],
) -> None:
    excerpt = "Anthropic announced a new financing round."
    msgs = build_relevance_messages([_cluster(1, excerpt=excerpt)], sample_categories)
    assert excerpt in msgs[1]["content"]


def test_user_message_omits_excerpt_section_when_empty(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages([_cluster(1, excerpt="")], sample_categories)
    assert "excerpt:" not in msgs[1]["content"]


def test_user_message_falls_back_when_title_blank(
    sample_categories: list[CategoryPlan],
) -> None:
    """Empty title shouldn't produce a confusing blank cluster line."""
    msgs = build_relevance_messages([_cluster(1, title="")], sample_categories)
    assert "(untitled)" in msgs[1]["content"]


def test_user_message_lists_member_urls_for_folded_clusters(
    sample_categories: list[CategoryPlan],
) -> None:
    """L2/L3 folds expose multiple member URLs as cross-source evidence."""
    msgs = build_relevance_messages(
        [
            _cluster(
                1,
                canonical_url="https://npr.org/story",
                member_urls=(
                    "https://npr.org/story",
                    "https://kqed.org/syndicated",
                    "https://wbur.org/syndicated",
                ),
            )
        ],
        sample_categories,
    )
    user = msgs[1]["content"]
    assert "additional source urls" in user
    assert "kqed.org/syndicated" in user
    assert "wbur.org/syndicated" in user


def test_user_message_deduplicates_member_urls_against_canonical(
    sample_categories: list[CategoryPlan],
) -> None:
    """The canonical URL is already shown — don't repeat it in the extra list."""
    msgs = build_relevance_messages(
        [
            _cluster(
                1,
                canonical_url="https://npr.org/story",
                member_urls=(
                    "https://npr.org/story",
                    "https://kqed.org/syndicated",
                ),
            )
        ],
        sample_categories,
    )
    user = msgs[1]["content"]
    # Canonical URL appears once (in `url:` line), not duplicated in the extras section.
    assert user.count("https://npr.org/story") == 1


def test_user_message_omits_additional_urls_when_single_member(
    sample_categories: list[CategoryPlan],
) -> None:
    msgs = build_relevance_messages(
        [
            _cluster(
                1,
                canonical_url="https://example.com/a",
                member_urls=("https://example.com/a",),
            )
        ],
        sample_categories,
    )
    assert "additional source urls" not in msgs[1]["content"]


# ── parse_relevance_verdicts ──────────────────────────────────────────────────


def _verdict(keep: bool = True, category: str = "ai_research") -> RelevanceVerdict:
    return RelevanceVerdict(
        keep=keep,
        category=category,
        score=0.7,
        reason="reason",
    )


def test_parse_returns_validated_verdicts_in_order() -> None:
    response = RelevanceBatchResponse(
        verdicts=[
            _verdict(keep=True, category="ai_research"),
            _verdict(keep=False, category="startup_funding"),
        ]
    )
    verdicts = parse_relevance_verdicts(response, expected_count=2)
    assert [v.keep for v in verdicts] == [True, False]
    assert [v.category for v in verdicts] == ["ai_research", "startup_funding"]


def test_parse_raises_on_count_mismatch_too_few() -> None:
    response = RelevanceBatchResponse(verdicts=[_verdict()])
    with pytest.raises(ValueError, match="expected 3, got 1"):
        parse_relevance_verdicts(response, expected_count=3)


def test_parse_raises_on_count_mismatch_too_many() -> None:
    response = RelevanceBatchResponse(verdicts=[_verdict(), _verdict()])
    with pytest.raises(ValueError, match="expected 1, got 2"):
        parse_relevance_verdicts(response, expected_count=1)


def test_parse_raises_on_empty_verdicts_when_batch_was_nonempty() -> None:
    """An empty batch response against a real batch makes positional
    verdict assignment meaningless — load-bearing guard.
    """
    response = RelevanceBatchResponse(verdicts=[])
    with pytest.raises(ValueError, match="expected 2, got 0"):
        parse_relevance_verdicts(response, expected_count=2)


# ── Orchestration fixtures (step 9b) ──────────────────────────────────────────


@pytest.fixture
def routing() -> RoutingConfig:
    """Minimal routing config sized for relevance-filter tests."""
    return RoutingConfig.model_validate(
        {
            "providers": {
                "vmlx": {"base_url": "http://127.0.0.1:8080/v1"},
            },
            "stages": {
                "relevance_filter": {
                    "provider": "vmlx",
                    "model": "stub-model",
                    "timeout_seconds": 30,
                    "batch_size": 12,
                },
            },
        }
    )


def _verdict_payload(
    keep: bool = True,
    category: str = "ai_research",
    score: float = 0.7,
    reason: str = "stub",
    event_type: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "keep": keep,
        "category": category,
        "score": score,
        "reason": reason,
    }
    if event_type is not None:
        out["event_type"] = event_type
    return out


def _chat_response(*, content: str, model: str = "stub-model") -> dict[str, Any]:
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


def _seed_run(conn: sqlite3.Connection) -> int:
    return worker_db.create_run(
        conn,
        run_type="daily",
        window_start="2026-05-10T00:00:00+00:00",
        window_end="2026-05-11T00:00:00+00:00",
    )


def _seed_cluster(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    key: str,
    title: str = "",
    excerpt: str = "",
    extra_member_urls: tuple[str, ...] = (),
) -> int:
    """Seed one cluster with a representative member and optional extras."""
    rep_id, _ = worker_db.upsert_raw_item(
        conn,
        run_id=run_id,
        source_type="rss",
        dedup_key=f"{key}-rep",
        title=title or key,
        url=key,
        canonical_url=key,
        excerpt=excerpt,
        content="",
    )
    extra_ids: list[int] = []
    for i, url in enumerate(extra_member_urls):
        extra_id, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="rss",
            dedup_key=f"{key}-extra-{i}",
            title=title or key,
            url=url,
            canonical_url=url,
            content="",
        )
        extra_ids.append(extra_id)
    cluster_id, _ = worker_db.create_cluster(
        conn,
        run_id=run_id,
        cluster_key=key,
        title=title or key,
        raw_item_ids=[rep_id, *extra_ids],
    )
    return cluster_id


# ── Orchestration tests ───────────────────────────────────────────────────────


async def test_filter_clusters_no_pending_returns_zero(temp_db, routing) -> None:
    """No pending clusters → return 0, never call the LLM."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        coverage = Coverage()
        kept = await filter_clusters(conn, run_id, client, coverage, categories=[], batch_size=12)
        assert kept == 0
        assert calls == []
        assert coverage.failed_filter_batches == 0


async def test_filter_clusters_applies_verdicts_in_order(temp_db, routing) -> None:
    """Three clusters, verdicts [keep, drop, keep] → two promoted, one rejected.

    Status, score, category, event_type, and filter_reason all land on the
    item_clusters row in the order the LLM emitted them.
    """
    verdicts = {
        "verdicts": [
            _verdict_payload(
                keep=True,
                category="ai_research",
                score=0.91,
                reason="Strong scaling result.",
                event_type="paper",
            ),
            _verdict_payload(
                keep=False,
                category="ai_research",
                score=0.12,
                reason="Marginal benchmark delta.",
            ),
            _verdict_payload(
                keep=True,
                category="startup_funding",
                score=0.77,
                reason="Series B announced.",
                event_type="funding_round",
            ),
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(content=json.dumps(verdicts)))

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        c1 = _seed_cluster(conn, run_id, key="https://x/a", title="Alpha")
        c2 = _seed_cluster(conn, run_id, key="https://x/b", title="Beta")
        c3 = _seed_cluster(conn, run_id, key="https://x/c", title="Gamma")
        coverage = Coverage()
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        kept = await filter_clusters(
            conn,
            run_id,
            client,
            coverage,
            categories=[CategoryPlan(name="ai_research")],
        )
        assert kept == 2

        rows = {
            row["id"]: row
            for row in conn.execute(
                "SELECT id, status, relevance_score, category, event_type, filter_reason "
                "FROM item_clusters WHERE run_id = ?",
                (run_id,),
            )
        }
        assert rows[c1]["status"] == "kept"
        assert rows[c1]["relevance_score"] == 0.91
        assert rows[c1]["category"] == "ai_research"
        assert rows[c1]["event_type"] == "paper"
        assert rows[c2]["status"] == "filtered_out"
        assert rows[c2]["relevance_score"] == 0.12
        assert rows[c3]["status"] == "kept"
        assert rows[c3]["category"] == "startup_funding"


async def test_filter_clusters_respects_batch_size(temp_db, routing) -> None:
    """``batch_size=2`` with 3 clusters → 2 LLM calls (2 + 1)."""
    call_batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user = body["messages"][1]["content"]
        # The user message announces "Batch of N clusters" — that's our
        # ground truth for how many verdicts to fabricate.
        n = int(user.split("Batch of ", 1)[1].split(" ", 1)[0])
        call_batch_sizes.append(n)
        verdicts = {"verdicts": [_verdict_payload() for _ in range(n)]}
        return httpx.Response(200, json=_chat_response(content=json.dumps(verdicts)))

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        for i in range(3):
            _seed_cluster(conn, run_id, key=f"https://x/{i}")
        coverage = Coverage()
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        kept = await filter_clusters(
            conn,
            run_id,
            client,
            coverage,
            categories=[CategoryPlan(name="ai_research")],
            batch_size=2,
        )
        assert kept == 3
        assert call_batch_sizes == [2, 1]


async def test_filter_clusters_per_batch_failure_degrades_coverage(temp_db, routing) -> None:
    """LLM 5xx for a batch → coverage.failed_filter_batches += 1; clusters
    stay at 'pending'; run continues without raising.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        cid = _seed_cluster(conn, run_id, key="https://x/a")
        coverage = Coverage()
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        kept = await filter_clusters(
            conn,
            run_id,
            client,
            coverage,
            categories=[CategoryPlan(name="ai_research")],
            batch_size=12,
        )
        assert kept == 0
        assert coverage.failed_filter_batches == 1

        row = conn.execute("SELECT status FROM item_clusters WHERE id = ?", (cid,)).fetchone()
        assert row["status"] == "pending"


async def test_filter_clusters_isolates_failed_batch(temp_db, routing) -> None:
    """First batch fails, second batch succeeds — the survivors still get
    verdicts. Failed-batch clusters stay at 'pending'.
    """
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(500, json={"error": "boom"})
        body = json.loads(request.content)
        n = int(body["messages"][1]["content"].split("Batch of ", 1)[1].split(" ", 1)[0])
        return httpx.Response(
            200,
            json=_chat_response(
                content=json.dumps({"verdicts": [_verdict_payload() for _ in range(n)]})
            ),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        c_fail = _seed_cluster(conn, run_id, key="https://x/0")
        c_ok = _seed_cluster(conn, run_id, key="https://x/1")
        coverage = Coverage()
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        kept = await filter_clusters(
            conn,
            run_id,
            client,
            coverage,
            categories=[CategoryPlan(name="ai_research")],
            batch_size=1,
        )
        assert kept == 1
        assert coverage.failed_filter_batches == 1

        statuses = {
            r["id"]: r["status"]
            for r in conn.execute(
                "SELECT id, status FROM item_clusters WHERE run_id = ?", (run_id,)
            )
        }
        assert statuses[c_fail] == "pending"
        assert statuses[c_ok] == "kept"


async def test_filter_clusters_sends_low_temperature(temp_db, routing) -> None:
    """Structured-output prompts pin a low temperature so local models keep
    JSON well-formed under load.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json=_chat_response(content=json.dumps({"verdicts": [_verdict_payload()]})),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        _seed_cluster(conn, run_id, key="https://x/a")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        await filter_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            categories=[CategoryPlan(name="ai_research")],
        )
        assert captured["temperature"] == 0.1


async def test_filter_clusters_max_tokens_scales_with_batch_size(temp_db, routing) -> None:
    """Discovered during the live-vMLX smoke: local MLX defaults to ~1024
    completion tokens, which truncates verdict arrays for batches of 12.
    The completion budget must scale with batch size.
    """
    captured: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body["max_tokens"])
        n = int(body["messages"][1]["content"].split("Batch of ", 1)[1].split(" ", 1)[0])
        return httpx.Response(
            200,
            json=_chat_response(
                content=json.dumps({"verdicts": [_verdict_payload() for _ in range(n)]})
            ),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        for i in range(5):
            _seed_cluster(conn, run_id, key=f"https://x/{i}")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        await filter_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            categories=[CategoryPlan(name="ai_research")],
            batch_size=2,
        )

        # 3 batches: [2, 2, 1]. max_tokens scales with the count in each.
        assert captured == [
            320 * 2 + 256,
            320 * 2 + 256,
            320 * 1 + 256,
        ]


async def test_filter_clusters_count_mismatch_treated_as_batch_failure(temp_db, routing) -> None:
    """LLM returns the wrong number of verdicts → batch fails; clusters stay
    pending; run continues. Positional verdict assignment would be wrong
    if applied, so the fail-loud guard is the right move.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        # Two clusters in batch but only one verdict returned.
        return httpx.Response(
            200,
            json=_chat_response(content=json.dumps({"verdicts": [_verdict_payload()]})),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        _seed_cluster(conn, run_id, key="https://x/a")
        _seed_cluster(conn, run_id, key="https://x/b")
        coverage = Coverage()
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        kept = await filter_clusters(
            conn,
            run_id,
            client,
            coverage,
            categories=[CategoryPlan(name="ai_research")],
        )
        assert kept == 0
        assert coverage.failed_filter_batches == 1


async def test_filter_clusters_skips_non_pending(temp_db, routing) -> None:
    """A cluster already at 'kept' or 'filtered_out' must not be reprocessed
    — replaying a partial run shouldn't double-apply or revert verdicts.
    """
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        n = int(body["messages"][1]["content"].split("Batch of ", 1)[1].split(" ", 1)[0])
        calls.append(n)
        return httpx.Response(
            200,
            json=_chat_response(
                content=json.dumps({"verdicts": [_verdict_payload() for _ in range(n)]})
            ),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        already_kept = _seed_cluster(conn, run_id, key="https://x/a")
        worker_db.update_cluster_verdict(
            conn,
            cluster_id=already_kept,
            status="kept",
            relevance_score=0.5,
            category="ai_research",
            event_type=None,
            filter_reason="prior",
        )
        _seed_cluster(conn, run_id, key="https://x/b")  # pending
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        kept = await filter_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            categories=[CategoryPlan(name="ai_research")],
        )
        # Only the pending cluster reached the LLM.
        assert calls == [1]
        assert kept == 1

        # Prior verdict on the already-kept cluster preserved.
        row = conn.execute(
            "SELECT filter_reason FROM item_clusters WHERE id = ?",
            (already_kept,),
        ).fetchone()
        assert row["filter_reason"] == "prior"


async def test_filter_clusters_writes_llm_calls_row(temp_db, routing) -> None:
    """Wiring sanity: the orchestration goes through ``LLMClient`` so the
    audit trail in ``llm_calls`` is populated.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_response(content=json.dumps({"verdicts": [_verdict_payload()]})),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        _seed_cluster(conn, run_id, key="https://x/a")
        client = _make_client(routing, handler, conn=conn, run_id=run_id)

        await filter_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            categories=[CategoryPlan(name="ai_research")],
        )

        rows = conn.execute(
            "SELECT stage, provider, model, status, prompt_version FROM llm_calls WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["stage"] == "relevance_filter"
        assert rows[0]["provider"] == "vmlx"
        assert rows[0]["status"] == "succeeded"
        assert rows[0]["prompt_version"] == PROMPT_VERSION


async def test_filter_clusters_loads_member_urls_from_folds(temp_db, routing) -> None:
    """L2/L3-folded clusters expose every member URL to the LLM prompt as
    cross-source evidence. The orchestrator's ``_load_pending_clusters``
    must surface them correctly.
    """
    captured_messages: list[list[dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_messages.append(body["messages"])
        return httpx.Response(
            200,
            json=_chat_response(content=json.dumps({"verdicts": [_verdict_payload()]})),
        )

    with closing(worker_db.connect(temp_db)) as conn:
        run_id = _seed_run(conn)
        _seed_cluster(
            conn,
            run_id,
            key="https://npr.org/story",
            title="Folded story",
            extra_member_urls=("https://kqed.org/syn", "https://wbur.org/syn"),
        )
        client = _make_client(routing, handler, conn=conn, run_id=run_id)
        await filter_clusters(
            conn,
            run_id,
            client,
            Coverage(),
            categories=[CategoryPlan(name="ai_research")],
        )

        user_content = captured_messages[0][1]["content"]
        assert "kqed.org/syn" in user_content
        assert "wbur.org/syn" in user_content
