from __future__ import annotations

import json
from contextlib import closing

import httpx
import pytest

from clawfeed_intel import db as worker_db
from clawfeed_intel.llm import LLMClient, RetryConfig
from clawfeed_intel.pipeline.orchestrator import run_daily
from clawfeed_intel.sources import build_source_plan


def _isolate_config(monkeypatch, tmp_path, body: str = "categories: {}\n"):
    """Point the resolver at a tmp config so orchestrator tests don't depend
    on the editorial config that ships in the repo."""
    config = tmp_path / "intel-sources.yaml"
    config.write_text(body, encoding="utf-8")
    monkeypatch.setattr("clawfeed_intel.sources.DEFAULT_CONFIG_PATH", config)
    return config


def _empty_fetcher_registry(monkeypatch):
    """Detach the orchestrator from whichever real fetchers happen to be
    registered. Each fetcher step (6.1 RSS, 6.2 arXiv, …) adds its own
    callable at import time, which would otherwise hit live HTTP from
    orchestrator tests that exercise the 'no-fetcher' code path.

    We patch ``runner.FETCHER_REGISTRY`` (the binding the runner actually
    reads) rather than ``base.FETCHER_REGISTRY``; the runner ``from .base
    import`` rebinds it into its own module namespace, and monkeypatch on
    that namespace is what the function lookup sees at call time.
    """
    monkeypatch.setattr("clawfeed_intel.fetchers.runner.FETCHER_REGISTRY", {})


def _stub_llm_client(monkeypatch, *, keep_all: bool = True) -> None:
    """Replace ``orchestrator._build_llm_client`` with a deterministic stub.

    The stub uses :class:`httpx.MockTransport` to short-circuit HTTP — no
    live vMLX calls under unit tests. Verdict count is read from the
    user message's "Batch of N clusters" header so the stub always
    matches the requested batch size.

    ``keep_all`` toggles between an all-keep and an all-reject response.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user_content = body["messages"][1]["content"]
        batch_size = int(user_content.split("Batch of ", 1)[1].split(" ", 1)[0])
        verdicts = [
            {
                "keep": keep_all,
                "category": "scratch",
                "score": 0.9 if keep_all else 0.1,
                "reason": "stub verdict",
            }
            for _ in range(batch_size)
        ]
        chat_body = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "model": body.get("model", "stub-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"verdicts": verdicts}),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return httpx.Response(200, json=chat_body)

    transport = httpx.MockTransport(handler)

    def _build(routing, *, conn, run_id):
        return LLMClient(
            routing,
            transport=transport,
            conn=conn,
            run_id=run_id,
            retry_config=RetryConfig(max_attempts=1, wait_min_seconds=0, wait_max_seconds=0),
        )

    monkeypatch.setattr("clawfeed_intel.pipeline.orchestrator._build_llm_client", _build)


def test_run_daily_publishes_digest_and_links_run(temp_db, monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    _stub_llm_client(monkeypatch)
    with closing(worker_db.connect(temp_db)) as conn:
        digest_id = run_daily("24h", conn=conn)

        digest = conn.execute("SELECT * FROM digests WHERE id = ?", (digest_id,)).fetchone()
        assert digest is not None
        assert digest["type"] == "daily"
        assert "Daily Intelligence Brief" in digest["content"]

        meta = json.loads(digest["metadata"])
        assert meta["brief_kind"] == "daily"
        assert meta["run_id"] > 0
        assert meta["window_start"].endswith("+00:00")
        assert meta["window_end"].endswith("+00:00")
        assert "coverage" in meta
        assert meta["coverage"]["sources_attempted"] == 0
        assert meta["coverage"]["failed_sources"] == []
        assert meta["coverage"]["skipped_sources"] == []
        assert meta["coverage"]["plan_warnings"] == []

        runs = conn.execute("SELECT * FROM intel_runs WHERE digest_id = ?", (digest_id,)).fetchall()
        assert len(runs) == 1
        run = runs[0]
        assert run["status"] == "published"
        assert run["started_at"] is not None
        assert run["finished_at"] is not None
        assert run["error"] is None
        assert run["run_type"] == "daily"


def test_run_daily_skips_resolved_tasks_when_no_fetcher_registered(temp_db, monkeypatch, tmp_path):
    """With the resolver wired in but no fetchers registered, every YAML task
    becomes a 'skipped' coverage entry. The run still publishes; the brief
    just reports an honest empty harness."""
    yaml_body = """
categories:
  scratch:
    sources:
      - kind: rss
        url: https://example.com/feed
      - kind: gdelt
        query: hello
"""
    _isolate_config(monkeypatch, tmp_path, yaml_body)
    _empty_fetcher_registry(monkeypatch)
    _stub_llm_client(monkeypatch)

    with closing(worker_db.connect(temp_db)) as conn:
        digest_id = run_daily("24h", conn=conn)

        plan = build_source_plan(conn)
        expected = sum(len(tasks) for tasks in plan.tasks_by_kind().values())

        meta = json.loads(
            conn.execute("SELECT metadata FROM digests WHERE id = ?", (digest_id,)).fetchone()[
                "metadata"
            ]
        )

    assert expected == 2
    assert meta["coverage"]["sources_attempted"] == 2
    assert meta["coverage"]["sources_succeeded"] == 0
    assert len(meta["coverage"]["skipped_sources"]) == 2
    assert all("no fetcher" in s for s in meta["coverage"]["skipped_sources"])


def test_run_daily_clusters_fetched_items_into_coverage(temp_db, monkeypatch, tmp_path):
    """End-to-end: a fetcher that emits two items with the same canonical_url
    plus one distinct item should produce two Level-1 clusters, surfaced via
    coverage.clusters in the published digest metadata."""
    from clawfeed_intel.fetchers import FetchedItem

    yaml_body = """
categories:
  scratch:
    sources:
      - kind: rss
        url: https://example.com/feed
"""
    _isolate_config(monkeypatch, tmp_path, yaml_body)

    async def stub_fetcher(_conn, _task):
        return [
            FetchedItem(
                source_type="rss",
                dedup_key="https://example.com/a",
                title="Alpha (RSS)",
                url="https://example.com/a",
                canonical_url="https://example.com/a",
                content="alpha body",
                content_hash="hash-a",
            ),
            FetchedItem(
                source_type="rss",
                dedup_key="https://example.com/a?utm=x",
                title="Alpha (also RSS)",
                url="https://example.com/a?utm=x",
                canonical_url="https://example.com/a",
                content="alpha body",
                content_hash="hash-a",
            ),
            FetchedItem(
                source_type="rss",
                dedup_key="https://example.com/b",
                title="Beta",
                url="https://example.com/b",
                canonical_url="https://example.com/b",
                content="beta body",
                content_hash="hash-b",
            ),
        ]

    monkeypatch.setattr(
        "clawfeed_intel.fetchers.runner.FETCHER_REGISTRY",
        {"rss": stub_fetcher},
    )
    _stub_llm_client(monkeypatch)

    with closing(worker_db.connect(temp_db)) as conn:
        digest_id = run_daily("24h", conn=conn)

        meta = json.loads(
            conn.execute("SELECT metadata FROM digests WHERE id = ?", (digest_id,)).fetchone()[
                "metadata"
            ]
        )
        assert meta["coverage"]["raw_items"] == 3
        assert meta["coverage"]["clusters"] == 2
        # Stub LLM keeps every cluster, so kept_clusters mirrors clusters.
        assert meta["coverage"]["kept_clusters"] == 2
        assert meta["coverage"]["failed_filter_batches"] == 0

        run_row = conn.execute(
            "SELECT id FROM intel_runs WHERE digest_id = ?", (digest_id,)
        ).fetchone()
        clusters = conn.execute(
            "SELECT cluster_key, status FROM item_clusters WHERE run_id = ? ORDER BY cluster_key",
            (run_row["id"],),
        ).fetchall()
        assert [c["cluster_key"] for c in clusters] == [
            "https://example.com/a",
            "https://example.com/b",
        ]
        assert all(c["status"] == "kept" for c in clusters)


def test_run_daily_folds_syndicated_items_via_content_hash(temp_db, monkeypatch, tmp_path):
    """End-to-end L2 check: a fetcher that emits two items with different
    canonical URLs but identical content_hash should produce ONE cluster,
    not two — proving that the orchestrator's filtering stage chains
    L1 → L2 in the persisted state."""
    from clawfeed_intel.fetchers import FetchedItem

    yaml_body = """
categories:
  scratch:
    sources:
      - kind: rss
        url: https://example.com/feed
"""
    _isolate_config(monkeypatch, tmp_path, yaml_body)

    async def stub_fetcher(_conn, _task):
        return [
            FetchedItem(
                source_type="rss",
                dedup_key="orig",
                title="Original",
                url="https://example.com/article",
                canonical_url="https://example.com/article",
                content="shared body",
                content_hash="hash-shared",
            ),
            FetchedItem(
                source_type="rss",
                dedup_key="syn",
                title="Yahoo's copy",
                url="https://yahoo.com/article",
                canonical_url="https://yahoo.com/article",
                content="shared body",
                content_hash="hash-shared",
            ),
            FetchedItem(
                source_type="rss",
                dedup_key="distinct",
                title="Beta",
                url="https://example.com/beta",
                canonical_url="https://example.com/beta",
                content="different body",
                content_hash="hash-beta",
            ),
        ]

    monkeypatch.setattr(
        "clawfeed_intel.fetchers.runner.FETCHER_REGISTRY",
        {"rss": stub_fetcher},
    )
    _stub_llm_client(monkeypatch)

    with closing(worker_db.connect(temp_db)) as conn:
        digest_id = run_daily("24h", conn=conn)

        meta = json.loads(
            conn.execute("SELECT metadata FROM digests WHERE id = ?", (digest_id,)).fetchone()[
                "metadata"
            ]
        )
        assert meta["coverage"]["raw_items"] == 3
        assert meta["coverage"]["clusters"] == 2  # L2 collapsed orig+syn → 1; beta stands alone

        run_row = conn.execute(
            "SELECT id FROM intel_runs WHERE digest_id = ?", (digest_id,)
        ).fetchone()
        clusters = conn.execute(
            "SELECT cluster_key FROM item_clusters WHERE run_id = ? ORDER BY cluster_key",
            (run_row["id"],),
        ).fetchall()
        assert [c["cluster_key"] for c in clusters] == [
            "https://example.com/article",  # smaller URL from the syndicated pair
            "https://example.com/beta",
        ]


def test_run_daily_folds_similar_titles_via_l3(temp_db, monkeypatch, tmp_path):
    """End-to-end L3 check: a fetcher emits two items with different URLs
    AND different content_hashes (so L1 and L2 both miss them) but
    overlapping titles within the date window. The orchestrator's
    filtering stage should produce ONE cluster via L3."""
    from clawfeed_intel.fetchers import FetchedItem

    yaml_body = """
categories:
  scratch:
    sources:
      - kind: rss
        url: https://example.com/feed
"""
    _isolate_config(monkeypatch, tmp_path, yaml_body)

    async def stub_fetcher(_conn, _task):
        return [
            FetchedItem(
                source_type="rss",
                dedup_key="outlet-a",
                title="Anthropic raises Series F funding round",
                url="https://outlet-a.example/anthropic",
                canonical_url="https://outlet-a.example/anthropic",
                content="Outlet A's writeup of the funding event.",
                content_hash="hash-outlet-a",
                published_at="2026-05-04T10:00:00+00:00",
            ),
            FetchedItem(
                source_type="rss",
                dedup_key="outlet-b",
                title="Anthropic raises Series F funding round announcement",
                url="https://outlet-b.example/anthropic",
                canonical_url="https://outlet-b.example/anthropic",
                content="Outlet B's distinct take on the same funding event.",
                content_hash="hash-outlet-b",
                published_at="2026-05-04T11:00:00+00:00",
            ),
            FetchedItem(
                source_type="rss",
                dedup_key="distinct",
                title="Stock market closes higher today",
                url="https://example.com/market",
                canonical_url="https://example.com/market",
                content="Unrelated market wrap.",
                content_hash="hash-market",
                published_at="2026-05-04T12:00:00+00:00",
            ),
        ]

    monkeypatch.setattr(
        "clawfeed_intel.fetchers.runner.FETCHER_REGISTRY",
        {"rss": stub_fetcher},
    )
    _stub_llm_client(monkeypatch)

    with closing(worker_db.connect(temp_db)) as conn:
        digest_id = run_daily("24h", conn=conn)

        meta = json.loads(
            conn.execute("SELECT metadata FROM digests WHERE id = ?", (digest_id,)).fetchone()[
                "metadata"
            ]
        )
        assert meta["coverage"]["raw_items"] == 3
        assert meta["coverage"]["clusters"] == 2  # L3 folded the two Anthropic items

        run_row = conn.execute(
            "SELECT id FROM intel_runs WHERE digest_id = ?", (digest_id,)
        ).fetchone()
        cluster_keys = [
            r["cluster_key"]
            for r in conn.execute(
                "SELECT cluster_key FROM item_clusters WHERE run_id = ? ORDER BY cluster_key",
                (run_row["id"],),
            ).fetchall()
        ]
        assert cluster_keys == [
            "https://example.com/market",
            "https://outlet-a.example/anthropic",  # smaller URL won the L3 merge
        ]


def test_run_daily_records_resolver_warning_in_coverage(temp_db, monkeypatch, tmp_path):
    """A missing config produces a PlanWarning that surfaces as
    coverage.plan_warnings — the brief should be able to explain why the
    pool was thin."""
    monkeypatch.setattr(
        "clawfeed_intel.sources.DEFAULT_CONFIG_PATH",
        tmp_path / "absent.yaml",
    )
    _stub_llm_client(monkeypatch)
    with closing(worker_db.connect(temp_db)) as conn:
        digest_id = run_daily("24h", conn=conn)
        meta = json.loads(
            conn.execute("SELECT metadata FROM digests WHERE id = ?", (digest_id,)).fetchone()[
                "metadata"
            ]
        )

    assert meta["coverage"]["plan_warnings"], "expected at least one plan warning"
    assert any("not found" in w for w in meta["coverage"]["plan_warnings"])


def test_run_daily_invalid_window_raises_value_error(temp_db):
    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(ValueError):
            run_daily("not-a-window", conn=conn)


def test_run_daily_marks_failure_on_db_error(temp_db, monkeypatch, tmp_path):
    """If a stage raises, the run row must end up in 'failed' state, not stuck mid-flow."""
    from clawfeed_intel.pipeline import orchestrator

    _isolate_config(monkeypatch, tmp_path)
    _empty_fetcher_registry(monkeypatch)
    _stub_llm_client(monkeypatch)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated stage failure")

    monkeypatch.setattr(orchestrator.db, "create_digest", boom)

    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(RuntimeError, match="simulated stage failure"):
            run_daily("24h", conn=conn)

        rows = conn.execute("SELECT * FROM intel_runs ORDER BY id DESC LIMIT 1").fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0]["finished_at"] is not None
        assert "simulated stage failure" in (rows[0]["error"] or "")
        assert rows[0]["digest_id"] is None
