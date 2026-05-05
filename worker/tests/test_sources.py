"""Tests for the source-plan resolver.

The resolver merges editorial YAML and DB-tagged ClawFeed sources into a
typed plan. These tests cover three concerns:

1. Successful merge — every kind round-trips, provenance is preserved.
2. Soft failure — single bad entries become warnings, the rest of the plan
   survives. This is the hard requirement that failed sources degrade
   coverage instead of failing the run.
3. Hard failure — only top-level YAML corruption raises.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from clawfeed_intel import db as worker_db
from clawfeed_intel.sources import (
    KNOWN_TASK_KINDS,
    GdeltTask,
    GithubTrendingTask,
    HnTask,
    RedditTask,
    RssTask,
    SecEdgarTask,
    WebsiteTask,
    build_source_plan,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "intel-sources.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _insert_source(
    conn: sqlite3.Connection,
    *,
    name: str,
    type_: str,
    config: dict,
    is_active: int = 1,
) -> int:
    cur = conn.execute(
        "INSERT INTO sources (name, type, config, is_active) VALUES (?, ?, ?, ?)",
        (name, type_, json.dumps(config), is_active),
    )
    return int(cur.lastrowid)


def _tag(conn: sqlite3.Connection, source_id: int, category: str) -> None:
    conn.execute(
        "INSERT INTO source_categories (source_id, category) VALUES (?, ?)",
        (source_id, category),
    )


# ── happy path: YAML only ─────────────────────────────────────────────────────


_FULL_YAML = """
profile:
  timezone: America/Los_Angeles
  daily_window_hours: 24
  default_language: en

categories:
  startup_funding:
    description: Funding rounds.
    include:
      - announced funding rounds
    exclude:
      - opinion posts
    sources:
      - kind: gdelt
        query: '"raised" "series a"'
      - kind: sec_edgar
        forms: ["D", "D/A"]

  ai_research:
    description: New research.
    sources:
      - kind: arxiv
        categories: ["cs.AI", "cs.LG"]

  ai_coding_tools:
    description: Coding tool moves.
    sources:
      - kind: github_search
        query: "topic:ai-coding"
      - kind: github_trending
        language: null

  scratch:
    description: Hand-curated.
    sources:
      - kind: rss
        url: https://example.com/feed
      - kind: website
        url: https://example.com/blog
      - kind: hn
        list: top
        min_score: 100
      - kind: reddit
        subreddit: MachineLearning

dynamic_search:
  enabled_sources:
    - raw_cache
    - gdelt
    - hn_algolia
"""


def test_full_yaml_resolves_every_known_kind(tmp_path, temp_db):
    """Every fetcher family has at least one task and provenance is yaml."""
    config_path = _write_yaml(tmp_path, _FULL_YAML)
    with closing(worker_db.connect(temp_db)) as conn:
        plan = build_source_plan(conn, config_path=config_path)

    assert plan.warnings == []
    assert plan.profile.timezone == "America/Los_Angeles"
    assert plan.profile.daily_window_hours == 24
    assert plan.profile.default_language == "en"
    assert plan.dynamic_search == ["raw_cache", "gdelt", "hn_algolia"]

    by_kind = plan.tasks_by_kind()
    # Every kind we know about has a bucket; the YAML exercises all of them.
    assert set(by_kind.keys()) == set(KNOWN_TASK_KINDS)
    for kind in KNOWN_TASK_KINDS:
        assert by_kind[kind], f"kind {kind!r} has no tasks"
        assert all(t.origin == "yaml" for t in by_kind[kind])
        assert all(t.source_id is None for t in by_kind[kind])

    # Spot-check a few specific shapes
    funding = plan.category("startup_funding")
    assert funding is not None
    assert funding.include_rules == ["announced funding rounds"]
    assert funding.exclude_rules == ["opinion posts"]
    gdelt = next(t for t in funding.tasks if isinstance(t.task, GdeltTask))
    assert gdelt.task.query == '"raised" "series a"'
    sec = next(t for t in funding.tasks if isinstance(t.task, SecEdgarTask))
    assert sec.task.forms == ["D", "D/A"]


def test_missing_config_returns_empty_plan_with_warning(tmp_path, temp_db):
    """No YAML on disk → we still return a plan, with a single warning."""
    nonexistent = tmp_path / "does-not-exist.yaml"
    with closing(worker_db.connect(temp_db)) as conn:
        plan = build_source_plan(conn, config_path=nonexistent)

    assert plan.categories == []
    assert plan.dynamic_search == []
    assert len(plan.warnings) == 1
    w = plan.warnings[0]
    assert w.origin == "config"
    assert w.category is None
    assert "not found" in w.message


def test_empty_categories_block_is_quiet(tmp_path, temp_db):
    """Empty config is not a degradation — no warning emitted."""
    config_path = _write_yaml(tmp_path, "profile: {}\ncategories: {}\n")
    with closing(worker_db.connect(temp_db)) as conn:
        plan = build_source_plan(conn, config_path=config_path)

    assert plan.categories == []
    assert plan.warnings == []


def test_top_level_yaml_corruption_raises(tmp_path, temp_db):
    """A non-mapping top-level is a deploy bug; let it raise."""
    config_path = _write_yaml(tmp_path, "- this is a list\n- not a mapping\n")
    with closing(worker_db.connect(temp_db)) as conn:
        with pytest.raises(ValueError, match="must be a mapping"):
            build_source_plan(conn, config_path=config_path)


# ── soft failures: single bad entries ────────────────────────────────────────


def test_unknown_kind_skipped_with_warning(tmp_path, temp_db):
    yaml_body = """
categories:
  scratch:
    sources:
      - kind: telepathy
        topic: thoughts
      - kind: rss
        url: https://example.com/feed
"""
    config_path = _write_yaml(tmp_path, yaml_body)
    with closing(worker_db.connect(temp_db)) as conn:
        plan = build_source_plan(conn, config_path=config_path)

    scratch = plan.category("scratch")
    assert scratch is not None
    assert len(scratch.tasks) == 1
    assert isinstance(scratch.tasks[0].task, RssTask)
    assert any("telepathy" in w.message for w in plan.warnings)
    assert all(w.origin == "yaml" for w in plan.warnings)


def test_missing_kind_skipped_with_warning(tmp_path, temp_db):
    yaml_body = """
categories:
  scratch:
    sources:
      - url: https://example.com/feed   # no kind
      - kind: rss
        url: https://example.com/feed2
"""
    config_path = _write_yaml(tmp_path, yaml_body)
    with closing(worker_db.connect(temp_db)) as conn:
        plan = build_source_plan(conn, config_path=config_path)

    scratch = plan.category("scratch")
    assert scratch is not None
    assert len(scratch.tasks) == 1
    assert any("missing 'kind'" in w.message for w in plan.warnings)


def test_invalid_field_shape_skipped_with_warning(tmp_path, temp_db):
    yaml_body = """
categories:
  scratch:
    sources:
      - kind: arxiv
        categories: not-a-list
      - kind: gdelt
        # missing required query field
      - kind: rss
        url: https://example.com/feed
"""
    config_path = _write_yaml(tmp_path, yaml_body)
    with closing(worker_db.connect(temp_db)) as conn:
        plan = build_source_plan(conn, config_path=config_path)

    scratch = plan.category("scratch")
    assert scratch is not None
    assert len(scratch.tasks) == 1
    assert isinstance(scratch.tasks[0].task, RssTask)

    bad_messages = [w.message for w in plan.warnings if w.origin == "yaml"]
    assert len(bad_messages) == 2
    assert any("arxiv" in m for m in bad_messages)
    assert any("gdelt" in m for m in bad_messages)


def test_non_mapping_source_entry_skipped(tmp_path, temp_db):
    yaml_body = """
categories:
  scratch:
    sources:
      - "just a string"
      - kind: rss
        url: https://example.com/feed
"""
    config_path = _write_yaml(tmp_path, yaml_body)
    with closing(worker_db.connect(temp_db)) as conn:
        plan = build_source_plan(conn, config_path=config_path)

    scratch = plan.category("scratch")
    assert scratch is not None
    assert len(scratch.tasks) == 1
    assert any("must be a mapping" in w.message for w in plan.warnings)


def test_non_mapping_category_body_skipped(tmp_path, temp_db):
    yaml_body = """
categories:
  scratch:
    - kind: rss
      url: https://example.com/feed
  ok:
    sources:
      - kind: rss
        url: https://example.com/feed2
"""
    config_path = _write_yaml(tmp_path, yaml_body)
    with closing(worker_db.connect(temp_db)) as conn:
        plan = build_source_plan(conn, config_path=config_path)

    assert plan.category("scratch") is None
    assert plan.category("ok") is not None
    assert any(w.category == "scratch" and "must be a mapping" in w.message for w in plan.warnings)


# ── DB join ───────────────────────────────────────────────────────────────────


def test_db_sources_join_under_tagged_category(tmp_path, temp_db):
    """A dashboard source tagged into a YAML category appears as a task."""
    yaml_body = """
categories:
  ai_coding_tools:
    description: Coding tool moves.
"""
    config_path = _write_yaml(tmp_path, yaml_body)

    with closing(worker_db.connect(temp_db)) as conn:
        sid = _insert_source(
            conn,
            name="Anthropic blog",
            type_="rss",
            config={"url": "https://www.anthropic.com/news/rss.xml"},
        )
        _tag(conn, sid, "ai_coding_tools")

        plan = build_source_plan(conn, config_path=config_path)

    assert plan.warnings == []
    cat = plan.category("ai_coding_tools")
    assert cat is not None
    assert len(cat.tasks) == 1
    task = cat.tasks[0]
    assert task.origin == "db"
    assert task.source_id == sid
    assert task.source_name == "Anthropic blog"
    assert isinstance(task.task, RssTask)
    assert task.task.url == "https://www.anthropic.com/news/rss.xml"


def test_db_source_tagged_into_unknown_category_creates_one(tmp_path, temp_db):
    """User mid-edit: tagged a category the YAML doesn't list yet."""
    config_path = _write_yaml(tmp_path, "categories: {}\n")

    with closing(worker_db.connect(temp_db)) as conn:
        sid = _insert_source(
            conn, name="r/LocalLLaMA", type_="reddit", config={"subreddit": "LocalLLaMA"}
        )
        _tag(conn, sid, "user_only_category")

        plan = build_source_plan(conn, config_path=config_path)

    cat = plan.category("user_only_category")
    assert cat is not None
    assert len(cat.tasks) == 1
    assert isinstance(cat.tasks[0].task, RedditTask)
    assert cat.tasks[0].task.subreddit == "LocalLLaMA"


def test_inactive_db_sources_excluded(tmp_path, temp_db):
    config_path = _write_yaml(tmp_path, "categories: {scratch: {}}\n")

    with closing(worker_db.connect(temp_db)) as conn:
        active = _insert_source(
            conn, name="active", type_="rss", config={"url": "https://a.example/feed"}
        )
        inactive = _insert_source(
            conn,
            name="inactive",
            type_="rss",
            config={"url": "https://b.example/feed"},
            is_active=0,
        )
        _tag(conn, active, "scratch")
        _tag(conn, inactive, "scratch")

        plan = build_source_plan(conn, config_path=config_path)

    cat = plan.category("scratch")
    assert cat is not None
    names = [t.source_name for t in cat.tasks]
    assert names == ["active"]


def test_untagged_db_sources_excluded(tmp_path, temp_db):
    config_path = _write_yaml(tmp_path, "categories: {scratch: {}}\n")

    with closing(worker_db.connect(temp_db)) as conn:
        _insert_source(conn, name="orphan", type_="rss", config={"url": "https://a.example/feed"})
        plan = build_source_plan(conn, config_path=config_path)

    assert plan.category("scratch") is not None
    assert plan.category("scratch").tasks == []


def test_db_source_with_unsupported_type_warns(tmp_path, temp_db):
    config_path = _write_yaml(tmp_path, "categories: {scratch: {}}\n")

    with closing(worker_db.connect(temp_db)) as conn:
        sid = _insert_source(conn, name="x feed", type_="twitter_feed", config={"handle": "@foo"})
        _tag(conn, sid, "scratch")
        plan = build_source_plan(conn, config_path=config_path)

    assert plan.category("scratch").tasks == []
    assert any(
        w.origin == "db" and "twitter_feed" in w.message and str(sid) in w.message
        for w in plan.warnings
    )


def test_db_source_with_corrupt_config_warns(tmp_path, temp_db):
    config_path = _write_yaml(tmp_path, "categories: {scratch: {}}\n")

    with closing(worker_db.connect(temp_db)) as conn:
        # Write invalid JSON straight into the config blob
        cur = conn.execute(
            "INSERT INTO sources (name, type, config, is_active) VALUES (?, ?, ?, 1)",
            ("bad", "rss", "{not valid json"),
        )
        sid = int(cur.lastrowid)
        _tag(conn, sid, "scratch")
        plan = build_source_plan(conn, config_path=config_path)

    assert plan.category("scratch").tasks == []
    assert any(w.origin == "db" and "not JSON" in w.message for w in plan.warnings)


def test_db_source_missing_required_config_warns(tmp_path, temp_db):
    config_path = _write_yaml(tmp_path, "categories: {scratch: {}}\n")

    with closing(worker_db.connect(temp_db)) as conn:
        sid = _insert_source(conn, name="empty rss", type_="rss", config={})
        _tag(conn, sid, "scratch")
        plan = build_source_plan(conn, config_path=config_path)

    assert plan.category("scratch").tasks == []
    assert any(w.origin == "db" and "missing required config" in w.message for w in plan.warnings)


def test_db_hn_filter_field_translates_to_list(tmp_path, temp_db):
    """ClawFeed historically stores HN list selector as 'filter'; we accept it."""
    config_path = _write_yaml(tmp_path, "categories: {scratch: {}}\n")

    with closing(worker_db.connect(temp_db)) as conn:
        sid = _insert_source(
            conn,
            name="HN top",
            type_="hackernews",
            config={"filter": "top", "min_score": 100},
        )
        _tag(conn, sid, "scratch")
        plan = build_source_plan(conn, config_path=config_path)

    cat = plan.category("scratch")
    assert cat is not None
    assert len(cat.tasks) == 1
    assert isinstance(cat.tasks[0].task, HnTask)
    assert cat.tasks[0].task.list == "top"
    assert cat.tasks[0].task.min_score == 100


def test_db_github_trending_language_all_normalizes_to_none(tmp_path, temp_db):
    """ClawFeed UI persists 'all' for any-language; resolver normalizes."""
    config_path = _write_yaml(tmp_path, "categories: {scratch: {}}\n")

    with closing(worker_db.connect(temp_db)) as conn:
        sid_all = _insert_source(
            conn,
            name="trending all",
            type_="github_trending",
            config={"language": "all", "since": "daily"},
        )
        sid_py = _insert_source(
            conn,
            name="trending python",
            type_="github_trending",
            config={"language": "python", "since": "daily"},
        )
        _tag(conn, sid_all, "scratch")
        _tag(conn, sid_py, "scratch")
        plan = build_source_plan(conn, config_path=config_path)

    cat = plan.category("scratch")
    assert cat is not None
    languages = sorted(
        (t.task.language for t in cat.tasks if isinstance(t.task, GithubTrendingTask)),
        key=lambda v: (v is not None, v or ""),
    )
    assert languages == [None, "python"]


def test_db_source_tagged_to_multiple_categories_appears_in_each(tmp_path, temp_db):
    yaml_body = """
categories:
  ai_coding_tools: {}
  ai_research: {}
"""
    config_path = _write_yaml(tmp_path, yaml_body)

    with closing(worker_db.connect(temp_db)) as conn:
        sid = _insert_source(
            conn, name="overlap", type_="website", config={"url": "https://x.example"}
        )
        _tag(conn, sid, "ai_coding_tools")
        _tag(conn, sid, "ai_research")
        plan = build_source_plan(conn, config_path=config_path)

    coding = plan.category("ai_coding_tools")
    research = plan.category("ai_research")
    assert coding is not None and research is not None
    assert len(coding.tasks) == 1
    assert len(research.tasks) == 1
    assert isinstance(coding.tasks[0].task, WebsiteTask)
    assert isinstance(research.tasks[0].task, WebsiteTask)
    assert coding.tasks[0].source_id == sid
    assert research.tasks[0].source_id == sid


def test_yaml_and_db_merge_under_same_category(tmp_path, temp_db):
    yaml_body = """
categories:
  ai_coding_tools:
    sources:
      - kind: gdelt
        query: '"Claude Code"'
"""
    config_path = _write_yaml(tmp_path, yaml_body)

    with closing(worker_db.connect(temp_db)) as conn:
        sid = _insert_source(
            conn, name="user blog", type_="rss", config={"url": "https://blog.example/feed"}
        )
        _tag(conn, sid, "ai_coding_tools")
        plan = build_source_plan(conn, config_path=config_path)

    cat = plan.category("ai_coding_tools")
    assert cat is not None
    origins = sorted(t.origin for t in cat.tasks)
    assert origins == ["db", "yaml"]
    kinds = sorted(t.kind for t in cat.tasks)
    assert kinds == ["gdelt", "rss"]


# ── default config wiring ────────────────────────────────────────────────────


def test_default_config_path_loads_repo_config(temp_db):
    """The shipped config/intel-sources.yaml is parseable and well-formed.

    Acts as a regression test against editorial changes that break the schema.
    """
    with closing(worker_db.connect(temp_db)) as conn:
        plan = build_source_plan(conn)

    assert plan.warnings == []
    assert any(c.name == "ai_coding_tools" for c in plan.categories)
    assert any(c.name == "ai_research" for c in plan.categories)
    assert any(c.name == "startup_funding" for c in plan.categories)
    assert any(c.name == "github_traction" for c in plan.categories)
