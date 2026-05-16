"""Step 12c live smoke for the Gemini CLI final-compose path.

Runs the architecture-doc-required Tier-1 frontier composition against
the real Gemini CLI v0.36.0 + the operator's Gemini Pro subscription.
Mirrors the shape of the 2026-05-15 vMLX-only compose smoke (5 seeded
summarized clusters across 4 categories, isolated temp DB) so the two
runs are directly comparable.

NOT a CI test. Re-run manually:

- After a Gemini CLI version bump.
- After any change to ``worker/clawfeed_intel/llm/gemini_cli.py`` or
  ``worker/clawfeed_intel/pipeline/compose.py``.
- Periodically to track prose-quality drift across Gemini model
  rollouts.

Invocation::

    uv run python worker/scripts/smoke_compose_gemini.py
    uv run python worker/scripts/smoke_compose_gemini.py --keep-output

The script writes the rendered brief and the captured raw stream-json
events to ``data/smoke/`` for post-hoc inspection when ``--keep-output``
is passed (default: delete on clean exit).

Auth / cost: the Gemini CLI is signed into the operator's Gemini Pro
subscription via OAuth. One 5-cluster compose call uses well under
the Pro plan's daily quota. If a quota-exceeded error fires, the
correct response is the standard Tier-2 vMLX fallback chain, NOT
swapping to a pay-per-token API integration — see
``docs/personal-intelligence-brief-architecture.md`` Decision 4
amendment and the ``gemini_cli_environment`` memory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

# Locate the project root so the script runs no matter the CWD.
_HERE = Path(__file__).resolve()
_WORKER_ROOT = _HERE.parents[1]
_REPO_ROOT = _WORKER_ROOT.parent
sys.path.insert(0, str(_WORKER_ROOT))

from clawfeed_intel import db as worker_db  # noqa: E402
from clawfeed_intel.llm import LLMClient, load_routing  # noqa: E402
from clawfeed_intel.llm.schemas import ClusterSummaryPayload  # noqa: E402
from clawfeed_intel.paths import MIGRATIONS_DIR  # noqa: E402
from clawfeed_intel.pipeline.compose import compose_brief  # noqa: E402
from clawfeed_intel.runs import Coverage  # noqa: E402
from clawfeed_intel.sources import CategoryPlan, ProfileConfig, SourcePlan  # noqa: E402

# ── Seeded clusters (mirror the 2026-05-15 vMLX smoke shape) ─────────────────

_WINDOW_START = "2026-05-14T13:00:00+00:00"
_WINDOW_END = "2026-05-15T13:00:00+00:00"

_PLAN_CATEGORIES = [
    CategoryPlan(
        name="startup_funding",
        description="Funding rounds, new funds, seed/Series A+ announcements.",
        include_rules=["announced funding rounds", "new fund closes"],
        exclude_rules=["generic opinion posts"],
    ),
    CategoryPlan(
        name="ai_research",
        description="New research likely to matter for models, agents, tooling, or infra.",
        include_rules=["new papers", "evaluation results"],
        exclude_rules=["incremental tutorials"],
    ),
    CategoryPlan(
        name="ai_coding_tools",
        description="Product, model, pricing, release, or ecosystem moves in AI coding tools.",
        include_rules=["product launches", "model releases", "pricing changes"],
        exclude_rules=["unrelated dev tools"],
    ),
    CategoryPlan(
        name="github_traction",
        description="Repositories gaining genuine traction (velocity-backed).",
        include_rules=["repos with sustained star growth", "agent / coding repos"],
        exclude_rules=["awesome-list refreshes"],
    ),
]

# Use the architecture-doc default 24h window for the smoke; matches
# what the daily-brief orchestrator passes through to compose_brief.
_SMOKE_PROFILE = ProfileConfig(timezone="UTC", daily_window_hours=24, default_language="en")


def _seeded_clusters() -> list[dict]:
    """Five realistic summarized clusters across the four categories.

    Mirrors the 2026-05-15 vMLX-27B compose smoke seed set so direct
    quality comparison is possible. Each cluster includes the fields a
    real cluster summary payload would have after the summarizing
    stage: headline, summary, why_it_matters, entities, key_facts,
    caveats, confidence, source_urls.
    """
    return [
        {
            "key": "https://techcrunch.com/2026/05/15/anthropic-series-e/",
            "category": "startup_funding",
            "relevance_score": 0.95,
            "payload": ClusterSummaryPayload(
                headline="Anthropic raises $4B Series E led by Iconiq at $300B valuation",
                summary=(
                    "Anthropic closed a $4 billion Series E led by Iconiq Capital "
                    "with participation from existing investors Lightspeed, "
                    "Spark Capital, and Salesforce Ventures. The round values "
                    "the company at $300 billion post-money, up from $183 billion "
                    "in the March 2026 round. Proceeds will fund continued "
                    "training of Claude 4 family models plus build-out of the "
                    "agentic-development product surface."
                ),
                why_it_matters=(
                    "Cements Anthropic as one of two pure-play foundation-model "
                    "companies with sub-trillion valuations and a public "
                    "agentic-product roadmap; the round was oversubscribed and "
                    "priced ahead of expectations, indicating sustained late-stage "
                    "appetite for foundation-model investment despite recent "
                    "questions about compute-spend efficiency."
                ),
                entities=["Anthropic", "Iconiq Capital", "Lightspeed", "Salesforce Ventures"],
                key_facts=[
                    "$4B round size",
                    "$300B post-money valuation",
                    "Iconiq Capital led; Lightspeed, Spark, Salesforce Ventures participated",
                    "Oversubscribed; priced above the indicative range",
                ],
                caveats=[
                    "Per-share details and option-pool dilution not yet disclosed.",
                ],
                confidence=0.9,
                source_urls=[
                    "https://techcrunch.com/2026/05/15/anthropic-series-e/",
                    "https://www.theinformation.com/articles/anthropic-4b-iconiq",
                ],
            ),
        },
        {
            "key": "https://arxiv.org/abs/2605.04321",
            "category": "ai_research",
            "relevance_score": 0.88,
            "payload": ClusterSummaryPayload(
                headline=(
                    "Qwen3.6-Coder paper shows 41% pass@1 on SWE-Bench Verified, "
                    "matching GPT-5.3-Codex on the open-source release"
                ),
                summary=(
                    "The Qwen team released the technical report for "
                    "Qwen3.6-Coder, a 35B-A3B mixture-of-experts coding model "
                    "trained on 8T tokens of curated code plus 2T tokens of "
                    "repository-level reasoning traces. The model reports 41% "
                    "pass@1 on SWE-Bench Verified, edging GPT-5.3-Codex's 40% "
                    "from earlier this quarter, on an open-weight release with "
                    "an Apache 2.0 license."
                ),
                why_it_matters=(
                    "First open-weight coding model to credibly match a "
                    "frontier closed-source coder on the SWE-Bench Verified "
                    "benchmark — meaningful for local-first agentic-coding "
                    "deployments and for evaluating prompt-engineering "
                    "approaches without API budget constraints."
                ),
                entities=["Qwen", "Alibaba", "SWE-Bench Verified", "GPT-5.3-Codex"],
                key_facts=[
                    "35B-A3B MoE architecture (~10B active params)",
                    "8T code tokens + 2T repo-level reasoning tokens",
                    "41% pass@1 on SWE-Bench Verified",
                    "Apache 2.0 license",
                ],
                caveats=[
                    "Benchmark scores reflect Qwen's evaluation harness; "
                    "third-party reproduction is pending.",
                ],
                confidence=0.85,
                source_urls=[
                    "https://arxiv.org/abs/2605.04321",
                    "https://qwenlm.github.io/blog/qwen3.6-coder",
                ],
            ),
        },
        {
            "key": "https://github.com/anthropic-research/awesome-agents",
            "category": "github_traction",
            "relevance_score": 0.72,
            "payload": ClusterSummaryPayload(
                headline=(
                    "awesome-agents repo gained 8,400 stars over the past week, now at 42K total"
                ),
                summary=(
                    "anthropic-research/awesome-agents — a curated index of "
                    "agentic-development frameworks, evaluations, and example "
                    "deployments — added 8,400 stars in the trailing 7-day "
                    "window per the GitHub fetcher's velocity calculation. The "
                    "repo crossed 42K total stars and 3.2K forks. The week's "
                    "growth correlates with three notable agentic-coding "
                    "framework launches plus a Hacker News front-page "
                    "discussion."
                ),
                why_it_matters=(
                    "Sustained star velocity at this magnitude usually signals "
                    "developer interest converging on a topic; the index's "
                    "PRs over the same window show the ecosystem actively "
                    "shipping new entries rather than the repo coasting on "
                    "discovery."
                ),
                entities=["anthropic-research", "awesome-agents"],
                key_facts=[
                    "+8,400 stars over 7d",
                    "42K total stars, 3.2K forks",
                    "37 PRs merged in the same window",
                ],
                caveats=[],
                confidence=0.7,
                source_urls=[
                    "https://github.com/anthropic-research/awesome-agents",
                ],
            ),
        },
        {
            "key": "https://www.anthropic.com/news/claude-code-3",
            "category": "ai_coding_tools",
            "relevance_score": 0.92,
            "payload": ClusterSummaryPayload(
                headline=(
                    "Anthropic releases Claude Code 3 with persistent agent "
                    "memory and cross-session context"
                ),
                summary=(
                    "Anthropic shipped Claude Code 3, a major update to its "
                    "agentic-coding CLI. Headline features include a "
                    "persistent agent memory file (auto-managed across "
                    "sessions), built-in MCP server discovery, parallel "
                    "background sub-agents, and a new policy engine that "
                    "supersedes the previous allowed-tools / approval-mode "
                    "flags. Pricing is unchanged on the Claude Max plan; the "
                    "Pro plan gains the persistent-memory feature without a "
                    "tier change."
                ),
                why_it_matters=(
                    "Persistent agent memory closes a long-standing gap for "
                    "iterative repo work; the cross-session context handoff "
                    "matches the workflow most agentic-coding teams have been "
                    "building manually with markdown handoff files."
                ),
                entities=["Anthropic", "Claude Code", "MCP", "Claude Max"],
                key_facts=[
                    "Persistent agent memory auto-managed across sessions",
                    "Built-in MCP server discovery",
                    "Parallel background sub-agents",
                    "Policy engine replaces allowed-tools flags",
                    "No price change on existing plans",
                ],
                caveats=[
                    "Policy engine is the new public surface; the deprecated "
                    "allowed-tools flag stays through Q3 2026 for migration.",
                ],
                confidence=0.9,
                source_urls=[
                    "https://www.anthropic.com/news/claude-code-3",
                    "https://docs.anthropic.com/claude-code/3.0/migration",
                ],
            ),
        },
        {
            "key": "https://news.ycombinator.com/item?id=42189000",
            "category": "ai_coding_tools",
            "relevance_score": 0.45,
            "payload": ClusterSummaryPayload(
                headline=("HN debate: 'Has local-first coding agents already lost to the cloud?'"),
                summary=(
                    "A Hacker News front-page post (492 points, 318 "
                    "comments) prompted a debate about whether locally-run "
                    "coding agents — backed by Qwen3-Coder, DeepSeek-Coder, "
                    "and similar 30-70B open-weight models — can compete "
                    "meaningfully with cloud-served frontier models on real "
                    "repository work. The original post argues the gap is "
                    "narrower than commonly reported; top-voted replies "
                    "split between agreement and skepticism around the "
                    "specific evaluation methodology cited."
                ),
                why_it_matters=(
                    "Useful signal of developer sentiment ahead of the "
                    "Qwen3.6-Coder release in the same week; the comments "
                    "surface concrete workflows where local coding agents "
                    "are already preferred (privacy-sensitive work, slow "
                    "networks) and where they still lag (long-context repo "
                    "reasoning)."
                ),
                entities=["Hacker News", "Qwen3-Coder", "DeepSeek-Coder"],
                key_facts=[
                    "492 points, 318 comments on HN front page",
                    "Original post argues local-vs-cloud gap is narrow",
                    "Top replies dispute the specific benchmark cited",
                ],
                caveats=[
                    "Lower-confidence — opinion-driven discussion, not a primary source.",
                ],
                confidence=0.35,
                source_urls=[
                    "https://news.ycombinator.com/item?id=42189000",
                ],
            ),
        },
    ]


# ── DB plumbing (mirrors conftest._apply_migrations) ──────────────────────────


def _split_statements(sql: str) -> list[str]:
    statements: list[str] = []
    for chunk in sql.split(";"):
        body_lines = [
            line
            for line in chunk.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        body = "\n".join(body_lines).strip()
        if body:
            statements.append(body)
    return statements


def _apply_migrations(db_path: Path) -> None:
    files = sorted(p for p in MIGRATIONS_DIR.iterdir() if p.suffix == ".sql")
    if not files:
        raise RuntimeError(f"no migrations found in {MIGRATIONS_DIR}")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for path in files:
            for stmt in _split_statements(path.read_text(encoding="utf-8")):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as exc:
                    msg = str(exc).lower()
                    if "duplicate column" in msg or "already exists" in msg:
                        continue
                    raise
        conn.commit()
    finally:
        conn.close()


def _seed_run_with_clusters(conn: sqlite3.Connection) -> int:
    """Create one daily run + seed the five summarized clusters into it.

    Returns the new run_id. Each cluster gets a representative raw_item
    (so the foreign-key chain is honest) plus a kept verdict and a
    final summary payload.
    """
    run_id = worker_db.create_run(
        conn,
        run_type="daily",
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
    )
    for cluster in _seeded_clusters():
        key = cluster["key"]
        rep_id, _ = worker_db.upsert_raw_item(
            conn,
            run_id=run_id,
            source_type="rss",
            dedup_key=f"smoke-{key}",
            title=cluster["payload"].headline,
            url=key,
            canonical_url=key,
            content="",
        )
        cluster_id, _ = worker_db.create_cluster(
            conn,
            run_id=run_id,
            cluster_key=key,
            title=cluster["payload"].headline,
            raw_item_ids=[rep_id],
        )
        worker_db.update_cluster_verdict(
            conn,
            cluster_id=cluster_id,
            status="kept",
            relevance_score=cluster["relevance_score"],
            category=cluster["category"],
            event_type=None,
            filter_reason="seeded for smoke",
        )
        worker_db.create_item_summary(
            conn,
            cluster_id=cluster_id,
            model="smoke-seed",
            prompt_version="summary.v1",
            payload=cluster["payload"],
        )
        worker_db.advance_cluster_to_summarized(conn, cluster_id)
    return run_id


# ── Prechecks ─────────────────────────────────────────────────────────────────


def _precheck_gemini_cli(routing) -> None:
    """Confirm the Gemini CLI binaries exist + auth + model identifier work.

    Catches the failure modes that the operational dry-run preflight
    can't (since the dry-run doesn't actually invoke gemini). Three
    discrete checks; any failure prints a recovery hint and exits 2.
    """
    provider = routing.providers.gemini_cli
    if provider is None:
        _die(
            "config/model-routing.yaml does not declare a gemini_cli provider; "
            "Step 12c smoke requires the Tier-1 frontier path to be configured."
        )

    node_path = provider.executable_path or "/opt/homebrew/bin/node"
    script_path = provider.script_path
    if not Path(node_path).exists():
        _die(f"node binary not found: {node_path}")
    if not Path(script_path).exists():
        _die(f"gemini script not found: {script_path}")

    print(f"  [precheck] node binary: {node_path}")
    print(f"  [precheck] gemini script: {script_path}")

    version = subprocess.run(
        [node_path, script_path, "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if version.returncode != 0:
        _die(f"gemini --version failed: {version.stderr.strip() or version.stdout.strip()}")
    print(f"  [precheck] gemini --version: {version.stdout.strip()}")

    # Load the model identifier from the routing config so the precheck
    # exercises exactly the identifier the live compose call will use.
    stage = routing.resolve("final_compose")
    print(f"  [precheck] testing OAuth + model identifier: {stage.model}")
    pong = subprocess.run(
        [
            node_path,
            script_path,
            "-p",
            "",
            "-m",
            stage.model,
            "--approval-mode",
            "plan",
        ],
        input="Reply with exactly: PONG\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if pong.returncode != 0:
        stderr_tail = (pong.stderr or "").strip().splitlines()
        tail = stderr_tail[-1] if stderr_tail else "(no stderr)"
        _die(
            f"PONG probe failed (exit {pong.returncode}): {tail}\n"
            "  Recovery: re-auth interactively via `gemini` and retry. "
            "If the failure is ModelNotFoundError / 404, the routing "
            "config's `final_compose.model` doesn't match what the CLI's "
            "Pro subscription routes to — see docs/runbook.md "
            "'Gemini CLI auth expired' for the diagnosis pattern."
        )
    if "PONG" not in pong.stdout:
        _die(f"PONG probe succeeded but didn't return 'PONG': {pong.stdout.strip()[:120]!r}")
    print(f"  [precheck] PONG ok ({pong.stdout.strip()})")


def _die(msg: str) -> None:
    print(f"\n[FAIL] {msg}", file=sys.stderr)
    sys.exit(2)


# ── Validation ────────────────────────────────────────────────────────────────


def _validate_brief(markdown: str, expected_min_chars: int = 2500) -> list[str]:
    """Return a list of human-readable validation findings.

    Empty list = all checks passed. Failures are warnings (printed,
    not raised) — a degraded brief is still useful smoke evidence.
    """
    findings: list[str] = []
    if not markdown.startswith("# "):
        findings.append("brief does not start with '# ' H1 heading")
    if len(markdown) < expected_min_chars:
        findings.append(
            f"brief is short: {len(markdown)} chars < {expected_min_chars} "
            f"expected (2026-05-15 vMLX smoke produced 4635)"
        )
    if "Coverage" not in markdown:
        findings.append("brief has no Coverage section")
    expected_citations = [
        "techcrunch.com",
        "arxiv.org",
        "github.com",
        "anthropic.com",
        "news.ycombinator.com",
    ]
    missing = [d for d in expected_citations if d not in markdown]
    if missing:
        findings.append(f"missing citation domains: {missing}")
    return findings


# ── Main ──────────────────────────────────────────────────────────────────────


async def _run_smoke() -> tuple[str, float, str]:
    """Run the live compose call against the real Gemini CLI.

    Returns ``(markdown, latency_seconds, provider_tag)``.
    """
    routing = load_routing()
    print(
        f"  [config] final_compose stage: provider={routing.resolve('final_compose').provider}, "
        f"model={routing.resolve('final_compose').model}"
    )
    print()

    print("Step 1: Precheck Gemini CLI binaries + OAuth + model identifier")
    _precheck_gemini_cli(routing)
    print()

    print("Step 2: Build isolated temp DB + apply migrations + seed clusters")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "smoke.db"
        _apply_migrations(db_path)
        with closing(worker_db.connect(db_path)) as conn:
            run_id = _seed_run_with_clusters(conn)
            clusters = _seeded_clusters()
            print(
                f"  seeded run_id={run_id} with {len(clusters)} summarized clusters "
                f"across {len({c['category'] for c in clusters})} categories"
            )
            print()

            print("Step 3: Live compose call via gemini_cli_completion")
            client = LLMClient(routing, conn=conn, run_id=run_id)
            plan = SourcePlan(
                profile=_SMOKE_PROFILE,
                categories=_PLAN_CATEGORIES,
                dynamic_search=[],
                warnings=[],
            )
            coverage = Coverage(
                sources_attempted=12,
                sources_succeeded=10,
                raw_items=412,
                clusters=148,
                kept_clusters=len(clusters),
                summarized_clusters=len(clusters),
                failed_sources=["github_search:ai-coding"],
            )

            t_start = time.perf_counter()
            result = await compose_brief(
                conn,
                run_id,
                client,
                plan=plan,
                coverage=coverage,
                window_start=_WINDOW_START,
                window_end=_WINDOW_END,
                model=routing.resolve("final_compose").model,
            )
            latency_s = time.perf_counter() - t_start

            return result.markdown, latency_s, result.provider_tag


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help=(
            "Write the rendered brief + the captured raw stream-json events "
            "to data/smoke/ for post-hoc inspection (default: do not persist)."
        ),
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Step 12c: Live Gemini CLI compose smoke")
    print("=" * 72)
    print()

    markdown, latency_s, provider_tag = asyncio.run(_run_smoke())

    print()
    print("Step 4: Validate result")
    print(f"  provider_tag: {provider_tag}")
    print(f"  brief length: {len(markdown)} chars")
    print(f"  wall time: {latency_s:.1f}s")
    findings = _validate_brief(markdown)
    if findings:
        print("  [warn] validation findings:")
        for f in findings:
            print(f"    - {f}")
    else:
        print("  [ ok ] all validation checks passed")
    print()

    print("Step 5: Brief preview (first 600 chars)")
    print("-" * 72)
    print(markdown[:600])
    print("..." if len(markdown) > 600 else "")
    print("-" * 72)

    if args.keep_output:
        out_dir = _REPO_ROOT / "data" / "smoke"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        brief_path = out_dir / f"smoke-{ts}.md"
        meta_path = out_dir / f"smoke-{ts}.meta.json"
        brief_path.write_text(markdown, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "timestamp": ts,
                    "provider_tag": provider_tag,
                    "latency_seconds": round(latency_s, 2),
                    "brief_chars": len(markdown),
                    "findings": findings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print()
        print(f"  [persisted] brief: {brief_path}")
        print(f"  [persisted] meta:  {meta_path}")

    print()
    if provider_tag == "gemini_cli" and not findings:
        print("SMOKE PASSED: Tier-1 frontier compose end-to-end against live Gemini CLI.")
        return 0
    if provider_tag == "gemini_cli":
        print("SMOKE PASSED with validation warnings (see above).")
        return 0
    print(
        f"SMOKE DEGRADED: provider_tag={provider_tag!r} (expected 'gemini_cli'). "
        "Inspect the LLM audit row for the Tier-1 failure that triggered fallback."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
