# Personal Intelligence Brief System Architecture

Status: final architecture recommendation for implementation planning  
Project base: `/Users/merlin/clawfeed`  
Primary runtime target: Mac Studio M4 Max, 128 GB unified memory, macOS arm64  
Date: 2026-05-04

## Executive Recommendation

Build the Personal Intelligence Brief System **inside and alongside ClawFeed**, using ClawFeed as the product shell, source registry, user-facing dashboard, feed output, auth surface, and finished-brief storage. Add a new collector and intelligence worker as a first-class module in this repository.

The system should **not** assume ClawFeed already has a scraping or LLM pipeline. It does not. ClawFeed currently provides source management, digest storage, marks, packs, subscriptions, auth, and a UI. The missing ingestion, deduplication, local inference, summarization, composition, scheduling, and topical search layers must be built.

The final architecture is therefore:

```text
ClawFeed UI/API/SQLite
  + new collector/intelligence worker
  + vMLX primary local inference
  + OpenClaw/Merlin for cron orchestration and final frontier composition
```

This avoids a greenfield rebuild, preserves the useful ClawFeed surface area, and keeps the hard intelligence pipeline testable and versioned in the application repo.

## Key Decisions

### 1. Use ClawFeed, But Do Not Treat It As The Existing Pipeline

Decision: **Use ClawFeed as the foundation, not as a completed ingestion engine.**

ClawFeed already gives us:

- SQLite-backed `digests`, `sources`, `user_subscriptions`, `source_packs`, `marks`, `users`, `sessions`, and `feedback`.
- A REST API using native Node `http`.
- A vanilla JavaScript dashboard.
- Source CRUD and source type detection.
- Public JSON/RSS feed output for finished digests.
- A simple internal write path: `POST /api/digests`.

ClawFeed does not currently provide:

- Real source collectors.
- `raw_items`.
- Content extraction.
- Event/item deduplication beyond simple source/bookmark uniqueness.
- Local or frontier LLM clients.
- Summarization or composition logic.
- A scheduler.
- Topical search.

The implementation should extend ClawFeed, not wrap imaginary existing scraping code.

### 2. vMLX Is The Primary Inference Runtime

Decision: **All high-volume LLM work goes through vMLX.**

The vMLX OpenAI-compatible endpoint at `http://127.0.0.1:8080` is the primary inference layer.

The originally available local models were:

- `mlx-community/Qwen3-8B-4bit`
- `mlx-community/Qwen3.5-27B-4bit`
- `mlx-community/Qwen3-30B-A3B-4bit`

As of May 4, 2026, these are no longer the best local defaults for this hardware. The Mac Studio M4 Max has enough unified memory to run a stronger local flagship. The target local model stack should be upgraded to Qwen3.5-family MLX models, with Qwen3.5-122B-A10B-4bit as the strongest practical local model for the brief pipeline.

Model review as of 2026-05-04:

| Candidate | Local fit on 128 GB Mac Studio | Use-case assessment |
|---|---:|---|
| `Qwen/Qwen3.6-35B-A3B` or stable MLX/vMLX equivalent | Fits; official FP8 artifact is about 37.5 GB | Newest Qwen open-weight direction as of this review. Stronger for agentic coding, repository-level reasoning, and "Claw"-style tasks than Qwen3.5-35B. Use as planner/agentic-routing candidate once a stable vMLX-serving path is available. |
| `mlx-community/Qwen3.5-122B-A10B-4bit` | Fits; MLX artifact is about 69.6 GB | Best practical default local flagship. Strong instruction following, long context, reasoning, coding, and summarization benchmarks; 122B total / 10B active MoE keeps it feasible locally. |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | Fits easily; about 20.4 GB | Good high-throughput model for source planning and broad relevance passes. Stronger and newer than the older Qwen3-30B-A3B route. |
| `mlx-community/Qwen3.5-27B-4bit` | Fits easily; about 16.1 GB | Useful fallback and small enough for experimentation, but no longer the best local summary model. |
| `mlx-community/Mistral-Small-4-119B-2603-4bit` | Fits; about 67.8 GB | Serious A/B challenger, especially for structured output and agentic behavior. Keep as an evaluation candidate, not the default until it beats Qwen3.5-122B on saved brief runs. |
| DeepSeek-V4-Flash local MLX | Does not fit cleanly; MLX 4-bit ports report about 149-160 GB and about 160 GB peak RAM | Strong model family, but outside this machine's practical no-swap envelope. Do not make it a local dependency for Phase 1. |
| DeepSeek-V4-Pro | Does not fit; official weights are about 865 GB | Not a local option for this machine. |
| Kimi K2.6 | Does not fit in a practical local configuration; official model is around 1T total params | Excellent agentic/coding model, but not appropriate as the local daily pipeline dependency on this hardware. |
| GLM-5.1 | Does not fit; official files are around 1.51 TB | Strong agentic model, but not viable locally on 128 GB unified memory. |

Updated model roles:

| Stage | Primary model | Reason |
|---|---|---|
| Query/source planning | `Qwen/Qwen3.6-35B-A3B` if vMLX-stable, otherwise `mlx-community/Qwen3.5-35B-A3B-4bit` | Small active-parameter MoE, much stronger than the old 8B planner, well suited to routing, tool-style decisions, and category planning |
| Batch relevance filtering | `mlx-community/Qwen3.5-122B-A10B-4bit` | Relevance mistakes hurt the final brief more than latency; use the strongest practical local model for keep/reject/category judgment |
| Ambiguous relevance arbitration | `mlx-community/Qwen3.5-122B-A10B-4bit` | Second-pass review for borderline rejects, contradiction flags, or high-value categories |
| Per-item or per-cluster summaries | `mlx-community/Qwen3.5-122B-A10B-4bit` | Best practical local model for factual condensation, caveat preservation, and citation-aware summaries |
| Final prose composition | OpenClaw gateway to `gpt-5.3-codex` | One or two calls where prose quality matters |
| Final local fallback | `mlx-community/Qwen3.5-122B-A10B-4bit` | Strongest local fallback if OpenClaw/frontier composition is unavailable |

Ollama remains a fallback, not the main path. It may be used for embeddings via `nomic-embed-text:latest` if vector search becomes useful, but the core LLM client should prefer vMLX.

### 3. Frontier Model Use Is Strictly Bounded

Decision: **Frontier calls are allowed only after the local pipeline has already filtered, clustered, and summarized.**

The system should never send hundreds of raw fetched items to the frontier model. Frontier use is reserved for final brief composition:

- Daily brief final composition.
- Topical search final composition.
- Optional second-pass rewrite if the first final brief fails quality checks.

Everything else should be local.

**Auth and cost model (as amended 2026-05-15):** the current Gemini CLI integration uses an operator-managed Gemini Pro subscription via OAuth — *not* the pay-per-token Gemini API. There is no API key in the worker, no per-token cost accounting, and no per-call cost surface in the metadata. Quota is the Pro plan's subscription ceiling, which comfortably covers one daily-brief composition plus several topical-search compositions per day. See "Decision 4 amendment" and the "LLM Architecture → Model Routing" cost-model note for the full mechanics.

### 4. OpenClaw Integrates As Orchestrator, Not Product Database

Decision: **Use OpenClaw/Merlin for cron, final composition access, and conversational integration. Keep product state in ClawFeed SQLite.**

OpenClaw is already the right control plane for scheduled agent tasks and final frontier composition. But the application pipeline should live in the ClawFeed repo, not as hidden Merlin workspace scripts.

OpenClaw should:

- Trigger the daily run via cron.
- Provide access to the frontier composition model.
- Optionally query finished briefs conversationally later.
- Optionally notify via Telegram after brief completion.

ClawFeed should own:

- Source configuration and subscriptions.
- Raw item cache.
- Pipeline runs and jobs.
- Summaries.
- Finished brief documents.
- Dashboard access.

#### Amendment, 2026-05-15: OpenClaw drops out of the brief-pipeline scope

The two pipeline touchpoints originally assigned to OpenClaw — daily-run scheduling and frontier final composition — have both been reassigned after investigation against the actual installed OpenClaw build and the available alternatives:

- **Daily-run scheduling** moves to macOS `launchd` (`~/Library/LaunchAgents/local.clawfeed.daily-brief.plist`, fired at `06:15` local). OpenClaw cron's real surface schedules `agentTurn` / `systemEvent` payloads — i.e. scheduled LLM calls — not arbitrary shell exec. The daily brief is a deterministic shell command, so `launchd` is the right macOS primitive. See "Scheduling And Deployment → Daily Schedule" below for the updated example.
- **Frontier final composition** moves to **Gemini CLI** (`gemini-3-pro-preview` — see Step 12c → 12d evolution below) invoked as a subprocess. The non-negotiable is that the final document is composed by a frontier-class model; the original mechanism (OpenClaw → `gpt-5.3-codex`) is blocked on the gateway wire protocol being undocumented for non-Node clients. The Gemini CLI gives the same model-class with a stable subprocess contract and the operator-managed OAuth auth that ships with the CLI. **Critically: the integration uses the operator's Gemini Pro subscription via the CLI's OAuth flow — not the pay-per-token Gemini API.** The worker holds no API key, has no per-token cost accounting, and is bounded only by the Pro plan's subscription quota (which comfortably covers a single composition call per day with substantial headroom). See "LLM Architecture → Model Routing" below for the updated stage config.

  *Step 12c correction (2026-05-15):* the originally-named `gemini-3-pro` returns 404 against the live Gemini CLI v0.36.0 + operator's Pro account. The CLI's reachable Pro-subscription frontier SKUs are `gemini-2.5-pro` and `gemini-3-pro-preview` (the latter aliases to `gemini-3.1-pro-preview` server-side).

  *Step 12d upgrade (2026-05-15):* swapped from `gemini-2.5-pro` to `gemini-3-pro-preview` after confirming reachability + characterizing quality on the same 5-cluster smoke seed. **`gemini-3-pro-preview` is Google's SKU name for the production default Gemini 3 Pro on this subscription — the `-preview` suffix is naming convention, not an early-access tier.** Gemini 3 produced a 7083-char brief (vs 6002 on 2.5 / 4635 on vMLX-27B) with notably better fact density, smarter Watchlist routing, and cross-cluster thesis synthesis. `gemini-2.5-pro` remains the documented fallback if Google rotates the Gemini 3 SKU name.

This is a **pipeline-scope amendment only**. The original ClawFeed product surface (dashboard, user-facing agent features) remains free to integrate with OpenClaw as it always did. A future conversational-query phase against finished briefs may revisit OpenClaw as the agent runtime; that's a separate UI-layer decision and does not affect the daily-brief pipeline.

### 5. Twitter/X Is Out Of Scope For v1

Decision: **Do not include Twitter/X in v1.**

Twitter/X is fragile, requires paid API access or session-cookie automation, and is likely to dominate maintenance cost. The v1 source plan should rely on RSS, arXiv, HN, Reddit, GitHub, GDELT, SEC EDGAR, and selected websites.

The pipeline should leave room for a future `twitter_*` source adapter, but the first usable system should not depend on it.

## Target Use Cases

## Function 1: Daily Brief

The daily brief runs once each morning and covers the last 24 hours across curated categories. It must be comprehensive inside the configured categories, not a top-five teaser.

Example categories:

- Startups that raised funding.
- Major AI research updates.
- Model releases.
- AI product and company moves.
- AI coding tool updates, including Codex, Claude Code, Cursor, Continue, Windsurf, Zed, JetBrains AI, Devin-like tools, and adjacent developer-agent tools.
- GitHub repositories gaining traction.
- Hacker News or Reddit discussions that signal meaningful developer interest.
- Additional categories added over time.

Expected behavior:

1. Fetch from all configured daily sources.
2. Normalize raw results into a common item schema.
3. Store raw items.
4. Deduplicate by URL, content fingerprint, and event similarity.
5. Filter for category relevance using local vMLX.
6. Summarize kept clusters locally.
7. Compose the final document with OpenClaw/frontier.
8. Store final markdown in ClawFeed `digests`.
9. Show it in the ClawFeed dashboard.

The brief should include all relevant items that pass the configured criteria. Ranking affects order and section prominence, not inclusion.

## Function 2: Topical Search

Topical search is an on-demand query from the dashboard.

Example query: `Khosla Ventures`

Expected behavior:

1. User submits query in ClawFeed UI.
2. ClawFeed API creates a search run/job.
3. Worker asks the local source planner which sources and query variants to use.
4. Worker queries local raw cache plus dynamic external sources.
5. Items flow through the same normalize, dedup, filter, summarize, compose path.
6. Final topical brief is stored in the same backend and viewed through the same UI.

Topical search should not be modeled as a permanent user subscription. It is a run type with dynamic source selection and run-scoped results.

## Component Architecture

```text
+-------------------------------------------------------------+
|                         ClawFeed UI                         |
| Daily brief view / Source manager / Topical search / History |
+------------------------------+------------------------------+
                               | REST
+------------------------------v------------------------------+
|                     ClawFeed API Server                      |
| Existing Node server + new intel endpoints                   |
| /api/digests / /api/sources / /api/intel/search / /api/runs  |
+------------------------------+------------------------------+
                               | SQLite WAL
+------------------------------v------------------------------+
|                         SQLite DB                           |
| Existing ClawFeed tables + new intelligence pipeline tables  |
+------------------------------+------------------------------+
                               |
+------------------------------v------------------------------+
|                 Collector + Intelligence Worker              |
| Fetch / normalize / dedup / filter / summarize / compose     |
+---------------+-------------------------------+-------------+
                |                               |
+---------------v--------------+   +------------v-------------+
| vMLX OpenAI-compatible LLMs  |   | OpenClaw/Merlin Gateway  |
| local planning/filter/summary|   | final gpt-5.3-codex prose|
+------------------------------+   +--------------------------+
```

## Repository Layout Recommendation

Keep the existing ClawFeed app, but add a clean intelligence subsystem.

```text
/Users/merlin/clawfeed
|-- src/
|   |-- server.mjs
|   |-- db.mjs
|   `-- intel/                    # new Node API helpers if needed
|-- worker/                       # new Python pipeline package
|   |-- pyproject.toml
|   `-- clawfeed_intel/
|       |-- cli.py
|       |-- config.py
|       |-- db.py
|       |-- models.py
|       |-- llm/
|       |   |-- vmlx.py
|       |   |-- openclaw.py
|       |   `-- schemas.py
|       |-- fetchers/
|       |   |-- rss.py
|       |   |-- arxiv.py
|       |   |-- hn.py
|       |   |-- reddit.py
|       |   |-- github.py
|       |   |-- gdelt.py
|       |   |-- sec.py
|       |   `-- website.py
|       |-- pipeline/
|       |   |-- collect.py
|       |   |-- dedup.py
|       |   |-- filter.py
|       |   |-- summarize.py
|       |   |-- compose.py
|       |   `-- publish.py
|       `-- search/
|           |-- planner.py
|           `-- executor.py
|-- config/
|   |-- intel-sources.yaml
|   |-- relevance-rules.yaml
|   `-- model-routing.yaml
|-- migrations/
|   `-- 010_intel_pipeline.sql
|-- prompts/
|   |-- relevance.md
|   |-- cluster-summary.md
|   |-- daily-compose.md
|   `-- topical-compose.md
`-- docs/
    `-- personal-intelligence-brief-architecture.md
```

Python is recommended for the worker because the required ecosystem is stronger for feed parsing, article extraction, async HTTP, data normalization, and structured source integrations.

Node remains the ClawFeed API/UI runtime.

## Technology Stack

### Existing ClawFeed Runtime

- Node.js ESM.
- Native `http` server.
- `better-sqlite3`.
- Vanilla JS frontend.
- SQLite WAL mode.

Keep this unchanged for v1 unless the new API endpoints become painful inside the existing server file.

### Worker Runtime

Recommended:

- Python 3.12+.
- `uv` for environment and command execution.
- `httpx` for HTTP, using `AsyncClient`, connection pooling, timeouts, and per-domain limits.
- `feedparser` for RSS/Atom.
- `trafilatura` for article extraction.
- `beautifulsoup4` or `selectolax` only where source-specific HTML parsing is needed.
- `pydantic` for normalized item and LLM output schemas.
- `sqlite-utils` optional, but direct `sqlite3`/`apsw` is enough.
- `tenacity` for retries with exponential backoff.
- `python-frontmatter` optional for prompt metadata.
- `PyYAML` or `ruamel.yaml` for configuration.

Avoid adding a heavy queue system in v1. SQLite is enough for a single-machine personal intelligence system.

## Source Configuration

Use both ClawFeed DB sources and repo-controlled YAML.

### ClawFeed `sources`

Use existing `sources` for user-visible configured sources:

- RSS feeds.
- Websites.
- HN source definitions.
- Reddit communities.
- GitHub trending/search definitions.
- Public/private source packs.

This preserves the existing UI and source sharing model.

### `config/intel-sources.yaml`

Use YAML for editorial categories and non-user-facing structured sources.

Example shape:

```yaml
profile:
  timezone: America/Los_Angeles
  daily_window_hours: 24
  default_language: en

categories:
  startup_funding:
    description: Funding rounds, new funds, seed/Series A+ announcements, notable acquisitions.
    include:
      - announced funding rounds
      - new fund closes
      - major portfolio company financings
    exclude:
      - generic opinion posts
      - old funding rounds recirculated without new information
    sources:
      - type: gdelt
        query: '(startup OR company) (raised OR funding OR seed OR "series a" OR "series b")'
      - type: rss
        url: https://techcrunch.com/category/startups/feed/
      - type: sec_edgar
        forms: ["D", "D/A"]

  ai_research:
    description: New research likely to matter for models, agents, tooling, or infra.
    sources:
      - type: arxiv
        categories: ["cs.AI", "cs.LG", "cs.CL", "stat.ML"]
      - type: rss
        url: https://huggingface.co/papers/rss

  ai_coding_tools:
    description: Product, model, pricing, release, or ecosystem moves in AI coding tools.
    sources:
      - type: gdelt
        query: '("Claude Code" OR Codex OR Cursor OR Windsurf OR Continue.dev OR "AI coding")'
      - type: github_search
        query: 'topic:ai-coding OR topic:agent'

dynamic_search:
  enabled_sources:
    - raw_cache
    - gdelt
    - hn_algolia
    - reddit
    - github
    - sec_edgar
```

The YAML defines editorial intent. The DB stores user-visible source records and fetched state.

## Source Fetchers

All fetchers must emit a normalized `RawItem`.

```text
RawItem
  source_type
  source_id nullable
  run_id
  title
  url
  canonical_url
  author
  published_at
  fetched_at
  text
  excerpt
  metadata_json
  raw_json
  dedup_key
```

### RSS / Atom

Use `feedparser`, then fetch article pages where needed and extract with `trafilatura`.

Fields:

- Feed title.
- Entry title.
- Link.
- Published/updated time.
- Author.
- Summary.
- Extracted full text when available.

### arXiv

Use the official arXiv API or RSS feeds.

Use cases:

- Daily AI research discovery.
- Topical search for papers mentioning a query.

Store:

- arXiv id.
- Title.
- Authors.
- Abstract.
- Categories.
- Published/updated.
- PDF/abstract URLs.

### Hacker News

Use two paths:

- Official Firebase API for top/new/best/show/ask lists.
- Algolia HN Search API for topical search and keyword queries.

Store:

- HN object id.
- Title.
- URL.
- Author.
- Points.
- Comment count.
- HN discussion URL.
- Creation timestamp.

### Reddit

Use public JSON/API listing endpoints with a clear user agent, conservative request rates, and backoff.

V1 should use a small configured set of subreddits. Broad Reddit search is noisy and can be rate-limited. Treat Reddit as supplementary signal, not guaranteed comprehensive coverage.

Candidate communities:

- `r/MachineLearning`
- `r/LocalLLaMA`
- `r/OpenAI`
- `r/ClaudeAI`
- `r/programming`
- `r/startups`
- targeted communities for coding tools as needed

### GitHub

Use:

- GitHub Trending HTML scrape for daily trending discovery.
- GitHub REST Search API for topical searches and repo discovery.
- Optional GitHub token for better rate limits.

Signals:

- Stars.
- Forks.
- Recent push time.
- Creation date.
- Topics.
- Description.
- README excerpt when needed.
- Star velocity if we store snapshots.

For "repositories gaining traction," GitHub Trending is an imperfect but useful v1 source. For better precision, store daily snapshots and compute deltas.

### GDELT

Use GDELT DOC 2.0 API for broad news discovery and topical search.

Strengths:

- Broad global media coverage.
- Recent time-window search.
- JSON output.
- Good for company/person/topic mentions.

Limitations:

- Not a full general web search engine.
- Quality varies by outlet.
- Can return duplicative syndication.

Use GDELT as the open-news backbone, with dedup and local relevance filtering doing the cleanup.

### SEC EDGAR

Use official SEC endpoints with a declared User-Agent and strict rate limiting. SEC guidance currently requires respectful automation and gives a fair-access max request rate of 10 requests/second.

Use cases:

- Form D / D-A for financing signals.
- Company filings.
- Fund/company activity in topical search.

For venture firms, SEC will not catch everything, but it adds structured evidence.

### Websites

Website sources should follow this fallback:

1. Fetch page.
2. Discover RSS/Atom link.
3. If feed exists, store/update source as RSS-backed.
4. If no feed exists, extract title, metadata, and page text with `trafilatura`.
5. Only crawl configured pages, not arbitrary site-wide crawls in v1.

## Deduplication Strategy

Dedup must happen at multiple levels.

### Level 1: Canonical URL

Normalize:

- Scheme/host case.
- `www.` where appropriate.
- Trailing slash.
- Fragments.
- Tracking params: `utm_*`, `fbclid`, `gclid`, `mc_cid`, etc.
- Common AMP/mobile canonical links when detectable.

Dedup key:

```text
url_sha256(canonical_url)
```

### Level 2: Content Fingerprint

Compute normalized text/title hash:

```text
sha256(normalize(title) + "\n" + normalize(first_4000_chars(text)))
```

Use when URL differs but content is syndicated.

### Level 3: Event Clustering

Cluster items that describe the same event.

Initial implementation:

- Same or highly similar title.
- Overlapping named entities.
- Same date window.
- Similar numeric facts, such as funding amount.
- Local LLM tie-breaker for ambiguous pairs.

Future implementation:

- Embeddings via `nomic-embed-text` or vMLX embedding model if available.
- MinHash/SimHash for scalable near-duplicate detection.

The unit passed to summarization should be `ItemCluster`, not raw article.

## LLM Architecture

## LLM Client Requirements

The LLM client must support:

- OpenAI-compatible `/v1/chat/completions`.
- Configurable base URLs.
- Per-stage model routing.
- Timeouts.
- Retries.
- JSON schema validation.
- Response repair only as a bounded fallback.
- Full prompt/version/model logging.
- Token and latency accounting.
- Failure classification.

Do not scatter direct LLM HTTP calls throughout the pipeline.

## Model Routing

`config/model-routing.yaml` (amended 2026-05-15 — `final_compose` moves from OpenClaw to Gemini CLI; OpenClaw provider is removed from the brief pipeline):

```yaml
providers:
  vmlx:
    base_url: http://127.0.0.1:8080/v1
    api_key_env: VMLX_API_KEY
    required: true
  gemini_cli:
    # Subprocess transport — no HTTP. The worker invokes the Gemini CLI
    # directly. `executable_path` lets us bypass a broken PATH-resolved
    # node binary (the local node@22 is linked against an absent simdjson
    # version; gemini's shebang resolves via PATH so we hardcode the
    # working node here).
    executable_path: /opt/homebrew/bin/node
    script_path: /opt/homebrew/bin/gemini
    approval_mode: plan  # read-only — model cannot take actions
    output_format: stream-json
    idle_timeout_seconds: 60   # per-line gap that triggers stall detection
    hard_timeout_seconds: 300  # total wall-clock cap

stages:
  source_planning:
    provider: vmlx
    model: Qwen/Qwen3.6-35B-A3B
    timeout_seconds: 90
    fallback:
      provider: vmlx
      model: mlx-community/Qwen3.5-35B-A3B-4bit
  relevance_filter:
    provider: vmlx
    model: mlx-community/Qwen3.5-122B-A10B-4bit
    timeout_seconds: 240
    batch_size: 12
    ambiguity_review:
      enabled: true
      model: mlx-community/Qwen3.5-122B-A10B-4bit
  cluster_summary:
    provider: vmlx
    model: mlx-community/Qwen3.5-122B-A10B-4bit
    timeout_seconds: 300
  final_compose:
    provider: gemini_cli
    model: gemini-3-pro-preview   # was gemini-3-pro (12c found 404) → gemini-2.5-pro (12c stable) → gemini-3-pro-preview (12d upgrade, aliases to gemini-3.1-pro-preview)
    timeout_seconds: 300
    retries: 1               # one retry on stall/timeout/non-zero exit
    retry_backoff_seconds: 10
    fallback:
      provider: vmlx
      model: mlx-community/Qwen3.5-122B-A10B-4bit
  model_evaluation:
    challengers:
      - mlx-community/Mistral-Small-4-119B-2603-4bit
      - Qwen/Qwen3.6-35B-A3B
```

If the Gemini CLI call fails after its single retry, the system falls back to the strongest local vMLX model and stamps `composition_provider: vmlx_fallback`. If that also fails, the deterministic `render_fallback_brief` path emits a structured Markdown brief built directly from the cluster summaries (no LLM call) and stamps `composition_provider: local_stub_failed`. The brief always publishes; the metadata always tells the truth about which tier produced it.

**Cost / auth model for the `gemini_cli` provider:** the CLI is signed into the operator's Gemini Pro subscription via OAuth at install time (`gemini` interactive command); the CLI manages the refresh-token lifecycle internally. The worker holds no API key, never makes a `generativelanguage.googleapis.com` HTTP call directly, and is not billed per-token. The only relevant quota is the Pro plan's subscription ceiling, which comfortably covers one composition call per day with substantial headroom. This is materially different from the originally-considered OpenClaw → `gpt-5.3-codex` path (which would have routed through OpenClaw's gateway against frontier-API credits) and from a direct Gemini-API integration (which would have required key management + per-token cost tracking). The CLI-subprocess shape is the load-bearing reason this integration is operationally simple.

## Prompt Responsibilities

### Source Planner

Input:

- Query or category.
- Source registry.
- Allowed source types.
- Time window.

Output JSON:

- Selected sources.
- Query variants.
- Required/excluded terms.
- Expected evidence types.
- Priority order.

### Relevance Filter

Input:

- Batch of raw items or clusters.
- Category rules.
- User preference rules.

Output JSON:

- `keep: boolean`
- `category`
- `score`
- `event_type`
- `reason`
- `entities`
- `evidence_urls`
- `uncertainty`

This stage must be allowed to keep many items. It is not a top-N selector.

### Cluster Summary

Input:

- One cluster with all supporting items.
- Extracted text snippets.
- Metadata.

Output JSON:

- `headline`
- `summary`
- `why_it_matters`
- `entities`
- `key_facts`
- `caveats`
- `source_urls`
- `confidence`

### Final Compose

Input:

- Structured cluster summaries.
- Category rules.
- Coverage report.
- Run metadata.

Output:

- Markdown brief.

Responsibilities:

- Organize.
- Prioritize.
- Write clean prose.
- Preserve citations/links.
- Include all kept items.
- Add coverage footer.

The final composer must not invent facts or add uncited claims.

## Data Model

Add one migration, e.g. `migrations/010_intel_pipeline.sql`.

### `intel_runs`

Represents a daily brief or topical search run.

```sql
CREATE TABLE IF NOT EXISTS intel_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_type TEXT NOT NULL CHECK(run_type IN ('daily', 'topic')),
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK(status IN ('pending', 'fetching', 'filtering', 'summarizing', 'composing', 'published', 'failed', 'cancelled')),
  query TEXT,
  window_start TEXT NOT NULL,
  window_end TEXT NOT NULL,
  config_hash TEXT,
  prompt_version TEXT,
  model_config_hash TEXT,
  digest_id INTEGER REFERENCES digests(id),
  error TEXT,
  metadata TEXT DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  started_at TEXT,
  finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_intel_runs_type_created ON intel_runs(run_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_intel_runs_status ON intel_runs(status);
```

### `intel_jobs`

Queue table for asynchronous work.

```sql
CREATE TABLE IF NOT EXISTS intel_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES intel_runs(id) ON DELETE CASCADE,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK(status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')),
  priority INTEGER NOT NULL DEFAULT 100,
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  run_after TEXT NOT NULL DEFAULT (datetime('now')),
  locked_at TEXT,
  locked_by TEXT,
  error TEXT,
  payload TEXT DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_intel_jobs_claim ON intel_jobs(status, run_after, priority, id);
```

SQLite does not support `FOR UPDATE SKIP LOCKED`. Use `BEGIN IMMEDIATE` and an atomic claim update:

```sql
UPDATE intel_jobs
SET status = 'running',
    locked_at = datetime('now'),
    locked_by = ?
WHERE id = (
  SELECT id FROM intel_jobs
  WHERE status = 'pending'
    AND run_after <= datetime('now')
  ORDER BY priority ASC, id ASC
  LIMIT 1
)
RETURNING *;
```

### `raw_items`

Use the simple name `raw_items` rather than `intel_raw_items` because this table is a natural ClawFeed primitive and already appears in ClawFeed docs.

```sql
CREATE TABLE IF NOT EXISTS raw_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
  run_id INTEGER REFERENCES intel_runs(id) ON DELETE SET NULL,
  source_type TEXT NOT NULL,
  source_name TEXT,
  title TEXT NOT NULL DEFAULT '',
  url TEXT NOT NULL DEFAULT '',
  canonical_url TEXT NOT NULL DEFAULT '',
  author TEXT DEFAULT '',
  content TEXT NOT NULL DEFAULT '',
  excerpt TEXT DEFAULT '',
  published_at TEXT,
  fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
  dedup_key TEXT NOT NULL,
  content_hash TEXT,
  metadata TEXT DEFAULT '{}',
  raw_payload TEXT DEFAULT '{}',
  UNIQUE(source_type, dedup_key)
);
CREATE INDEX IF NOT EXISTS idx_raw_items_source_fetched ON raw_items(source_id, fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_items_published ON raw_items(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_items_canonical_url ON raw_items(canonical_url);
```

The uniqueness constraint should be tuned during implementation. For topical search, the same external item may appear in multiple runs. Use join tables for run membership if needed:

```sql
CREATE TABLE IF NOT EXISTS run_raw_items (
  run_id INTEGER NOT NULL REFERENCES intel_runs(id) ON DELETE CASCADE,
  raw_item_id INTEGER NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
  PRIMARY KEY (run_id, raw_item_id)
);
```

### `item_clusters`

```sql
CREATE TABLE IF NOT EXISTS item_clusters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES intel_runs(id) ON DELETE CASCADE,
  cluster_key TEXT NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK(status IN ('pending', 'filtered_out', 'kept', 'summarized')),
  category TEXT,
  event_type TEXT,
  relevance_score REAL,
  filter_reason TEXT,
  metadata TEXT DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(run_id, cluster_key)
);
CREATE INDEX IF NOT EXISTS idx_item_clusters_run_status ON item_clusters(run_id, status);
```

### `cluster_items`

```sql
CREATE TABLE IF NOT EXISTS cluster_items (
  cluster_id INTEGER NOT NULL REFERENCES item_clusters(id) ON DELETE CASCADE,
  raw_item_id INTEGER NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
  PRIMARY KEY (cluster_id, raw_item_id)
);
```

### `item_summaries`

```sql
CREATE TABLE IF NOT EXISTS item_summaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cluster_id INTEGER NOT NULL REFERENCES item_clusters(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  headline TEXT NOT NULL,
  summary TEXT NOT NULL,
  why_it_matters TEXT DEFAULT '',
  entities TEXT DEFAULT '[]',
  key_facts TEXT DEFAULT '[]',
  caveats TEXT DEFAULT '[]',
  confidence REAL,
  source_urls TEXT DEFAULT '[]',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_item_summaries_cluster ON item_summaries(cluster_id);
```

### `llm_calls`

```sql
CREATE TABLE IF NOT EXISTS llm_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER REFERENCES intel_runs(id) ON DELETE SET NULL,
  stage TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_version TEXT,
  input_hash TEXT,
  output_hash TEXT,
  latency_ms INTEGER,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  status TEXT NOT NULL CHECK(status IN ('succeeded', 'failed')),
  error TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run_stage ON llm_calls(run_id, stage);
```

### `source_fetch_state`

Use this instead of overloading `sources.last_fetched_at` for every pipeline mode.

```sql
CREATE TABLE IF NOT EXISTS source_fetch_state (
  source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  fetcher TEXT NOT NULL,
  last_success_at TEXT,
  last_attempt_at TEXT,
  last_error TEXT,
  consecutive_errors INTEGER NOT NULL DEFAULT 0,
  cursor TEXT,
  metadata TEXT DEFAULT '{}',
  PRIMARY KEY (source_id, fetcher)
);
```

## Daily Brief Flow

```text
OpenClaw cron
  -> clawfeed-intel run daily --window 24h
  -> create intel_runs row
  -> resolve source plan from YAML + active ClawFeed sources
  -> fetch sources
  -> store raw_items
  -> cluster/dedup
  -> local relevance filtering via vMLX
  -> local cluster summaries via vMLX
  -> final composition via OpenClaw/frontier
  -> POST/insert into digests(type='daily', content=markdown, metadata=json)
  -> update intel_runs.digest_id and status='published'
```

Daily brief `digests.metadata` example:

```json
{
  "brief_kind": "daily",
  "run_id": 42,
  "window_start": "2026-05-03T06:00:00-07:00",
  "window_end": "2026-05-04T06:00:00-07:00",
  "composition_provider": "openclaw",
  "composition_model": "gpt-5.3-codex",
  "local_models": {
    "planner": "Qwen/Qwen3.6-35B-A3B",
    "filter": "mlx-community/Qwen3.5-122B-A10B-4bit",
    "summary": "mlx-community/Qwen3.5-122B-A10B-4bit"
  },
  "coverage": {
    "sources_attempted": 32,
    "sources_succeeded": 29,
    "raw_items": 418,
    "clusters": 151,
    "kept_clusters": 43,
    "failed_sources": ["example-source-id"]
  }
}
```

## Topical Search Flow

Dashboard request:

```http
POST /api/intel/search
Content-Type: application/json

{
  "query": "Khosla Ventures",
  "window_days": 30
}
```

API behavior:

1. Validate query.
2. Create `intel_runs(run_type='topic', query='Khosla Ventures')`.
3. Create `intel_jobs(job_type='topic_search')`.
4. Return `run_id`.

Worker behavior:

```text
claim topic_search job
  -> source planner via Qwen/Qwen3.6-35B-A3B, or Qwen3.5-35B-A3B fallback
  -> selected sources:
      raw_cache
      gdelt
      hn_algolia
      reddit
      github
      sec_edgar
      configured RSS/company-news sources
  -> generate query variants:
      "Khosla Ventures"
      "Vinod Khosla"
      "Khosla led round"
      "Khosla Ventures portfolio"
      "Khosla Ventures Form D"
  -> execute fetches
  -> normalize raw_items
  -> cluster by event
  -> relevance filter:
      keep investments, fund activity, partnerships, portfolio moves,
      material interviews, regulatory filings
      reject incidental mentions
  -> summarize clusters
  -> compose topical brief
  -> store in digests with metadata.brief_kind='topic'
```

Topical search UI:

- Search input in the dashboard header or a dedicated Search tab.
- Run status view: pending/fetching/summarizing/composing/published/failed.
- Search history list.
- Final topical brief viewer using the same digest rendering component.

## API Additions

Add minimal endpoints to the existing Node server:

```text
POST /api/intel/search
GET  /api/intel/runs
GET  /api/intel/runs/:id
GET  /api/intel/runs/:id/items
GET  /api/intel/runs/:id/digest
POST /api/intel/runs/:id/cancel
```

Authentication:

- For personal local v1, these can require either login or API key.
- If auth is disabled locally, allow loopback access with API key for mutations.

Do not expose raw LLM prompts or full raw scraped content publicly by default.

## Scheduling And Deployment

### Services

Use LaunchAgents for long-lived services:

- ClawFeed API server.
- Worker daemon, once topical search exists.
- vMLX server, if not already separately managed.
- Ollama only if needed as fallback.

Use OpenClaw cron for scheduled daily brief triggers.

Reasoning:

- LaunchAgent is appropriate for keeping local services alive on macOS.
- OpenClaw cron is appropriate for agent-orchestrated scheduled tasks and final composition access.
- Plain crontab adds another control plane and should be avoided.

### Daily Schedule

Amended 2026-05-15: the daily brief is scheduled via macOS **`launchd`**, not OpenClaw cron. OpenClaw cron's real surface schedules `agentTurn` / `systemEvent` payloads (LLM calls), not arbitrary shell exec — so it isn't the right primitive for triggering the daily-brief CLI. `launchd` is the macOS-native scheduler for shell commands and ships in Phase 6b through a small installer subcommand.

```bash
# Installed by `clawfeed-intel cron install --install` in Phase 6b.
# Dry-run preview without `--install` prints the rendered plist without
# touching disk; mirrors the destructive-default-off posture from
# Phase 6a's cleanup CLI.
```

The installed agent (path: `~/Library/LaunchAgents/local.clawfeed.daily-brief.plist`) fires at `06:15` local each day and invokes:

```bash
cd /Users/merlin/clawfeed && uv run clawfeed-intel run daily --window 24h
```

Load/unload uses the modern `launchctl bootstrap gui/<uid>` / `launchctl bootout` interface (not the deprecated `launchctl load`). Standard output and standard error are captured to `data/logs/daily-brief.{out,err}.log` for post-hoc inspection.

### Worker Daemon

For Phase 1, a CLI invoked by cron is enough.

For Phase 2 topical search, add:

```bash
uv run clawfeed-intel worker --poll-interval 5
```

The worker claims `intel_jobs` from SQLite and processes them.

## Reliability Requirements

### Inference Reliability

The system must handle:

- vMLX temporarily unavailable.
- Model not loaded.
- Slow local responses.
- Invalid JSON.
- Partial completions.
- Frontier gateway unavailable.

Required behavior:

- Health check vMLX before run.
- Fail fast if required local model is unavailable.
- Retry transient local inference errors.
- Validate every structured output.
- Store failed LLM calls in `llm_calls`.
- Use local final composition fallback if OpenClaw/frontier fails.
- Mark brief metadata with degraded mode.

### Fetch Reliability

Every fetcher must have:

- Per-source timeout.
- Per-domain rate limit.
- Retries with backoff for transient errors.
- Permanent failure classification.
- Consecutive error count.
- User-Agent.
- SSRF protection for arbitrary user-provided URLs.

### Data Reliability

SQLite is acceptable for this single-machine system if:

- WAL mode is enabled.
- Worker uses short transactions.
- Job claiming uses `BEGIN IMMEDIATE`.
- Raw content has TTL cleanup.
- Large article bodies are capped.
- DB backups are configured before heavy use.

Recommended retention:

- Raw items: 30-90 days.
- Runs and summaries: indefinite until manually pruned.
- LLM call logs: 30 days or compressed after 90 days.

## Security And Privacy

Requirements:

- Keep all high-volume content local.
- Do not send raw article pools to frontier models.
- Do not expose raw scraped content via unauthenticated endpoints.
- Use API key or local auth for mutations.
- Continue SSRF protection for user-provided URLs.
- Store API keys in `.env`, not YAML.
- Respect API terms and robots/rate limits.

Frontier prompt should receive:

- Local summaries.
- Source URLs.
- Category metadata.
- Coverage stats.

Frontier prompt should not receive:

- Hundreds of raw article bodies.
- Private credentials.
- Local filesystem paths unless needed.

## Brief Format

The final daily brief should be Markdown.

Recommended structure:

```markdown
# Daily Intelligence Brief - May 4, 2026

## Executive Read

Short synthesis of the day.

## Highest-Signal Developments

Items across all categories that matter most.

## Startup Funding

Each relevant funding event, not just top N.

## AI Research

Research items with practical explanation.

## Models And Product Moves

Model releases, platform changes, major company moves.

## AI Coding Tools

Codex, Claude Code, Cursor, etc.

## GitHub And Developer Signals

Trending repos, HN/Reddit developer attention, notable launches.

## Watchlist

Lower-confidence or developing items worth monitoring.

## Coverage

Sources queried, counts, failures, and caveats.
```

Topical brief structure:

```markdown
# Topic Brief: Khosla Ventures

## Executive Read
## Recent Investments And Financing Activity
## Portfolio Company Moves
## Partnerships, Hiring, And Product Signals
## News Mentions And Interviews
## Filings And Structured Evidence
## Timeline
## Sources And Coverage
```

## Phased Build Plan

This plan is a build chronology, not a product downgrade path. Each phase narrows a different class of risk: data shape, source coverage, evidence quality, model behavior, publishing, operations, and then user-facing expansion. The daily brief should not be considered complete until the full daily path works end to end across the configured source families and produces a document that is worth reading.

### Phase 0: Shared Definition Of Done

Goal: make the target brief, acceptance bar, and operating assumptions explicit before engineering begins.

Work:

- Confirm the morning run window, time zone, source families, brief sections, and expected coverage footer.
- Decide what "kept" means for each category: funding, AI research, models, product moves, coding tools, repositories gaining traction, and developer-discussion signals.
- Identify the first realistic source set for each category, including both structured sources and dashboard-managed sources.
- Define the manual test questions the finished daily run must answer: what came in, what was rejected, what was clustered together, what was summarized, what was published, and which sources failed.

Gate:

- An engineer can describe the complete daily run in one page without ambiguity.
- The expected finished brief structure is stable enough to evaluate against real output.

### Phase 1: Durable Run And Source Foundation

Goal: establish the durable backbone that every later stage depends on.

Work:

- Add the persistent run record, job tracking, raw item storage, cluster storage, model-call logging, source fetch state, and digest linkage described in the data model.
- Unify category configuration with dashboard-added sources so category tags determine which user-visible sources join each category's effective source pool.
- Create the baseline source configuration for all daily categories and all daily source families.
- Establish coverage accounting from the beginning: attempted sources, successful sources, failed sources, item counts, cluster counts, kept counts, model choices, and degraded-mode flags.

Gate:

- A daily run can be created, updated through each lifecycle state, and inspected after completion or failure.
- Source selection can explain why each source was included in a category.
- Coverage can be recorded even before intelligence stages are active.

### Phase 2: Complete Daily Source Coverage

Goal: get all daily evidence streams producing normalized raw items before judging relevance.

Work:

- Bring up RSS, arXiv, Hacker News, Reddit, GitHub, GDELT, SEC EDGAR, and configured website collection as first-class daily sources.
- Treat GitHub Trending as repository discovery, not as the traction signal by itself; repository momentum needs stored observations over time.
- For each source family, run against realistic configured sources and verify that the output contains enough text and metadata for later judgment.
- Record source failures as coverage facts instead of stopping the run.

Gate:

- A single daily collection pass can gather items from all eight source families.
- Each source family has at least one real source represented in the raw item pool.
- Failed or rate-limited sources are visible in coverage and do not prevent the rest of the run from continuing.

### Phase 3: Evidence Hygiene And Event Formation

Goal: turn a noisy pile of pages, posts, filings, papers, and repositories into a cleaner set of candidate developments.

Work:

- Normalize links so tracking variants and duplicate URLs collapse.
- Add content-level duplicate detection so syndicated or mirrored articles do not inflate importance.
- Group distinct items that describe the same real-world event into clusters.
- Preserve supporting evidence inside each cluster so cross-source confirmation is available later.
- Add repository momentum calculations to distinguish genuine GitHub traction from simple appearance in a discovery source.

Gate:

- A repeated article found through multiple paths appears once as evidence, not many times as separate stories.
- A multi-source event, such as a funding announcement with news coverage and a filing, becomes one cluster with multiple supporting items.
- Repository traction can be explained with observed momentum rather than an opaque trending label.

### Phase 4: Local Intelligence Stages

Goal: use local inference for the high-volume judgment and condensation work.

Work:

- Route source planning, relevance filtering, and cluster summarization through the central model layer.
- Validate every structured model response and record failures without crashing the run.
- Tune relevance decisions against real examples until the filter keeps all meaningful items while rejecting obvious feed noise.
- Summarize each kept cluster into a grounded brief building block with headline, facts, why-it-matters, citations, caveats, confidence, and category placement.

Gate:

- The relevance stage keeps all clusters that should appear in the brief, not merely a top-N subset.
- Rejected clusters have understandable rejection reasons.
- Summaries are grounded in their source clusters and include enough citation context for final composition.

### Phase 5: Final Composition And Publishing

Goal: transform structured cluster summaries into the finished daily brief.

Work:

- Compose the final daily document through OpenClaw and the frontier model, using only condensed cluster summaries and coverage metadata.
- Preserve local fallback composition so a brief still publishes if the frontier path is unavailable.
- Publish the finished Markdown into ClawFeed as a daily digest with run metadata.
- Make the dashboard display the published brief and enough coverage context to explain thin or degraded days.

Gate:

- A manual daily run produces a finished digest visible in ClawFeed.
- The final document includes every relevant kept cluster.
- The brief clearly distinguishes confirmed signals, lower-confidence watchlist items, and source coverage gaps.
- Frontier composition and local fallback both produce publishable output, with metadata marking which path was used.

### Phase 6: Daily Operations

Goal: make the morning run reliable enough to trust as a routine.

Work:

- Register the OpenClaw scheduled trigger for the 6:15am daily run.
- Exercise common failure modes: source timeout, malformed model output, unavailable local model, unavailable frontier composition, and partial source coverage.
- Add retention and backup practices appropriate for a local personal system.
- Review several real daily runs and compare the finished brief against the raw evidence pool.

Gate:

- The scheduled run completes without manual intervention.
- A partially degraded run still publishes a useful brief and clearly names what degraded.
- The system leaves enough trace information to debug why a story appeared, disappeared, or moved sections.

### Phase 7: Topical Search

Goal: reuse the daily intelligence pipeline for on-demand research briefs.

Work:

- Add a search run lifecycle separate from the scheduled daily run.
- Let the local source planner choose source families, query variants, time windows, and expected evidence types for a topic.
- Reuse raw-cache search, dynamic external collection, deduplication, clustering, relevance filtering, summarization, composition, and publishing.
- Add a dashboard surface for submitting a topic, watching run status, reviewing search history, and opening the finished topic brief.

Gate:

- A query such as "Khosla Ventures" produces a consolidated topic brief with news, filings, developer signals, repository signals, and local cache evidence where applicable.
- The search result is organized by event, not by source.
- The daily brief path remains unaffected by topical-search failures.

### Phase 8: Quality Review And Feedback Loop

Goal: make the brief better over time by learning from real misses, false positives, and weak summaries.

Work:

- Add ways to mark irrelevant items, missed items, wrong categories, weak summaries, and useful clusters.
- Compare runs over time to see whether category rules, source choices, or prompts improve output quality.
- Build a small set of saved real-world runs that can be replayed when changing model prompts or relevance rules.
- Track coverage health by category so gaps are visible before they become trust problems.

Gate:

- Feedback can be tied back to the event cluster and stage that produced the issue.
- A prompt or relevance-rule change can be evaluated against saved runs before becoming the new default.
- Repeated source failures or category gaps are visible without reading logs.

### Phase 9: Expanded Signals And Optional Paid Sources

Goal: improve coverage where open sources prove insufficient.

Work:

- Evaluate paid or credentialed sources only where coverage gaps are demonstrated by actual runs.
- Add deeper structured sources where they improve confidence, such as richer filing workflows, package registry signals, vendor release feeds, or email/newsletter ingestion.
- Consider semantic search or embeddings if saved raw material becomes large enough that keyword retrieval misses relevant prior evidence.
- Revisit Twitter/X only as a separate maintenance and cost decision, not as a hidden dependency of the daily brief.

Gate:

- Each new signal source has a clearly identified gap it closes.
- Added sources improve final brief quality enough to justify their maintenance cost.
- The system remains understandable: the coverage footer can still explain where the brief's evidence came from.

## Pushback And Risks

### "Comprehensive Across The Open Web" Is Not Fully Achievable For Free

The system can be comprehensive across configured categories and accessible sources. It cannot guarantee total web coverage without a paid search/news index.

Mitigation:

- Include coverage footer.
- Add paid search only if gaps justify it.
- Keep curated source lists explicit and inspectable.

### ClawFeed's Current Digest Type Schema Is Too Narrow

The existing `digests.type` constraint only allows `4h`, `daily`, `weekly`, and `monthly`. Topical search does not naturally fit this.

V1 workaround:

- Store topic briefs as `daily` or `4h` with `metadata.brief_kind='topic'`.

Better Phase 2 fix:

- Relax the constraint or add `brief_kind` column.
- Update UI filtering to distinguish scheduled daily briefs from topical search results.

### SQLite Is Fine, But Only If We Respect It

SQLite can handle this personal system, but not if long transactions and huge raw blobs accumulate forever.

Mitigation:

- WAL mode.
- Short transactions.
- Raw item TTL.
- Content size caps.
- Periodic backup.

### Local LLM JSON Is A Reliability Risk

Local models may emit malformed JSON under load or long context.

Mitigation:

- Strict schemas.
- Small batches.
- JSON repair once, then fail the item.
- Store failures for inspection.
- Prefer deterministic prompts and low temperature for structured stages.

### GitHub Trending Is A Weak Signal Alone

Trending pages are useful for discovery but opaque as evidence. "Gaining traction" should be based on observed repository momentum, with trending treated as one way to find candidates rather than the signal itself.

Mitigation:

- Record repository observations over time.
- Compare short-window and longer-window momentum.
- Pass repository traction forward as an evidence-backed signal, not as a page-placement claim.

### Reddit Can Be Noisy And Rate-Limited

Reddit should inform the brief, not dominate it.

Mitigation:

- Curated subreddits.
- Conservative limits.
- Local relevance threshold.
- Clear source weighting.

### Gemini CLI Tail Latency And Stream Stalls (Added 2026-05-15)

Gemini CLI (the Phase-1-final-compose mechanism after the 2026-05-15 amendment) occasionally stalls mid-response on long generations. The model continues to be reachable but the local CLI process stops emitting tokens without erroring, producing an indefinite hang.

Mitigation:

- Use `stream-json` output mode so the provider sees per-token events. Track the gap between events as the stall signal — far faster than waiting out a wall-clock timeout.
- Per-token-idle threshold (60s) plus a hard wall-clock cap (300s). Either trigger sends `SIGTERM` to the subprocess; if it doesn't exit within 5s, send `SIGKILL`.
- One retry on stall / timeout / non-zero exit with a 10s backoff.
- Tier-2 fallback to local vMLX compose (`Qwen3.5-122B-A10B-4bit`) using the same prompt. Tier-3 fallback to the deterministic `render_fallback_brief` path.
- `composition_provider` metadata always reflects which tier produced the brief, so persistent Tier-2/3 days are visible to the operator.

### Node Toolchain Drift Can Disable Locally-Installed Node CLIs (Added 2026-05-15)

The local `/opt/homebrew/opt/node@22/bin/node` is linked against `libsimdjson.30.dylib`, which homebrew has upgraded out (only versions 33/34.x are installed). The system PATH puts this broken `node@22` ahead of the working `/opt/homebrew/bin/node` v25.9.0, so any Node CLI whose shebang resolves through PATH (`gemini`, `openclaw`, others) fails at startup with a dyld linker error.

Mitigation:

- The Gemini CLI provider invokes the working node binary explicitly via `executable_path` in `model-routing.yaml`, rather than relying on PATH resolution.
- This is brittle long-term. The correct fix is to repair the local node toolchain (reinstall `simdjson@4.2`, or migrate `node@22` consumers to node 25). Tracked as an environmental risk in `docs/current-project-status.md`.

### Frontier CLI Subprocess Reliability vs API Reliability (Added 2026-05-15)

Calling a frontier model through a CLI subprocess (rather than directly via an HTTP API) trades one set of failure modes for another. The CLI handles auth, model selection, and request shaping for us — but it introduces process-level failure modes (broken shebang resolution, OAuth refresh during long calls, stream stalls in the CLI's internal buffering) that an HTTP client would not have.

This tradeoff is accepted because:

- **The Gemini CLI is signed into the operator's Gemini Pro subscription via OAuth — not the pay-per-token Gemini API.** The worker holds no API key, has no per-token cost accounting, and is bounded only by the Pro plan's subscription quota. Removing API-key management from the worker is the load-bearing operational simplification; pivoting to a direct API integration would re-introduce key rotation, billing reconciliation, and per-call cost tracking that the subscription-via-CLI shape avoids entirely.
- The CLI's subprocess contract (stdin → stdout → exit code) is stable across CLI versions in ways the underlying gateway/API protocol may not be.
- For one composition call per day, the additional process-management cost is a few dozen lines of provider code, and the fallback chain ensures a degraded brief still publishes even if every CLI mode fails.

The cost model implication for Phase 7 (topical search): topical briefs incur a *second* compose call per topic query. Pro plan ceilings comfortably cover several queries per day; if Phase 7 reaches a usage pattern that approaches the ceiling, the right response is to surface a `quota_exceeded`-style error from the CLI and rely on the Tier-2 vMLX fallback, **not** to switch to a per-token API model.

## Build Chronology Summary

The engineering sequence should move from foundations to evidence breadth, then to judgment, publishing, operations, and expansion:

1. Align on the daily brief's acceptance bar.
2. Establish durable run state, source-category mapping, and coverage accounting.
3. Bring all daily source families online and normalize their output.
4. Clean the evidence pool through deduplication, clustering, and signal enrichment.
5. Add local relevance and summary stages.
6. Add frontier final composition, fallback composition, and publishing.
7. Register and harden the scheduled daily run.
8. Reuse the same pipeline for topical search.
9. Add feedback, replay, and quality review loops.
10. Expand into paid or specialized sources only where real runs show coverage gaps.

This chronology protects the core daily brief first, then builds outward. It is intentionally ordered around dependency and confidence rather than around the fastest visible demo.

## External References

- ClawFeed current source/digest implementation: `/Users/merlin/clawfeed/src/server.mjs`, `/Users/merlin/clawfeed/src/db.mjs`
- ClawFeed current schema: `/Users/merlin/clawfeed/migrations/`
- ClawFeed source personalization PRD: `/Users/merlin/clawfeed/docs/prd/source-personalization.md`
- vMLX endpoint: `http://127.0.0.1:8080/v1`
- OpenClaw cron jobs: `/Users/merlin/.openclaw/cron/jobs.json`
- Qwen3.6-35B-A3B model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Qwen3.6-35B-A3B FP8 files: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
- Qwen3.5-122B-A10B model card: https://huggingface.co/Qwen/Qwen3.5-122B-A10B
- Qwen3.5-122B-A10B MLX 4-bit port: https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-4bit
- Qwen3.5-35B-A3B model card: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- Qwen3.5-35B-A3B MLX 4-bit port: https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit
- Mistral Small 4 119B model card: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
- Mistral Small 4 119B MLX 4-bit port: https://huggingface.co/mlx-community/Mistral-Small-4-119B-2603-4bit
- DeepSeek V4 Flash model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- DeepSeek V4 Flash MLX 4-bit port: https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit
- Kimi K2.6 model card: https://huggingface.co/moonshotai/Kimi-K2.6
- GLM-5.1 model card: https://huggingface.co/zai-org/GLM-5.1
- arXiv API manual: https://info.arxiv.org/help/api/user-manual.html
- SEC EDGAR data access guidance: https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm
- GDELT DOC 2.0 API: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
- Hacker News official API: https://github.com/HackerNews/API
- Reddit API docs: https://www.reddit.com/dev/api/
- HTTPX docs: https://www.python-httpx.org/
- feedparser docs: https://feedparser.readthedocs.io/
- Trafilatura docs: https://trafilatura.readthedocs.io/
