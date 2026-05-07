# Handoff — Steps 9–12: Relevance Filter, Cluster Summary, Final Composition, Publish

**Audience:** the engineer (likely a new Claude session) picking up Phase 1 of the Personal Intelligence Brief System after step 8c.

**Repo:** `/Users/merlin/clawfeed` · branch `main` · remote `origin` = `https://github.com/Charliemorrone/the-front-page.git`.

---

## Your role

You are a senior engineer continuing implementation of the Personal Intelligence Brief System for The Front Page (the user's personal fork of ClawFeed). Phase 1 is mid-stream. Step 8 (LLM client + active doctor probe) shipped cleanly across three commits (8a, 8b, 8c). You are picking up after step 8c with the LLM client built, tested, doctor-probed against live vMLX — ready for the four pipeline stages that close the run loop.

Your job is to drive the remaining substeps that take Phase 1 to its acceptance gate: **relevance filter → cluster summary → final composition → publish**. When that lands, `clawfeed-intel run daily --window 24h` produces a finished, readable digest visible at `http://127.0.0.1:8767/`. That is the Phase 1 acceptance gate.

---

## Hard requirements (non-negotiable, carried from the original brief)

- Single-user, local-only.
- Final daily brief quality is the primary goal.
- All eight daily fetchers belong in Phase 1 — already shipped.
- GitHub "repos gaining traction" must use real velocity from stored observations, not Trending alone — already shipped.
- Final composition must use OpenClaw/`gpt-5.3-codex` with local `Qwen3.5-122B-A10B-4bit` fallback (or whatever flagship is cached on the machine).
- The brief includes all relevant kept items, not a top-N subset.
- Failed sources degrade coverage; they do not fail the run. (Extends to LLM stages too — a vMLX call failing should degrade the brief, not crash the run.)
- ClawFeed (now bundled into The Front Page) remains the product shell.

**Plus the Option-A reaffirmation from 2026-05-05:**

- **Frontier cloud (`gpt-5.3-codex` via OpenClaw) is strictly bounded to the final-compose stage only.** Local vMLX handles everything else — relevance filter, cluster summary, source planning. This is non-negotiable. Routing 56 batches of 12 clusters through frontier is exactly what the architecture-doc rule prohibits.
- **Local tokens are unbounded.** The architecture is built around generous local inference. Don't ration vMLX calls; the cost discipline is "no paid tokens for high-volume work," not "make as few local calls as possible."
- **Twitter stays out of v1.** If you're tempted to add it, refer back to the Option-A scope decision. The cheapest future path is Nitter mirrors as RSS sources — no architectural change required.
- **OpenClaw enters only at step 11 (final compose) and Phase 6 (cron trigger).** Not for fetching, not for relevance, not for summary.

## Engineering hygiene (also non-negotiable)

- `git status` before editing.
- Small, inspectable steps. One concern per change. Don't bundle unrelated work.
- `uv run pytest` and `uv run ruff check . && uv run ruff format --check .` must stay clean. **Current baseline: 561/561 tests passing.**
- Update [docs/current-project-status.md](current-project-status.md) after each substantive change — Built / In progress / Build log / new risks. Follow the "How to keep this file current" section at the top.
- Don't overwrite unrelated existing changes.
- Masterful code quality: type hints, focused docstrings explaining *why* (not *what*), no premature abstraction, no comments restating code, validation at boundaries only.
- Trust internal code; validate at system boundaries (HTTP, user input, file I/O). Don't add error handling for scenarios that can't happen.
- High-quality code matters more than speed. Be smart, careful, and calculated.

---

## Required reading, in order

Before you write any code, read these and confirm you understand them:

1. **[docs/personal-intelligence-brief-architecture.md](personal-intelligence-brief-architecture.md)** — full architecture, data model, model routing, phased build plan, per-stage specifications. **The "Prompt Responsibilities" section is the load-bearing spec for steps 9, 10, and 11** — input/output JSON contracts for relevance filter, cluster summary, final compose. Pay attention to:
   - "Prompt Responsibilities" → "Relevance Filter" — output JSON shape (`keep`, `category`, `score`, `event_type`, `reason`, `entities`, `evidence_urls`, `uncertainty`).
   - "Prompt Responsibilities" → "Cluster Summary" — output JSON shape (`headline`, `summary`, `why_it_matters`, `entities`, `key_facts`, `caveats`, `source_urls`, `confidence`).
   - "Prompt Responsibilities" → "Final Compose" — input is structured cluster summaries + category rules + coverage report + run metadata; output is the markdown brief.
   - "Frontier Model Use Is Strictly Bounded" — the rule.
   - "Brief Format" — recommended Markdown structure for the final daily brief.
   - "Reliability Requirements / Inference Reliability" — failure handling.
   - "Data model: `item_summaries`" — the row-per-kept-cluster table you'll write to in step 10.

2. **[docs/current-project-status.md](current-project-status.md)** — source of truth for current build state. Tells you exactly what's built, what's next, what risks are live, and the full append-only build log.

3. **`/Users/merlin/.claude/projects/-Users-merlin-clawfeed/memory/MEMORY.md`** and the **`vmlx_environment.md`** memory file it indexes. Verified facts about the running vMLX server: port, loaded model, on-disk inventory, multi-model hot-swap behavior. **Verify the memory still matches reality** before relying on it (vMLX may have been restarted; the user may have downloaded new models).

4. Skim what's shipped that you'll consume:
   - **`worker/clawfeed_intel/llm/`** — `LLMClient.chat_completion(stage, messages, response_schema=None, prompt_version=None) -> CallResult`. Constructor: `LLMClient(routing, *, transport=None, conn=None, run_id=None, retry_config=None)`. When you pass `conn` and `run_id`, every call writes a row to `llm_calls`. When you pass `response_schema: type[BaseModel]`, the response content is JSON-parsed and validated; one repair attempt fires on failure; final failure raises `LLMSchemaError`. **All your LLM HTTP goes through this.**
   - **`worker/clawfeed_intel/llm/routing.py`** — load_routing reads `config/model-routing.yaml`. The four stages you care about are `source_planning`, `relevance_filter`, `cluster_summary`, `final_compose`. The latter is currently routed to vmlx as a Phase-1 placeholder; step 11 is where you switch its provider to `openclaw`.
   - **`worker/clawfeed_intel/db.py`** — connection management, `BEGIN IMMEDIATE` transactions, run-state CRUD. You'll add new helpers for cluster-status updates and `item_summaries` insert.
   - **`worker/clawfeed_intel/pipeline/orchestrator.py`** — the `_drive_run` function walks states. You'll fill in the `filtering` stage (after `cluster_run`), the `summarizing` stage, and the `composing` stage.
   - **`worker/clawfeed_intel/pipeline/cluster.py`** — `cluster_run` produces clusters with `status='pending'`. Your relevance filter promotes them to `'kept'` or `'filtered_out'`. Your cluster summary promotes `'kept'` → `'summarized'`.
   - **`worker/clawfeed_intel/runs.py`** — `Coverage` (set `kept_clusters`) and `RunMetadata` (set `composition_provider`, `composition_model`, `local_models`).
   - **`config/intel-sources.yaml`** — category configs the relevance filter prompts will reference.
   - **`migrations/010_intel_pipeline.sql`** lines 110–145 (`item_summaries`, `llm_calls`).

5. **Skim a representative existing test** (`worker/tests/test_fetchers_hn.py` or `worker/tests/test_cluster.py` or `worker/tests/test_llm_client.py`) to absorb the project's testing style: pure-parse with hand-built fixtures + `httpx.MockTransport` for HTTP. The same fixture-driven style applies to LLM-stage tests — never hit live vMLX in unit tests.

---

## After reading, summarize back to the user

Confirm you understand:

1. The current build state (steps 1–8c done, 561/561 tests pass; the orchestrator's `filtering` stage produces clusters with `status='pending'` waiting for relevance promotion; the LLM client is built, tested, and doctor-validated against live vMLX).
2. The architecture-doc spec for relevance/summary/compose prompts — input/output JSON contracts.
3. The Option-A scope rule: frontier-only-for-compose, local-vmlx-for-everything-else.
4. Where each LLM call slots into the orchestrator state machine (filter inside `filtering`, summary inside `summarizing`, compose inside `composing`).
5. Your proposed plan for the substep you are going to ship first.

**Wait for the user to confirm your summary before you start writing code.**

---

## Where the previous engineer left off

- Phase 1 step 8 (LLM client + active doctor probe) is complete. Three commits on `main`:
  - `50445de` feat(intel): add LLM routing config + minimal client (Phase 1 step 8a)
  - `1c6450e` feat(intel): add LLM client reliability layer (Phase 1 step 8b)
  - `4941712` feat(intel): add active vMLX probe to doctor command (Phase 1 step 8c)
- Plus three earlier clustering commits (`f3e2e89`, `8a50c57`, `cd6027a`) and step 6 fetcher work. **Six local commits ahead of `origin`, not pushed.**
- 561/561 tests pass; ruff check + format clean.
- Live-vMLX smoke confirmed working: `clawfeed-intel doctor` exits 0 with all probes green; chat probe to Qwen3.5-27B-4bit returns 'PONG' in ~2s.
- The orchestrator's `filtering` stage runs `cluster_run` and stops. `summarizing` and `composing` stages advance status and do nothing else. Your work fills those bodies in.

**State of the world:**

- Repo: `/Users/merlin/clawfeed`, branch `main`.
- All 11 intel pipeline tables are live (migrations 010 + 011).
- vMLX is verified running on `127.0.0.1:8080`; default warm model `mlx-community/Qwen3-8B-4bit`; cached: Qwen3.5-27B-4bit (15G), Qwen3-30B-A3B-4bit (16G), Qwen2.5-32B-Instruct-4bit (17G), Llama-3.3-70B-Instruct-4bit (37G), Qwen2.5-VL-72B-Instruct-4bit (39G), Qwen3-Coder-Next-4bit (42G), Qwen3-Coder-Next-6bit (60G), plus image/audio models. Architecture-doc flagship `Qwen3.5-122B-A10B-4bit` (~70G) and `Qwen3.6-35B-A3B` are still NOT on disk — Phase 1 routes filter+summary to Qwen3.5-27B-4bit as a substitute. The user can swap to the doc-target models by editing `config/model-routing.yaml`; no code change required.
- OpenClaw is installed at `~/.openclaw/`, gateway on `ws://127.0.0.1:18789`, OAuth'd to `gpt-5.3-codex`. Step 11 wires the openclaw provider into `LLMClient`.
- Python venv provisioned via `uv sync --extra dev`. `uv run pytest -q` and `uv run ruff check .` are the validation commands.

---

## The next development phases

This handoff covers four substeps in dependency order. Each is a self-contained commit gated on green tests + ruff. **Stop and check in with the user between substeps. Don't bundle.**

### Step 9 — Relevance filter

Batched cluster judgement against category rules. Lives in `worker/clawfeed_intel/pipeline/relevance.py`. Two layers (mirror fetchers + clustering):

**Pure layer:**
- `build_relevance_messages(clusters_batch, categories) -> list[dict[str, str]]` — constructs the OpenAI-style messages list. System message lays out the categories + rules; user message lists the batch's clusters with their titles + representative URLs + (optionally) excerpts. Deterministic and fixture-testable.
- `parse_relevance_verdicts(parsed_payload, expected_count) -> list[RelevanceVerdict]` — the LLM's response is a JSON list of one verdict per cluster (in the same order as the batch). The LLM client already validates the JSON via `response_schema`; this helper just unpacks the validated object.
- `RelevanceVerdict` pydantic schema in `worker/clawfeed_intel/llm/schemas.py` (new file). Fields per architecture doc: `keep: bool`, `category: str`, `score: float` (0–1), `event_type: str | None`, `reason: str`, `entities: list[str]`, `evidence_urls: list[str]`, `uncertainty: float | None`. The wrapper schema is `RelevanceBatchResponse` with a `verdicts: list[RelevanceVerdict]` field.

**Async orchestration layer:**
- `async filter_clusters(conn, run_id, llm_client, *, batch_size=12, prompt_version="relevance.v1") -> int` — load clusters with `status='pending'` for the run, batch them (default 12 per architecture doc), call `llm_client.chat_completion(stage="relevance_filter", messages=..., response_schema=RelevanceBatchResponse, prompt_version=...)` per batch, parse verdicts, update `item_clusters.status` to `'kept'` or `'filtered_out'` plus `relevance_score`, `category`, `event_type`, `filter_reason`. Returns the number of `'kept'` clusters.
- New `db.py` helper: `update_cluster_verdict(conn, *, cluster_id, status, relevance_score, category, event_type, filter_reason)`. One UPDATE. Idempotent per cluster.
- Wired into `orchestrator._drive_run`'s `filtering` stage **after** `cluster_run`. Sets `metadata.coverage.kept_clusters`.

**Architecture-doc rule that's load-bearing:** "This stage must be allowed to keep many items. It is not a top-N selector." Design prompts and thresholds accordingly. If the filter rejects 90% of clusters from a typical run, your threshold is wrong.

**Per-batch failure model:** if a batch's LLM call raises (transient retry exhausted, schema validation failed twice), the run **continues** — those clusters stay at `status='pending'` and don't make it into the brief. Coverage records this as `coverage.failed_filter_batches += 1` (you'll add this field to `Coverage`). This matches the "failed sources degrade coverage; they do not fail the run" hard requirement, extended to LLM stages.

**Tests:** pure prompt construction over fixture clusters + categories, pure response parsing across valid + edge-case shapes (e.g., partial verdicts, ordering mismatch — the count check should fail loudly), orchestration end-to-end with a stub LLM client (no live vMLX). Mirror the cluster.py test structure — pure passes test deterministic shapes, orchestration tests verify DB state changes.

**Suggested split:**
- 9a: schemas + pure helpers + DB helper. No orchestration yet. Tests cover schemas/parsing/DB updates.
- 9b: async `filter_clusters` + orchestrator wiring. Tests cover end-to-end with stub LLM client.

### Step 10 — Cluster summary

One LLM call per kept cluster. Lives in `worker/clawfeed_intel/pipeline/summary.py`. Same two-layer pattern.

**Pure layer:**
- `build_summary_messages(cluster_with_items, category) -> list[dict[str, str]]` — constructs messages. System lays out the summary contract (headline, why-it-matters, etc., per architecture doc); user message provides the cluster's title, all member items' titles + URLs + excerpts.
- `parse_summary(parsed_payload) -> ClusterSummaryPayload` — unpacks the validated pydantic object.
- `ClusterSummaryPayload` schema in `llm/schemas.py`: `headline: str`, `summary: str`, `why_it_matters: str = ""`, `entities: list[str] = []`, `key_facts: list[str] = []`, `caveats: list[str] = []`, `source_urls: list[str] = []`, `confidence: float | None = None`.

**Async orchestration layer:**
- `async summarize_clusters(conn, run_id, llm_client, *, prompt_version="summary.v1") -> int` — load clusters with `status='kept'`, for each: load member items (title + url + excerpt + content), one chat-completion per cluster against `cluster_summary` stage with the schema, write one row to `item_summaries`, advance cluster status to `'summarized'`. Returns the number of summaries written.
- New `db.py` helper: `create_item_summary(conn, *, cluster_id, model, prompt_version, headline, summary, ...)`. Validates required fields; serializes `entities`/`key_facts`/`caveats`/`source_urls` to JSON strings (the schema stores them as TEXT).
- Wired into `orchestrator._drive_run`'s `summarizing` stage.

**Per-cluster failure model:** if a summary call raises, that cluster stays at `'kept'` and doesn't get summarized. The compose stage filters by `status='summarized'` so the brief just won't include it. `coverage.failed_summary_clusters` records the count. Don't crash the run.

**Tests:** pure prompt + parsing, end-to-end orchestration with stub LLM client + fixture clusters at `'kept'` status, verifying `item_summaries` rows are written and statuses advance to `'summarized'`. One test covers the per-cluster-failure-doesn't-abort-batch case.

### Step 11 — Final composition

**This is where OpenClaw enters.** Architecture doc references the gateway at `ws://127.0.0.1:18789`. Lives in `worker/clawfeed_intel/pipeline/compose.py`.

**Routing config widening:** Add `OpenclawProviderConfig` to `llm/routing.py` with fields `gateway_url: str`, `model: str`, `auth_token_env: str | None = None`. Widen `StageConfig.provider` from `Literal["vmlx"]` to `Literal["vmlx", "openclaw"]`. Add `openclaw: OpenclawProviderConfig | None = None` to `ProvidersConfig`. Update `config/model-routing.yaml`'s `providers` block to declare openclaw, and switch `final_compose.provider` to `openclaw` (with a fallback that points back to vmlx + the local flagship model).

**Client widening:** `LLMClient._build_url` and the HTTP layer need a branch on `stage_config.provider`. For `openclaw`, the transport is **WebSocket**, not plain HTTP. The gateway accepts authenticated requests; the auth token lives in `~/.openclaw/openclaw.json` under `gateway.auth.token`. **Do not commit that token.** Read it at runtime from the file or from the env var the routing config names. The existing OpenClaw config on this machine has `auth.mode = "token"` with a token already set; load it from `~/.openclaw/openclaw.json` unless the env var is set.

  - Implementation note: the OpenClaw gateway's exact wire protocol isn't documented in this repo. Probe it before designing the client: `curl --include --header "Upgrade: websocket" ...` against the gateway and inspect what shape it expects. The gateway is OAuth'd to `gpt-5.3-codex` so it's likely a thin proxy that accepts an OpenAI-shaped chat-completion request. If it does accept that shape, the existing `LLMClient` HTTP path can be reused with one branch on transport. If it requires a different message envelope, isolate that in a separate `_OpenclawTransport` adapter and keep the public `chat_completion` interface unchanged.
  - **Fallback policy is load-bearing.** When the openclaw call fails (any reason), retry the compose stage against the local fallback (`stage_config.fallback`, which points at vmlx + the strongest cached local model). Stamp `RunMetadata.composition_provider = "vmlx_fallback"` instead of `"openclaw"`. The brief still publishes — degraded mode, but readable.

**Pure layer:**
- `build_compose_messages(summaries, coverage, run_metadata, brief_template) -> list[dict[str, str]]` — constructs the compose prompt. System message is the editorial direction (sections, voice, "include all kept items", "preserve citations"). User message is the structured payload: every cluster summary as JSON, plus the coverage block, plus any category rules.
- The architecture doc's "Frontier Model Use Is Strictly Bounded" rule means **only condensed cluster summaries + coverage metadata** go to OpenClaw. Never raw article bodies. The compose prompt's user-message payload should be small (a few KB at most for a typical 30-cluster brief). Verify this with a token-count assertion in tests.

**Async orchestration layer:**
- `async compose_brief(conn, run_id, llm_client_openclaw, llm_client_local_fallback, run_metadata) -> str` — load all `'summarized'` clusters with their summaries + member item URLs, call openclaw, on failure call local fallback, return the markdown brief. Does not write to the DB itself — the caller (orchestrator) handles the digest insert.

**Tests:** pure prompt construction, schema validation, openclaw transport mock, fallback path triggered on openclaw failure, metadata stamping verified. The openclaw transport tests will need new fixtures since they're WebSocket — probably worth a small `worker/tests/test_openclaw_transport.py` with a mock WebSocket server (or a high-fidelity AsyncMock).

**Suggested split:**
- 11a: routing widening + config update. No transport code yet. Tests verify the schema accepts both providers and the YAML loads.
- 11b: openclaw transport. Tests use mock WebSocket.
- 11c: compose stage prompt + orchestration with fallback. Tests cover both provider paths.

### Step 12 — Publish (Phase 1 acceptance gate)

Replace the orchestrator's `_compose_stub` with the real markdown brief from step 11. Wire the digest insert with full coverage `metadata`.

- `orchestrator._drive_run` calls `compose_brief(...)` during the `composing` state. The returned markdown becomes the `digests.content`.
- `RunMetadata` updated with `composition_provider`, `composition_model`, `local_models` dict (populated as each stage runs — relevance filter sets `local_models["filter"] = stage_config.model`, summary sets `local_models["summary"]`, compose sets `composition_model`).
- `db.create_digest` already exists. The metadata blob includes the full coverage object plus the model choices.

**Acceptance gate:** a manual `clawfeed-intel run daily --window 24h` produces a finished digest visible at `http://127.0.0.1:8767/`. The brief should be readable and include every relevant kept cluster — not a top-N subset. Phase 1 done.

---

## What "done" looks like for step 9 (the immediate task)

After your work lands:

- `worker/clawfeed_intel/pipeline/relevance.py` exists with the two-layer split.
- `worker/clawfeed_intel/llm/schemas.py` is created with at least `RelevanceVerdict` + `RelevanceBatchResponse` pydantic models (more schemas land in step 10).
- `db.py` has a new `update_cluster_verdict(...)` helper.
- `Coverage` dataclass extended with `failed_filter_batches: int = 0` (and similar for failed_summary in step 10).
- The orchestrator's `filtering` stage calls `cluster_run` *then* `filter_clusters`.
- Tests cover prompt construction, response parsing, DB writes, and end-to-end orchestration with a stub LLM client. **No live vMLX in CI.**
- A manual `clawfeed-intel run daily --window 24h` smoke against real vMLX produces a digest with `coverage.kept_clusters > 0` and clusters in the DB at `status='kept'` ready for step 10. Capture the count for the build log.
- Status doc updated with a build-log entry.
- `uv run pytest -q` and `uv run ruff check . && uv run ruff format --check .` clean.

### Suggested split into substeps

- **Step 9a — schemas + pure layer + DB helper.** Create `llm/schemas.py` with `RelevanceVerdict` + `RelevanceBatchResponse`. Pure `build_relevance_messages` + `parse_relevance_verdicts`. New `db.update_cluster_verdict`. Tests cover the schemas, the prompt construction, the parsing, and the DB write. Orchestration not touched yet.

- **Step 9b — async filter_clusters + orchestrator wiring.** Add the async `filter_clusters` function that loads pending clusters, batches them, calls the LLM client with the schema, and writes verdicts. Wire it into `orchestrator._drive_run`. Tests cover end-to-end orchestration with a stub LLM client + verify `coverage.kept_clusters` is set + verify per-batch-failure-doesn't-crash-run.

Each substep is one commit, gated on green tests + ruff. **Stop between substeps.**

---

## Specific design guidance

- **Two-layer pattern.** Mirror the fetchers and clustering. Pure helpers (prompt construction, response parsing, DB updates) are testable from fixtures; thin async orchestration on top threads `LLMClient` and `sqlite3.Connection`. The async layer doesn't do business logic — it just sequences calls and persists results.

- **Prompt versioning.** Every `chat_completion` call gets a `prompt_version` string (e.g. `"relevance.v1"`, `"summary.v1"`, `"compose.v1"`). When you change a prompt's behavior, bump the version. The `llm_calls.prompt_version` column lets you correlate run quality against prompt iterations later.

- **Schemas live in `llm/schemas.py`.** Not in the pipeline-stage modules. Keeping them centralized lets the LLM client and the pipeline stage import the same definition. Pydantic models with `extra="forbid"` for tight validation; bounded numeric fields (`Field(ge=0, le=1)` for scores, etc.) so the schema rejects nonsense.

- **Batch construction discipline (relevance filter).** A batch shouldn't exceed the model's context window. Architecture doc default is 12 clusters; assume each cluster contributes ~500-1000 tokens of context (title + URL + excerpt + entities). With 12 clusters and a 27B model, you're well within context. If a future stage uses larger excerpts, drop batch_size to fit.

- **JSON-mode prompts.** Local models malformed-JSON-under-load is a named risk. Defenses:
  - Set `temperature=0` or `temperature=0.1` for structured stages (you may need to extend `chat_completion` to accept a `temperature` argument; the OpenAI-shaped body supports it).
  - Make the prompt explicit: "Reply with valid JSON matching this exact schema. No markdown, no commentary, no preamble. Begin your response with `{`."
  - Use small batches.
  - The LLM client's repair retry catches one failure; tighter prompts catch the rest.

- **Where to call the LLM client.** Construct `LLMClient` in `orchestrator._drive_run` once per run, pass it down to each stage. Constructor: `LLMClient(routing, conn=conn, run_id=run_id)`. Don't construct it inside the pure helpers. Don't construct it inside per-cluster loops.

- **Failure model for the new stages.**
  - Per-batch failure (relevance): record `coverage.failed_filter_batches += 1`, leave clusters at `'pending'`, continue with next batch. The run still publishes.
  - Per-cluster failure (summary): record `coverage.failed_summary_clusters += 1`, leave cluster at `'kept'`, continue. The brief just won't include that cluster.
  - Compose failure (openclaw): retry against local fallback (architecture-doc requirement). Stamp `composition_provider="vmlx_fallback"`. Brief still publishes.
  - Compose failure (both openclaw AND fallback): finish run with `status='failed'`. The state machine handles this — no special logic needed.

- **Coverage block stays honest.** Failed sources, failed filter batches, failed summary clusters all flow into `coverage.*`. The brief's "Coverage" section at the bottom names them. Don't hide degradation.

- **prompt_version naming convention.** `<stage>.<version>` — `relevance.v1`, `summary.v1`, `compose.v1`. When you tweak a prompt template's behavior, bump to `relevance.v2`. The architecture doc's "Phase 8 Quality Review" later reads `llm_calls.prompt_version` to evaluate whether prompt iterations improved quality.

- **Don't import from `fetchers/`.** The LLM stages are independent of the fetcher subsystem. Same architecture rule that kept the LLM client decoupled from `fetchers/http.py`.

- **Docstrings explain WHY.** Not what. Not "fetches clusters from the database" — instead "relevance verdicts must be applied in the same order as the batch was sent; the LLM is instructed to preserve order and the parsed list length must match the input or the verdict assignment is meaningless."

## Don't

- Don't build OpenClaw integration before step 11. Steps 9 and 10 use vmlx exclusively.
- Don't expand `intel_runs.status` for sub-states. LLM calls happen *inside* existing states.
- Don't store full prompt or response text in `llm_calls`. Hashes only — that's enforced by the column shape, but don't add new columns either.
- Don't add streaming completions. Phase 1 is synchronous.
- Don't expand the `digests.type` CHECK constraint. Topic briefs piggyback on `daily` per the open risks list.
- Don't try to live-vMLX-test in CI. Mock at the HTTP boundary; manual smoke against live vMLX is a separate one-shot action.
- Don't make the relevance filter a top-N selector. Architecture doc explicitly: "This stage must be allowed to keep many items."
- Don't refactor the fetcher contract or the clustering layer. They work. Touching them adds risk for no benefit.
- Don't invent new schemas or contracts. Use the architecture doc's "Prompt Responsibilities" section as the spec.
- Don't bundle steps. 9a, 9b, 10, 11a, 11b, 11c, 12 are each their own commit.
- Don't push to origin without explicit user instruction. Six commits are already local — they stay there until the user says otherwise.

## Pattern to mirror (from every shipped step)

Every shipped step follows this rhythm:

1. Read the architecture doc + status doc + memory + a representative existing module.
2. Summarize back to the user; propose the substep plan.
3. Wait for approval.
4. Implement: schema/data layer first if needed, then pure helpers, then async orchestration, then orchestrator integration, then tests, then status doc update.
5. `uv run pytest -q` + `uv run ruff check .` + `uv run ruff format --check .` must all be clean.
6. Commit with a descriptive message. **Don't push to origin without explicit user instruction.**
7. Stop and check in with the user before the next substep.

## Open risks

| Risk / decision | Status | Notes |
|---|---|---|
| `digests.type` CHECK only allows `4h`/`daily`/`weekly`/`monthly` | accepted for Phase 1 | Topic briefs (later phase) piggyback on `daily` + `metadata.brief_kind='topic'`. |
| Architecture-doc flagship model not yet on disk | open | `Qwen3.5-122B-A10B-4bit` (~70G) and `Qwen3.6-35B-A3B` not downloaded. Phase 1 uses `Qwen3.5-27B-4bit` (15G, on disk). Configurable via YAML. |
| Local LLM JSON reliability under load | open until 9b lands | Already mitigated by pydantic schemas + `LLMClient`'s bounded JSON repair. Step 9 will exercise this in earnest at scale (~56 batches × 12 clusters per daily run). |
| OpenClaw gateway wire protocol unknown | open until step 11 | Probe the gateway before designing the openclaw transport. May or may not accept the OpenAI-shaped chat-completion body unchanged. |
| Two-runtime SQLite access (Node + Python) | accepted | WAL on; worker writes use `BEGIN IMMEDIATE` and short transactions. LLM-stage DB writes follow the same discipline. |
| Six local commits not pushed to origin | open | Steps 7a/b/c, 8a/b/c are local on `main`. Don't push without explicit user instruction. |

---

## Begin

Read [docs/personal-intelligence-brief-architecture.md](personal-intelligence-brief-architecture.md) (especially "Prompt Responsibilities") and [docs/current-project-status.md](current-project-status.md) in full. Read `~/.claude/projects/-Users-merlin-clawfeed/memory/MEMORY.md` and the `vmlx_environment.md` it points to. Skim `worker/clawfeed_intel/llm/client.py`, `worker/clawfeed_intel/pipeline/orchestrator.py`, `worker/clawfeed_intel/pipeline/cluster.py`, `worker/clawfeed_intel/runs.py`, `worker/clawfeed_intel/db.py`, `migrations/010_intel_pipeline.sql` (lines 110–145 for `item_summaries` and `llm_calls`), `config/intel-sources.yaml`, and one existing test (`worker/tests/test_llm_client.py` for the LLM-stage testing style).

Then summarize back to the user and propose your plan for step 9a.
