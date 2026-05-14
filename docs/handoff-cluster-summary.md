# Handoff — Step 10: Cluster Summary

**Audience:** the engineer (likely a fresh Claude session) picking up Phase 1 of the Personal Intelligence Brief System after step 9 + four follow-up fixes have landed and been pushed.

**Repo:** `/Users/merlin/clawfeed` · branch `main` · remote `origin` = `https://github.com/Charliemorrone/the-front-page.git` (fully synced through commit `5715f5e`).

---

## Your role

You are a senior engineer continuing implementation of the Personal Intelligence Brief System for The Front Page (the user's personal fork of ClawFeed). Phase 1 is mid-stream. The relevance-filter stage (step 9) shipped across three commits (9a, 9b, 9c) plus four follow-up fixes from an audit cycle. A full end-to-end live-vMLX smoke against `mlx-community/Qwen3.5-27B-4bit` published digest 1 successfully — 242 kept clusters of 674 over a 24h window in ~63 minutes. You are picking up immediately after that with the relevance filter validated end-to-end, ready for the next intelligence stage.

Your job is to ship **step 10: cluster summary** — the LLM stage that turns each kept cluster into a structured summary that the final composer (step 11) will weave into the brief. When that lands, every cluster at `status='kept'` produces one `item_summaries` row carrying the architecture-doc's summary contract (headline, summary, why-it-matters, entities, key facts, caveats, source URLs, confidence), and the cluster's status advances to `'summarized'`.

---

## Hard requirements (non-negotiable, carried forward)

- Single-user, local-only.
- Final daily brief quality is the primary goal.
- All eight daily fetchers and the relevance filter are already shipped and working.
- **Final composition must use OpenClaw/`gpt-5.3-codex` with local `Qwen3.5-122B-A10B-4bit` fallback** (or whatever flagship is cached on the machine). Step 11 implements this; step 10 does not.
- The brief includes all relevant kept items, not a top-N subset.
- Failed sources / failed batches / failed clusters degrade coverage; they do not fail the run.
- ClawFeed (now bundled into The Front Page) remains the product shell.

**The Option-A scope rule still applies:**

- **Frontier cloud (`gpt-5.3-codex` via OpenClaw) is strictly bounded to the final-compose stage only.** Local vMLX handles everything else — relevance filter, **cluster summary**, source planning. This is non-negotiable. Routing 200+ kept clusters through frontier would cost real money for high-volume work the architecture explicitly reserves for local inference.
- **Local tokens are unbounded.** Don't ration vMLX calls; the cost discipline is "no paid tokens for high-volume work," not "make as few local calls as possible." Phase 1 routes `cluster_summary` to `mlx-community/Qwen3.5-27B-4bit` (the current Phase-1 placeholder; the doc target is `Qwen3.5-122B-A10B-4bit` once downloaded).
- **OpenClaw enters only at step 11.** Step 10 is vMLX-exclusive.

## Engineering hygiene (also non-negotiable)

- `git status` before editing.
- Small, inspectable steps. One concern per commit. Don't bundle unrelated work.
- `uv run pytest` (from `worker/`) and `uv run ruff check . && uv run ruff format --check .` must stay clean. **Current baseline: 692/692 tests passing.**
- Update `docs/current-project-status.md` after each substantive change — Built / In progress / Build log / new risks. Follow the "How to keep this file current" section at the top.
- Don't overwrite unrelated existing changes.
- Masterful code quality: type hints, focused docstrings explaining *why* (not *what*), no premature abstraction, no comments restating code, validation at boundaries only.
- Trust internal code; validate at system boundaries (HTTP, user input, file I/O). Don't add error handling for scenarios that can't happen.
- **Don't push to origin without explicit user instruction.** The recent push to origin was a deliberate operator-initiated action; future commits stay local until told otherwise.

---

## Required reading, in order

Before you write any code, read these and confirm you understand them:

1. **`docs/personal-intelligence-brief-architecture.md`** — full architecture, data model, model routing, phased build plan, per-stage specifications. **The load-bearing sections for step 10 are:**
   - "Prompt Responsibilities" → "Cluster Summary" — the input shape (one cluster + supporting items + metadata) and the output JSON contract (`headline`, `summary`, `why_it_matters`, `entities`, `key_facts`, `caveats`, `source_urls`, `confidence`).
   - "Data model" → "`item_summaries`" — the row-per-kept-cluster table you'll write to. Already created by migration 010 lines 110–126.
   - "Frontier Model Use Is Strictly Bounded" — the rule that keeps step 10 on vMLX.
   - "Reliability Requirements / Inference Reliability" — failure handling.

2. **`docs/current-project-status.md`** — source of truth for current build state. The "Built" section shows everything shipped to date; the build log is an append-only audit trail starting at line ~338 (newest first). Read at least the entries for steps 9a/9b/9c and the four follow-up fixes — they document the *lessons* that should shape your step-10 implementation (see "Specific design guidance" below).

3. **The relevance filter, as the reference pattern.** Step 10 mirrors step 9 in structure. Read these to understand the existing pattern you'll copy:
   - `worker/clawfeed_intel/pipeline/relevance.py` — full two-layer split (pure prompt/parse helpers + async orchestration).
   - `worker/clawfeed_intel/llm/schemas.py` — where `RelevanceVerdict` + `RelevanceBatchResponse` live. Your `ClusterSummaryPayload` belongs here.
   - `worker/clawfeed_intel/llm/client.py` — the chokepoint. The `LLMClient.chat_completion(stage, messages, *, response_schema, prompt_version, temperature, max_tokens)` signature is what you'll call.
   - `worker/clawfeed_intel/pipeline/orchestrator.py` — `_drive_run` walks the lifecycle states. You'll fill in the `summarizing` stage between `filtering` (now real) and `composing` (still `_compose_stub` until step 11).
   - `worker/clawfeed_intel/db.py` — the `update_cluster_verdict` helper is the closest analog to the `create_item_summary` helper you'll add.
   - `worker/clawfeed_intel/runs.py` — `Coverage.failed_filter_batches` is the analog to the `failed_summary_clusters` counter you'll add.

4. **`~/.claude/projects/-Users-merlin-clawfeed/memory/MEMORY.md`** and the `vmlx_environment.md` memory file it indexes. Verified facts about the running vMLX server: port, loaded model, on-disk inventory, multi-model hot-swap behavior. **Verify the memory still matches reality** before relying on it (vMLX may have been restarted; the user may have downloaded new models). The architecture-doc-target flagship `Qwen3.5-122B-A10B-4bit` may still be missing from disk; Phase 1 uses `Qwen3.5-27B-4bit` as the placeholder.

5. **Skim one representative existing test for style.** `worker/tests/test_relevance.py` is the closest analog: pure-helper tests plus orchestration tests against `LLMClient` backed by `httpx.MockTransport`. Mirror its structure for the cluster-summary equivalents. **No live vMLX in CI** — mock at the HTTP boundary; a manual smoke against live vMLX is a separate one-shot action.

6. **`config/model-routing.yaml`** — the `cluster_summary` stage is already configured (Phase 1 placeholder: `mlx-community/Qwen3.5-27B-4bit`, 300s timeout). No routing change required for step 10.

7. **`migrations/010_intel_pipeline.sql`** lines 110–126 — the `item_summaries` table shape. No migration change required; the schema is ready.

---

## After reading, summarize back to the user

Confirm you understand:

1. The current build state (Phase 1 step 9 done, four follow-up fixes done, full live smoke passing, 692/692 tests pass, repo synced to origin).
2. The architecture-doc spec for cluster summary — input shape and output JSON contract.
3. The Option-A scope rule: frontier-only-for-compose, local-vmlx-for-everything-else; step 10 is vMLX-exclusive.
4. The two lessons from step 9 that should shape your step 10:
   - Write schemas **permissive from the start** (9c). Local models emit `null` for narrative fields under load; making them required forces the bounded repair-retry and still likely fails. The architecture-doc field list specifies *what* the schema covers, not *which fields are required* — that's an engineering judgment informed by what the model reliably emits.
   - Size `max_tokens` at the call site (9b). vMLX defaults to ~1024 completion tokens; cluster summaries can be 400-800 tokens of structured output, so pin a generous budget per call. `temperature=0.1` is appropriate for JSON-mode prompts.
5. Where the new stage slots into the orchestrator state machine (inside `_drive_run`'s `summarizing` state, after relevance filtering, reusing the same `LLMClient` instance already constructed for filtering).
6. Your proposed plan for the substep you are going to ship first (10a or 10b — see below).

**Wait for the user to confirm your summary before you start writing code.**

---

## Where the previous engineer left off

- **Phase 1 step 9 (relevance filter) is complete.** Three commits on origin/main:
  - `59bc7d3` — 9a: schemas + pure helpers + verdict DB helper
  - `2909bb6` — 9b: async `filter_clusters` + orchestrator wiring + temperature/max_tokens plumbing
  - `f7dd438` — 9c: schema robustness — `category` and `reason` accept `null`
- **Four follow-up fixes** from an audit cycle, also on origin/main:
  - `51d26b5` — fixed broken default GitHub-search queries (no longer 422)
  - `3c7845a` — preflight vMLX health check in `clawfeed-intel run daily`
  - `0c5d649` — dashboard UI for tagging sources to intel categories (Node-side: db.mjs + server.mjs + web/index.html)
  - `5715f5e` — SSRF protection (`validate_safe_url`) on worker RSS top-level / trafilatura-article / website fetch paths
- **Live-vMLX smoke (2026-05-12)** passed end-to-end: 688 raw items → 674 clusters (L1+L2+L3) → 242 kept (36%) + 420 filtered_out + 12 pending (one batch failed count mismatch and degraded coverage as designed) → digest 1 published. 57/57 LLM HTTP calls succeeded; 0 retries fired; ~28-30 tps generation on the 27B-4bit. See the build log entry for full detail.
- **Test suite: 692/692 pass; ruff check + format clean.**
- The orchestrator's `summarizing` lifecycle stage currently advances status and does nothing else (look at `_drive_run` in `pipeline/orchestrator.py`). The `composing` stage runs `_compose_stub` to publish skeleton markdown — step 11 replaces that. Your work fills in `summarizing`.

**State of the world:**

- Repo: `/Users/merlin/clawfeed`, branch `main`, fully synced to origin.
- All 11 intel pipeline tables are live (migrations 010 + 011).
- vMLX is verified running on `127.0.0.1:8080`; on-disk model inventory in the memory file. Architecture-doc flagship `Qwen3.5-122B-A10B-4bit` is the doc-target; Phase 1 routes `cluster_summary` to `Qwen3.5-27B-4bit` (15G, on disk) as the placeholder. The user can swap by editing `config/model-routing.yaml`; no code change required.
- Python venv provisioned via `uv sync --extra dev`. `uv run pytest -q` and `uv run ruff check .` are the validation commands.

---

## What step 10 is

Step 10 makes the brief possible. Without it, the orchestrator has a list of `'kept'` clusters but no condensed text for the final composer to weave together. The cluster-summary stage:

- Loads every cluster at `status='kept'` for the run, along with all member items (title, URL, excerpt, content).
- Calls vMLX once per cluster with a JSON-mode prompt asking for the architecture-doc's summary shape.
- Validates the response against `ClusterSummaryPayload` (your new pydantic schema).
- Writes one row to `item_summaries` (cluster_id, model, prompt_version, the JSON fields).
- Advances the cluster's status to `'summarized'`.
- Per-cluster failure: leaves the cluster at `'kept'`, increments `coverage.failed_summary_clusters`, continues with the next cluster. The brief just won't include the unsummarized cluster.

This is conceptually the same as the relevance filter except **one call per cluster** instead of one call per batch — each cluster gets its own prompt because the model needs the cluster's full member items as context, which doesn't share well across batches.

### Suggested split into substeps

Mirror the 9a/9b rhythm:

- **Step 10a — schemas + pure helpers + DB helper.** Create `ClusterSummaryPayload` in `llm/schemas.py` (write it permissive per the 9c lesson). Pure `build_summary_messages(cluster_with_items, category)` and `parse_summary(parsed)` helpers in a new `pipeline/summary.py`. New `db.create_item_summary(...)` helper validating required fields and serializing the JSON-list fields (`entities`/`key_facts`/`caveats`/`source_urls`) to text-column-compatible JSON strings (the schema stores them as TEXT). Tests cover schemas + prompt construction + parsing + DB writes. No orchestration yet.

- **Step 10b — async `summarize_clusters` + orchestrator wiring + Coverage + live-vMLX smoke.** Add the async function that loads `'kept'` clusters with their members, iterates one chat-completion per cluster against the `cluster_summary` stage with the schema, writes the row, advances status to `'summarized'`. Add `Coverage.failed_summary_clusters` counter. Wire into `_drive_run`'s `summarizing` stage. Stamp `metadata.local_models["summary"]` from the stage config. Tests cover end-to-end orchestration with a stub LLM client + per-cluster-failure path. Manual `clawfeed-intel run daily --window 24h` smoke against real vMLX should produce `item_summaries` rows; capture the count for the build log.

Each substep is one commit, gated on green tests + ruff. **Stop between substeps and check in with the user.** That cadence served step 9 well — the user wants to inspect each unit.

---

## Specific design guidance (lessons from steps 9a / 9b / 9c)

- **Two-layer pattern.** Mirror the relevance filter exactly. Pure helpers (prompt construction, response parsing, DB writes) are testable from fixtures; thin async orchestration on top threads `LLMClient` and `sqlite3.Connection`. The async layer doesn't do business logic — it just sequences calls and persists results.

- **Write the schema permissive.** This is the 9c lesson and it's load-bearing. The architecture doc lists `headline`, `summary`, `why_it_matters`, `entities`, `key_facts`, `caveats`, `source_urls`, `confidence` as the summary fields. Of those, `headline` and `summary` are the load-bearing signals (the brief needs both to render the cluster). The rest are narrative additions that local models will sometimes emit as `null` or empty arrays. Default everything except `headline` and `summary` to optional with reasonable defaults (`""` for strings, `[]` for lists, `None` for `confidence`). When the architecture-doc-target flagship lands later, you can consider tightening — but for Phase 1's Qwen3.5-27B-4bit, permissive wins. Add a regression-guard test that exercises a payload with everything-except-headline-and-summary as null.

- **Size `max_tokens` at the call site.** This is the 9b lesson. vMLX defaults to ~1024 tokens. Cluster summaries can run 400-800 tokens of structured output. Pin `max_tokens` at something generous (e.g., 1500-2000) so the schema-repair retry doesn't fire on truncation. Pin `temperature=0.1` for JSON-mode prompts (consistent with the relevance filter). Both are now keyword arguments on `LLMClient.chat_completion`.

- **Prompt versioning.** `PROMPT_VERSION = "summary.v1"`. Bump on behavioral changes. The `llm_calls.prompt_version` column already supports this.

- **One LLM client, reused.** The orchestrator already constructs an `LLMClient(routing, conn=conn, run_id=run_id)` for the filtering stage. Don't construct another. Pass the same instance into `summarize_clusters`. Every cluster's call automatically writes one row to `llm_calls`.

- **Per-cluster failure model.** The relevance filter degrades **per batch**. Cluster summary degrades **per cluster** — that's the right granularity because each call is independent. If a summary call raises (transient HTTP, schema validation after repair, response parse error), log + increment `coverage.failed_summary_clusters` + leave the cluster at `'kept'` + continue with the next cluster. Sibling clusters in the same orchestration call are unaffected.

- **Architecture-doc rule that's load-bearing:** every cluster summary must be **grounded in its source items**. Don't include speculation or claims that aren't supported by the member content. The prompt should explicitly tell the model: "preserve citations to the member URLs," "do not invent facts not present in the input." This becomes part of the system message.

- **Pending-only filter.** Like the relevance filter, write your loader to filter by `status='kept'` in SQL. A re-run picks up only un-summarized clusters; previously-summarized clusters keep their `item_summaries` row and don't get re-processed. Replay-safe.

- **Schemas live in `llm/schemas.py`.** Not in the pipeline-stage module. Keep them centralized so the LLM client and the pipeline stage import the same definition. `model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)` matches the existing schema posture. Bounded numerics for `confidence` (`Field(ge=0, le=1)`).

- **Where to call the LLM client.** Construct once in `_drive_run` (already done for filtering). Pass it down to each stage. Don't construct it inside the pure helpers. Don't construct it inside per-cluster loops.

- **Don't import from `fetchers/`.** The LLM stages are independent of the fetcher subsystem. Same architecture rule that kept the LLM client decoupled.

- **Docstrings explain WHY.** Not what. Not "summarizes a cluster" — instead "summary fields beyond headline + summary are permissive because local models emit null on rejected items; tightening would force the schema-repair retry on every batch and still likely fail. See step 9c for the precedent." Future-you reading the docstring should understand the trade-off, not just the surface behavior.

## Don't

- Don't build OpenClaw integration. Step 11 owns that.
- Don't expand `intel_runs.status` for sub-states. LLM calls happen *inside* existing states.
- Don't store full prompt or response text in `llm_calls`. Hashes only — enforced by the column shape, but don't add new columns either.
- Don't add streaming completions. Phase 1 is synchronous.
- Don't expand the `digests.type` CHECK constraint. Not relevant to step 10 — listed for completeness.
- Don't try to live-vMLX-test in CI. Mock at the HTTP boundary; manual smoke against live vMLX is a separate one-shot action.
- Don't add new migrations. The `item_summaries` table is already correct in migration 010.
- Don't refactor the fetcher contract, the clustering layer, the LLM client, the relevance filter, or the SSRF guard. They work. Touching them adds risk for no benefit.
- Don't invent new schemas or contracts. Use the architecture doc's "Prompt Responsibilities → Cluster Summary" section as the spec.
- Don't bundle 10a and 10b. Two commits, each gated on green tests + ruff.
- Don't push to origin without explicit user instruction.

## Pattern to mirror (from every shipped step)

Every shipped step follows this rhythm:

1. Read the architecture doc + status doc + memory + a representative existing module.
2. Summarize back to the user; propose the substep plan.
3. Wait for approval.
4. Implement: schema/data layer first if needed, then pure helpers, then async orchestration, then orchestrator integration, then tests, then status doc update.
5. `uv run pytest -q` + `uv run ruff check .` + `uv run ruff format --check .` must all be clean.
6. Commit with a descriptive message (`feat(intel): …` matching the prior cadence).
7. Stop and check in with the user before the next substep.

For step 10b's live smoke specifically: budget time. The 27B-4bit at one call per kept cluster runs ~10-60 seconds per call depending on summary length. A run with 200+ kept clusters can take an hour-plus end to end. The relevance-filter smoke completed in ~63 minutes wall-time over a 24h window; expect step 10's smoke to add similar wall-time. Use `--window 4h` against the smoke DB if you want a quicker validation pass during iteration, then a full 24h smoke for the acceptance capture.

## Open risks carried forward

| Risk / decision | Status | Notes |
|---|---|---|
| Architecture-doc flagship model not yet on disk | open | `Qwen3.5-122B-A10B-4bit` (~70G) and `Qwen3.6-35B-A3B` not downloaded. Phase 1 uses `Qwen3.5-27B-4bit` (15G, on disk). Configurable via YAML. |
| Local LLM JSON reliability under load | open | Mitigated by pydantic schemas + `LLMClient`'s bounded JSON repair. Step 9c proved schema permissiveness is the right structural defense. Step 10 will exercise this at hundreds-of-calls scale. |
| OpenClaw gateway wire protocol unknown | open until step 11 | Not your problem in step 10. |
| Two-runtime SQLite access (Node + Python) | accepted | WAL on; worker writes use `BEGIN IMMEDIATE` and short transactions. LLM-stage DB writes follow the same discipline. |
| Local Node binary has a broken simdjson dyld link | environmental | System-level brew issue; running ClawFeed server uses a working node. Doesn't affect Python worker tests. |

---

## Begin

1. Read `docs/personal-intelligence-brief-architecture.md` (especially "Prompt Responsibilities → Cluster Summary" and "Data model → `item_summaries`") in full.
2. Read `docs/current-project-status.md` in full — pay attention to the recent build-log entries for 9a/9b/9c and the four follow-up fixes.
3. Read `~/.claude/projects/-Users-merlin-clawfeed/memory/MEMORY.md` and the `vmlx_environment.md` it points to. Verify against reality.
4. Skim `worker/clawfeed_intel/pipeline/relevance.py`, `worker/clawfeed_intel/llm/schemas.py`, `worker/clawfeed_intel/llm/client.py`, `worker/clawfeed_intel/pipeline/orchestrator.py`, `worker/clawfeed_intel/db.py` (especially the `update_cluster_verdict` helper and the `iter_pending_clusters_with_members` helper), `worker/clawfeed_intel/runs.py` (the `Coverage` dataclass), `config/intel-sources.yaml`, `config/model-routing.yaml`, and `migrations/010_intel_pipeline.sql` lines 110–126.
5. Skim one existing test for the LLM-stage testing style: `worker/tests/test_relevance.py`.

Then summarize back to the user and propose your plan for step 10a.
