# Personal Intelligence Brief — Current Project Status

**Source of truth for engineers building this system. Update after every critical change.**

- Repo: `/Users/merlin/clawfeed`
- Last updated: 2026-05-04 (Phase 1 step 6.4 — SEC EDGAR fetcher)
- Last update by: implementation engineer (Claude)
- Authoritative design docs:
  - [personal-intelligence-brief-architecture.md](personal-intelligence-brief-architecture.md) — full architecture
  - [how-the-daily-brief-works.md](how-the-daily-brief-works.md) — plain-English walkthrough

---

## How to keep this file current

1. Update **Last updated** and **Last update by** at the top.
2. Move completed items from "In progress" / "Next up" into "Built".
3. If you discovered a new risk, add it to "Open risks & decisions".
4. If you changed scope, note it in "Scope changes & decisions".
5. Keep entries terse. One bullet per change. Link to files with `[name](path)`.
6. Do **not** delete history from "Build log" — append, don't overwrite. That section is the audit trail.

---

## Phase scope

The build target for this slice is the **first end-to-end daily brief that publishes a real digest**, with all eight fetchers and frontier+local-fallback composition. We are calling this "Phase 1" per the user's hard requirements, even though the architecture doc subdivides it further.

Hard requirements that must be preserved at all times:

- Single-user, local-only personal system.
- Final daily brief quality is the primary goal.
- All eight daily fetchers must be in Phase 1: RSS, arXiv, HN, Reddit, GitHub w/ velocity, GDELT, SEC EDGAR, websites.
- GitHub "repos gaining traction" must use real velocity from stored observations, not Trending alone.
- Final composition must use OpenClaw/`gpt-5.3-codex` with local fallback (`Qwen3.5-122B-A10B-4bit`).
- The brief must include all relevant kept items — not a top-N subset.
- Failed sources degrade coverage; they do not fail the whole run.
- ClawFeed remains the product shell: dashboard, SQLite, digest storage, source manager, API.

---

## Built (✅)

### Schema + migration loader
- [migrations/010_intel_pipeline.sql](../migrations/010_intel_pipeline.sql) — 10 pipeline tables, all idempotent: `intel_runs`, `intel_jobs`, `raw_items`, `run_raw_items`, `item_clusters`, `cluster_items`, `item_summaries`, `llm_calls`, `source_fetch_state`, `source_categories`.
- [src/db.mjs](../src/db.mjs#L94-L100) — migration 010 wired into the existing migration loader chain; matches the 008/009 pattern.
- Verified: live DB has all 10 tables; re-applying the migration is a no-op.

### Worker scaffold
- [worker/pyproject.toml](../worker/pyproject.toml) — Python 3.12 + `uv`. Runtime deps: `httpx`, `feedparser`, `trafilatura`, `selectolax`, `pydantic`, `tenacity`, `PyYAML`, `python-dotenv`. Dev deps: `pytest`, `pytest-asyncio`, `ruff`. Console script `clawfeed-intel`. `pytest pythonpath = ["."]` so tests run without an editable install.
- [worker/clawfeed_intel/__init__.py](../worker/clawfeed_intel/__init__.py) — `__version__ = "0.1.0"`.
- [worker/clawfeed_intel/paths.py](../worker/clawfeed_intel/paths.py) — repo-root-relative paths; `DIGEST_DB` env var override supported.

### Run lifecycle skeleton
- [worker/clawfeed_intel/db.py](../worker/clawfeed_intel/db.py) — `connect()`, `transaction()` context manager (BEGIN IMMEDIATE + busy_timeout), `RunStateError`. Run CRUD: `create_run`, `get_run`, `mark_run_started`, `advance_run_status`, `update_run_metadata`, `finish_run`. Digest insert: `create_digest`. Status enums (`RUN_TYPES`, `RUN_STATUSES`, `INTERMEDIATE_RUN_STATUSES`, `TERMINAL_RUN_STATUSES`, `DIGEST_TYPES`) mirror SQL CHECK constraints and validate at the boundary.
- [worker/clawfeed_intel/timewindow.py](../worker/clawfeed_intel/timewindow.py) — `parse_window`, `now_utc`, `to_iso`, `window_for`. All persisted timestamps are tz-aware UTC ISO 8601 (`YYYY-MM-DDTHH:MM:SS+00:00`).
- [worker/clawfeed_intel/runs.py](../worker/clawfeed_intel/runs.py) — `Coverage` and `RunMetadata` dataclasses. JSON shape matches the architecture doc's digest-metadata example exactly.
- [worker/clawfeed_intel/pipeline/orchestrator.py](../worker/clawfeed_intel/pipeline/orchestrator.py) — `run_daily(window_spec, conn=None)`. Walks `pending → fetching → filtering → summarizing → composing → published`. Failures route to `finish_run(status='failed', error=...)` so a partial run is observable. Stage call sites are `# TODO`-marked for steps 4–11.
- [worker/clawfeed_intel/cli.py](../worker/clawfeed_intel/cli.py) — `clawfeed-intel doctor` (now reports intel-table count), `clawfeed-intel run daily --window 24h` (real, prints `published digest <id>`).

### Normalization + raw_items upsert
- [worker/clawfeed_intel/normalize.py](../worker/clawfeed_intel/normalize.py) — `canonicalize_url` (scheme/host case fold, `www.`/`amp.` strip, default-port strip, fragment drop, `/amp` path strip, trailing-slash strip on non-root, tracking-param drop covering UTM/click-id/Mailchimp/HubSpot/share-ref/etc., remaining-param sort, idempotent). `content_hash` (SHA-256 of normalized title + first 4000 chars of body, whitespace collapsed and case folded). Per-source dedup-key helpers `hn_dedup_key`, `reddit_dedup_key`, `github_dedup_key`, `arxiv_dedup_key`, `sec_dedup_key` enforcing canonical-form-per-source.
- [worker/clawfeed_intel/db.py](../worker/clawfeed_intel/db.py) — added `upsert_raw_item(...)` (single transaction: `INSERT ... ON CONFLICT(source_type, dedup_key) DO NOTHING RETURNING id` plus `INSERT OR IGNORE INTO run_raw_items`; returns `(raw_item_id, was_new)`; preserves first-sight `raw_items.run_id` across runs while linking the current run via `run_raw_items`) and `link_raw_item_to_run(...)` (idempotent, returns whether a new link was created; FK violations propagate as `IntegrityError`).

### Source plan resolver
- [config/intel-sources.yaml](../config/intel-sources.yaml) — starter editorial config: `profile`, four categories (`startup_funding`, `ai_research`, `ai_coding_tools`, `github_traction`) with `description`/`include`/`exclude`/`sources`, plus `dynamic_search.enabled_sources` carried forward to Phase 7. Uses `kind:` (not `type:`) as the structured-source discriminator to avoid clashing with ClawFeed's `sources.type` column.
- [worker/clawfeed_intel/sources.py](../worker/clawfeed_intel/sources.py) — `build_source_plan(conn, *, config_path=None) -> SourcePlan`. Pydantic discriminated union (`RssTask`, `ArxivTask`, `HnTask`, `RedditTask`, `GdeltTask`, `SecEdgarTask`, `GithubSearchTask`, `GithubTrendingTask`, `WebsiteTask`) validates YAML at the boundary with `extra="forbid"`. DB join: `sources` filtered by `is_active = 1`, joined via `source_categories`, mapped through `_DB_TYPE_TO_KIND` (`rss`/`atom`→rss, `website`→website, `hackernews`/`hn`→hn, `reddit`→reddit, `github_trending`→github_trending; Twitter intentionally absent). Translates ClawFeed-historic config keys (HN `filter` → task `list`; github_trending `language: 'all'` → `None`). Soft-failure model: missing config / single bad entry / unknown DB type → `PlanWarning`, never raises. Hard-failure: top-level YAML not a mapping → raises (deploy bug, not a degradation). `SourcePlan.tasks_by_kind()` groups tasks across categories so fetchers see one bucket each.

### Fetcher harness + orchestrator wiring
- [worker/clawfeed_intel/fetchers/base.py](../worker/clawfeed_intel/fetchers/base.py) — `FetchedItem` dataclass (matches `db.upsert_raw_item` parameters via `upsert_kwargs()`); `FetchOutcome` with `status` ∈ `{succeeded, failed, skipped}`, items_seen, items_new, latency_ms, error; `FetcherCallable = Callable[[ResolvedTask], Awaitable[list[FetchedItem]]]`; module-level `FETCHER_REGISTRY` for production wiring (concrete fetcher modules will register at import time).
- [worker/clawfeed_intel/fetchers/runner.py](../worker/clawfeed_intel/fetchers/runner.py) — `async run_fetch_stage(conn, *, run_id, plan, coverage, fetchers=None) -> list[FetchOutcome]`. Tasks of the same kind run concurrently via `asyncio.gather`; fetcher exceptions become `failed` outcomes (sibling tasks continue); kinds with no registered fetcher become `skipped` outcomes. Per-item upsert failures are logged but don't blackhole the batch. Plan warnings (`PlanWarning`) flow into `Coverage.plan_warnings`. SQLite writes serialize naturally on the single-threaded event loop, so concurrency only fans out HTTP, not DB.
- [worker/clawfeed_intel/fetchers/http.py](../worker/clawfeed_intel/fetchers/http.py) — shared `httpx.AsyncClient` factory (`build_client()` async-context-manager), polite `User-Agent` `ClawFeed-Intel/<ver> (+contact: <email>)` with email from `CLAWFEED_CONTACT_EMAIL` env (SEC EDGAR / Reddit explicitly require this), default timeouts and connection limits. Used by RSS today; arXiv/HN/SEC/GDELT/Reddit/GitHub/website fetchers will reuse it.

### RSS / Atom fetcher
- [worker/clawfeed_intel/fetchers/rss.py](../worker/clawfeed_intel/fetchers/rss.py) — registers `kind="rss"` in `FETCHER_REGISTRY`. Two-layer design: `parse_feed_text(text, *, source_name, feed_url)` is pure (feedparser → `FetchedItem`s), and `async fetch_rss(task)` wraps it with the HTTP fetch + trafilatura fallback. Per entry: `dedup_key = canonical_url`, content stripped of HTML via selectolax, excerpt = first 320 chars, `published_at` prefers `<published>` over `<updated>` (so re-emitted entries don't appear new on every run), Atom `<content>` wins over `<summary>` (richer body), tags surfaced into metadata. Trafilatura fallback fires when summary text is below 400 chars; failure is best-effort (returns `None`, runner keeps the original summary).

### arXiv fetcher
- [worker/clawfeed_intel/fetchers/arxiv.py](../worker/clawfeed_intel/fetchers/arxiv.py) — registers `kind="arxiv"` in `FETCHER_REGISTRY`. Hits `https://export.arxiv.org/api/query` once per task with the task's categories `OR`-joined as `cat:cs.AI OR cat:cs.LG …`, sorted `submittedDate desc`, `max_results=500`. Two layers: pure `parse_atom_response(text, *, source_name, query_url)` and async `fetch_arxiv(task)`. Stdlib `xml.etree.ElementTree` is used directly (no defusedxml needed — arXiv is trusted; no feedparser — its handling of the `arxiv:` namespace is version-flaky). Per entry: `dedup_key = arxiv_dedup_key("<id>v<n>")` keeps revisions distinct (clustering will recognize them as the same logical paper); abstract becomes `content` (no trafilatura fallback — abstracts are already terse); whitespace collapsed in title and summary; authors joined with `, ` for `author` and preserved as a list in `raw_payload`. Metadata includes `arxiv_id`, `primary_category`, all `categories` (primary first, deduped), `abs_url`, `pdf_url` if present, `doi` if present, `query_url`. Both modern (`2405.12345v1`) and legacy (`math/0506203v2`) ID layouts handled.

### Hacker News fetcher
- [worker/clawfeed_intel/fetchers/hn.py](../worker/clawfeed_intel/fetchers/hn.py) — registers `kind="hn"` in `FETCHER_REGISTRY`. Uses the Firebase API (`https://hacker-news.firebaseio.com/v0`) per the architecture doc; Algolia is reserved for topical search (Phase 7). Per task: GET the list endpoint (`topstories.json`/`beststories.json`/`newstories.json`/`showstories.json`/`askstories.json`) → truncate to `task.limit` (default 200) → fan out item-detail fetches under `asyncio.Semaphore(10)` → optionally drop items below `task.min_score`. Two layers: pure `parse_hn_item(raw, *, list_name)` and async `fetch_hn(task)`. Per item: `dedup_key = hn_dedup_key(<id>)`; `canonical_url` from external URL when present, else the discussion URL `https://news.ycombinator.com/item?id=<id>` (Ask HN pattern); `text` (HTML-encoded body for Ask/Show) becomes `content` after selectolax strip; `published_at` from unix `time` epoch in UTC. Metadata carries `hn_id`, `list`, `type`, `score`, `descendants`, `discussion_url`, optional `external_url`. `raw_payload` strips `kids` and `parts` to avoid blob bloat from comment trees / poll options. Deleted, dead, comment-type, and untitled items are skipped. `topstories` is treated as an attention snapshot, not a time window — no client-side `published_at` filter.

### SEC EDGAR fetcher
- [worker/clawfeed_intel/fetchers/sec.py](../worker/clawfeed_intel/fetchers/sec.py) — registers `kind="sec_edgar"` in `FETCHER_REGISTRY`. Uses the legacy `getcurrent` Atom endpoint (`https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=<F>&output=atom`) — stable for over a decade, returns the most recent filings of a given form type. Modern `data.sec.gov` JSON API is per-CIK only and is reserved for Phase 7 topical search. Two layers: pure `parse_atom_response` (stdlib `xml.etree.ElementTree`) + async `fetch_sec`. One request per form in `task.forms`, fanned out concurrently, results merged and deduplicated by accession number. `dedup_key = sec_dedup_key(accession_number)` — the SEC-canonical, globally unique filing identifier. Title parsed for `<FORM> - <COMPANY> (<CIK>) (Filer)`; on parse failure the entry is kept (accession is enough to dedup) with empty `company_name`/`cik` and the requested form copied into `form_type`. Metadata: `accession_number`, `filing_url`, `form_type` (from title), optional `requested_form` (when amendments slip through), `company_name`, `cik`, `query_url`. `published_at` normalized from `-04:00`/`-05:00` to UTC. **Partial-failure semantics**: if one of `["D", "D/A"]` fails and the other succeeds, the successful items are returned (a noisy SEC outage shouldn't drop the filings we did retrieve); if all forms fail, the first exception propagates so the runner records the task as failed. `task.ciks` field is reserved for Phase 7 — daily brief is form-only.
- [worker/clawfeed_intel/fetchers/__init__.py](../worker/clawfeed_intel/fetchers/__init__.py) — imports `arxiv`, `hn`, `rss`, and `sec` to trigger registration.
- [worker/clawfeed_intel/runs.py](../worker/clawfeed_intel/runs.py) — `Coverage` extended with `skipped_sources: list[str]` and `plan_warnings: list[str]`, plus `record_skipped(source_id, reason)` and `record_plan_warning(message)` helpers. `record_skipped` counts as attempted (we wanted to try) but lives on its own list to keep "tried-and-broke" distinct from "harness-incomplete" in the brief.
- [worker/clawfeed_intel/db.py](../worker/clawfeed_intel/db.py) — added `record_fetch_success(...)` (resets `consecutive_errors` to 0, clears `last_error`, optional cursor/metadata preserved if not supplied via COALESCE) and `record_fetch_failure(...)` (bumps `consecutive_errors` by 1, never touches `last_success_at` or `cursor` so a flaky stretch retains its last-known-good resumption point).
- [worker/clawfeed_intel/pipeline/orchestrator.py](../worker/clawfeed_intel/pipeline/orchestrator.py) — `_drive_run` now resolves the plan via `build_source_plan(conn)` and runs the fetch stage via `asyncio.run(run_fetch_stage(...))` during the `fetching` lifecycle state. With no fetchers registered yet, every resolved task records honestly as `skipped: no fetcher for kind <X>`, which proves the resolver→harness→storage→coverage spine end-to-end without yet doing any HTTP.

### Tests
- [worker/tests/conftest.py](../worker/tests/conftest.py) — `temp_db` fixture applies all migrations to a fresh DB; tests never touch `data/digest.db`.
- [worker/tests/test_timewindow.py](../worker/tests/test_timewindow.py) — 18 tests: window parsing edge cases, naive-datetime guards, UTC normalization.
- [worker/tests/test_db.py](../worker/tests/test_db.py) — 13 tests: lifecycle happy path, illegal transitions, transaction rollback, metadata persistence.
- [worker/tests/test_orchestrator.py](../worker/tests/test_orchestrator.py) — 3 tests: end-to-end run, invalid window, mid-flow failure marks `intel_runs.status='failed'` with no orphan digest.
- [worker/tests/test_normalize.py](../worker/tests/test_normalize.py) — 42 tests (48 cases inc. parametrize): canonicalize_url across scheme/host/port/path/query/fragment/userinfo, every tracking-param family, idempotency, duplicate-form folding, type errors, content_hash collision/divergence properties, prefix truncation, per-source dedup-key normalization.
- [worker/tests/test_raw_items.py](../worker/tests/test_raw_items.py) — 15 tests: was_new semantics, source_type-scoped conflict, run linkage, cross-run first-sight preservation, optional-field persistence, blank/FK validation, cascade on run delete (raw_items.run_id → NULL, run_raw_items rows removed).
- [worker/tests/test_sources.py](../worker/tests/test_sources.py) — 21 tests: full-yaml round-trip across all nine kinds; missing-config warning; empty-categories quiet; top-level corruption raises; unknown/missing/invalid kind soft-skip; non-mapping entry/category soft-skip; DB join under tagged category; lazy category materialization for unknown tag; inactive/untagged exclusion; unsupported `sources.type` warns; corrupt JSON config warns; missing required config warns; HN `filter`→`list` translation; github_trending `language: all` normalization; multi-category tagging; YAML+DB merge; default-config-path regression.
- [worker/tests/test_fetch_state.py](../worker/tests/test_fetch_state.py) — 8 tests: success creates row with consecutive_errors=0; failure creates row and increments on each call; success resets the counter; cursor + last_success_at preserved across failures; success preserves existing cursor when omitted; metadata replaced only when explicitly supplied; per-(source_id, fetcher) PK isolation; FK violation on unknown source_id.
- [worker/tests/test_fetchers_runner.py](../worker/tests/test_fetchers_runner.py) — 13 tests: stub fetcher persists items + updates coverage; run_raw_items linkage; YAML-origin tasks skip source_fetch_state; idempotent re-runs report items_seen=N/items_new=0; fetcher exceptions become failed outcomes with fetch_state recorded; one failure doesn't kill sibling tasks; per-item upsert failures don't blackhole the batch; missing-fetcher kind → skipped outcomes; skipped doesn't touch source_fetch_state; plan warnings flow into coverage; empty plan runs clean; concurrent dispatch within a kind verified via timing.
- [worker/tests/test_fetchers_rss.py](../worker/tests/test_fetchers_rss.py) — 17 tests: pure `parse_feed_text` over RSS 2.0 + Atom fixtures; tracking-param normalization in dedup_key; entries-without-link skipped; empty/garbage XML → empty list (no raise); `<published>` preferred over `<updated>` with fallback; HTML stripped from content; tags surfaced. Plus `httpx.MockTransport`-driven `fetch_rss` tests covering UA contact-info header, 4xx/5xx propagation (so the runner records `failed`), non-RSS task type guard, trafilatura fallback enriching thin summaries, fallback skipped when summary already long enough, fallback failure preserved as original summary, 4xx article URL → `None` from extractor. Registration sanity check.
- [worker/tests/test_fetchers_arxiv.py](../worker/tests/test_fetchers_arxiv.py) — 20 tests: pure `parse_atom_response` over hand-written Atom fixtures covering modern + legacy IDs (with version preserved in dedup_key), whitespace collapse in title/summary, multi-author join, primary + secondary categories deduped with primary first, `pdf_url` and `doi` recorded only when present, `published` UTC normalization, naive-timestamp-as-UTC fallback, empty feed, malformed XML → empty list, entries without `<id>` or alternate link, dedup_key being the arXiv ID rather than canonical URL. Plus MockTransport tests verifying query-URL construction (search_query OR-join, sortBy, sortOrder, start, max_results), single-category path, query_url stamped in metadata, 5xx propagation, non-arxiv task type guard. Registration sanity check.
- [worker/tests/test_fetchers_hn.py](../worker/tests/test_fetchers_hn.py) — 27 tests: pure `parse_hn_item` over hand-built dicts covering story+external URL, Ask HN (text body, HTML stripped, entities decoded, no external_url metadata), Show HN with both URL and text, deleted/dead/comment/untitled items dropped, missing-score/missing-descendants tolerance, non-dict input, `kids` stripped from raw_payload while `descendants` is preserved in metadata, invalid `time` → `published_at = None`. Plus MockTransport tests parametrized over all 5 list types (top/best/new/show/ask), `limit` truncation, `min_score` filter, JSON `null` item skipped, **per-item 5xx doesn't abort the batch**, list-endpoint 5xx propagates as failed task, list-endpoint non-array degrades to empty (rather than raise), UA contact-info verified, non-hn task type guard. Registration sanity check.
- [worker/tests/test_fetchers_sec.py](../worker/tests/test_fetchers_sec.py) — 18 tests: pure `parse_atom_response` over hand-written EDGAR Atom — accession extraction, title-shape parsing for form/company/CIK, title-parse-failure fallback (item still kept, requested form used), `requested_form` recorded when title's form differs from query, ET-zoned timestamps normalized to UTC, entries without accession id or alternate link skipped, empty/malformed XML → empty list, dedup_key being the accession number rather than the URL. Plus MockTransport tests verifying per-form query construction (`getcurrent`, `output=atom`, `count=100`), the merge-and-dedup-by-accession behavior across `D` + `D/A` queries (a filing appearing in both is deduped), **partial-failure policy** (one form 5xx, other succeeds → return survivors), **total-failure propagation** (both forms 5xx → raises so runner records `failed`), contact-bearing UA (SEC compliance), non-sec task type guard. Registration sanity check.
- [worker/tests/test_orchestrator.py](../worker/tests/test_orchestrator.py) — 5 tests: existing `run_daily` tests use a `_isolate_config` helper to monkeypatch `DEFAULT_CONFIG_PATH`; the no-fetcher test additionally monkeypatches `runner.FETCHER_REGISTRY` to `{}` so it stays valid as fetcher modules register themselves at import time. New tests cover resolver-wired skipped tasks and resolver `PlanWarning` surfacing in `coverage.plan_warnings`.
- **Result: 223/223 pass via `uv run pytest` in 1.93s. `uv run ruff check .` and `uv run ruff format --check .` clean.**

---

## In progress (🔨)

_Nothing in flight._

---

## Next up (📋)

Ordered by dependency. Each is intended to ship as its own small inspectable PR.

1. **Four remaining fetchers**, plugged into `FETCHER_REGISTRY` in this order (RSS in 6.1, arXiv in 6.2, HN in 6.3, SEC in 6.4):
   1. GDELT DOC 2.0 (per-category queries from yaml)
   2. Reddit (curated subreddits; conservative rate; clear UA)
   3. GitHub (Trending HTML for discovery + REST for repo metadata + **daily star snapshots stored in `source_fetch_state.metadata`** for real velocity)
   4. Websites (configured URLs only; trafilatura extraction; no site-wide crawl)
   The harness is already in place — each fetcher only needs to implement its `async (ResolvedTask) -> list[FetchedItem]` callable and register itself. Fetchers consume `canonicalize_url`, `content_hash`, and the per-source `*_dedup_key` helpers; storage and coverage flow through the runner.
2. **Dedup + clustering.** Level 1 canonical URL → Level 2 content fingerprint → Level 3 event clustering (entity overlap + numeric facts + date window; LLM tie-breaker only on ambiguous pairs).
3. **LLM client (single chokepoint).** OpenAI-compatible `httpx` client wrapping vMLX (`http://127.0.0.1:8080/v1`) and OpenClaw. Per-stage routing from `config/model-routing.yaml`. JSON schema validation via `pydantic`. One bounded JSON-repair retry. Every call logged to `llm_calls`. **No direct LLM calls anywhere else.**
4. **Relevance filter.** Batched cluster judgement against category rules. Keeps every cluster that clears the bar — not top-N. Records `relevance_score` + `filter_reason` on `item_clusters`.
5. **Cluster summary.** Per kept cluster → headline / facts / why-it-matters / caveats / citations. Stored in `item_summaries`.
6. **Final composition.** OpenClaw/`gpt-5.3-codex` from condensed summaries + coverage. **Local `Qwen3.5-122B-A10B-4bit` fallback** if frontier path fails; metadata stamped `composition_provider: vmlx_fallback`.
7. **Publish.** Direct `INSERT` into `digests` (type `daily`) with full coverage `metadata` (run_id, window, model choices, source counts, failed sources). `intel_runs.digest_id` linked, status `published`.
8. **Acceptance gate.** Manual `clawfeed-intel run daily --window 24h` produces a digest visible at `http://127.0.0.1:8767/`. Phase 1 done.

---

## Open risks & decisions

| Risk / decision | Status | Notes |
|---|---|---|
| `digests.type` CHECK only allows `4h`/`daily`/`weekly`/`monthly` | accepted for Phase 1 | Topic briefs (later phase) piggyback on `daily` + `metadata.brief_kind='topic'`. Schema relaxation deferred. |
| `sources` table has no category column | resolved | New `source_categories(source_id, category)` join table — supports many-to-many. |
| GitHub velocity needs ≥2 days of snapshots | accepted | Mitigation: start storing star snapshots from the first run; first usable velocity readings appear by day 2–3. |
| Two-runtime SQLite access (Node `better-sqlite3` + Python `sqlite3`) | accepted | WAL is on; worker writes use `BEGIN IMMEDIATE` and short transactions (<100ms). |
| Migration loader is open-coded in [src/db.mjs](../src/db.mjs#L30-L107) | accepted | Each migration adds another try/exec block. Refactor to a loop is out of scope for Phase 1. |
| vMLX availability is assumed but unverified | open | First step of LLM client work is a real `clawfeed-intel doctor` health-check that pings `http://127.0.0.1:8080/v1/models`. |
| Reddit/SEC/GDELT rate-limit & UA contracts | accepted | Each fetcher declares contact email in UA. SEC: ≤10 req/s. Reddit: conservative. GDELT: polite. |
| Twitter/X | out of scope | Explicitly excluded from v1 per architecture doc. |
| Worker tests separate from `test/e2e.sh` | accepted | Python worker uses `pytest` under `worker/tests/`; bash e2e stays as-is. |
| Local Node binary has a broken simdjson dyld link | environmental | System-level brew issue, unrelated to this project. Use `sqlite3` for DB checks; the running ClawFeed server presumably uses a different node binary. |

---

## Scope changes & decisions

_None yet beyond what's captured in the design docs._

---

## Build log (append-only audit trail)

### 2026-05-04 — Implementation kickoff
- Read full architecture + walkthrough + ClawFeed source.
- Produced kickoff plan; user approved.
- Created [migrations/010_intel_pipeline.sql](../migrations/010_intel_pipeline.sql) (10 tables).
- Wired migration 010 into [src/db.mjs](../src/db.mjs) loader chain.
- Verified migration applies idempotently to live `data/digest.db`.
- Scaffolded `worker/` package: pyproject, `__init__`, `paths`, `cli` stub.
- `python3 -m clawfeed_intel.cli doctor` returns clean status output.
- Created this status file as the engineer-facing source of truth.

### 2026-05-04 — Run lifecycle skeleton (Phase 1 step 3)
- `worker/clawfeed_intel/db.py`: connection, `BEGIN IMMEDIATE` transaction wrapper, run-state CRUD, digest insert. Status enums mirror SQL CHECKs and validate at the boundary; transitions raise `RunStateError` on illegal moves.
- `worker/clawfeed_intel/timewindow.py`: `--window 24h`/`7d` parser; UTC ISO 8601 helpers; rejects naive datetimes.
- `worker/clawfeed_intel/runs.py`: `Coverage` + `RunMetadata` dataclasses matching the architecture doc's JSON shape.
- `worker/clawfeed_intel/pipeline/orchestrator.py`: `run_daily()` walks the full state machine, publishes a stub digest, links `intel_runs.digest_id`, and routes any stage exception to `finish_run(status='failed')`.
- `worker/clawfeed_intel/cli.py`: real `run daily` driver; `doctor` now counts the 10 intel tables.
- `worker/tests/`: 34 tests across `test_timewindow.py`, `test_db.py`, `test_orchestrator.py` — all pass under `uv run pytest`. `uv sync --extra dev` provisions the venv.
- `uv run ruff check .` and `uv run ruff format --check .` are clean.
- Manual smoke: `DIGEST_DB=/tmp/clawfeed_smoke.db python3 -m clawfeed_intel.cli run daily` produces an `intel_runs` row in `published` state and a corresponding `digests` row whose metadata blob contains `brief_kind`, `run_id`, window timestamps, and the full `coverage` object.

### 2026-05-04 — Normalization + raw_items upsert (Phase 1 step 4)
- `worker/clawfeed_intel/normalize.py`: `canonicalize_url` (case fold scheme/host, drop `www.`/`amp.`/userinfo, strip default ports, drop fragment, strip `/amp` path suffix and trailing slash on non-root, drop tracking params — UTM/click-id/Mailchimp/HubSpot prefixed/share/ref/etc. — and sort remaining query params; idempotent). `content_hash` (SHA-256 over normalized title + first 4000 chars of body, whitespace-collapsed and case-folded). Per-source dedup-key helpers `hn_dedup_key`, `reddit_dedup_key`, `github_dedup_key`, `arxiv_dedup_key`, `sec_dedup_key`.
- `worker/clawfeed_intel/db.py`: added `upsert_raw_item(...)` (single transaction: `INSERT ... ON CONFLICT(source_type, dedup_key) DO NOTHING RETURNING id` plus `INSERT OR IGNORE INTO run_raw_items`; preserves first-sight `raw_items.run_id` across runs while linking the current run via `run_raw_items`; returns `(raw_item_id, was_new)`) and `link_raw_item_to_run(...)` (idempotent; FK violations propagate as `IntegrityError`).
- 63 new tests across `test_normalize.py` (42) and `test_raw_items.py` (15), plus 6 parametrized cases. Total worker suite is now 97 tests passing in 0.49s. ruff check + format clean.
- Cascade behavior verified: deleting an `intel_runs` row sets `raw_items.run_id` to NULL (by `ON DELETE SET NULL`) and removes matching `run_raw_items` rows (by `ON DELETE CASCADE`). raw items survive run deletion.

### 2026-05-04 — Source plan resolver (Phase 1 step 5)
- `config/intel-sources.yaml`: starter editorial config (`profile`, four categories, `dynamic_search.enabled_sources`). Uses `kind:` as the structured-source discriminator — `type:` would clash with ClawFeed's `sources.type` column, which the resolver also reads.
- `worker/clawfeed_intel/sources.py`: `build_source_plan(conn, *, config_path=None) -> SourcePlan`. Pydantic discriminated union of nine `*Task` models (rss / arxiv / hn / reddit / gdelt / sec_edgar / github_search / github_trending / website) with `extra="forbid"` for tight YAML validation. DB join filters `is_active = 1`, joins `source_categories`, maps `sources.type` to fetcher kind via `_DB_TYPE_TO_KIND` (Twitter excluded by design). Translates ClawFeed history: HN `filter` → task `list`; github_trending `language: 'all'` → `None`. Lazily materializes a category if a DB tag references one the YAML doesn't list (user mid-edit). Returns structured `PlanWarning`s; `SourcePlan.tasks_by_kind()` groups across categories for fetcher dispatch.
- Soft-failure surface (warnings, never raises): missing config file, single bad source entry, non-mapping category body, unknown YAML `kind`, unsupported `sources.type`, corrupt `sources.config` JSON, missing required DB config fields.
- Hard-failure surface (raises): top-level YAML is not a mapping. This is a deploy bug — degrading silently would hide it.
- `worker/tests/test_sources.py`: 21 new tests covering full-yaml round-trip across all nine kinds, every soft-failure path, top-level corruption, multi-category DB tagging, YAML+DB merge under one category, default-config-path regression. Total worker suite is now 118 tests passing in 0.85s. ruff check + format clean.
- Resolver is **not yet wired into the orchestrator** — that's part of step 6 (fetchers), where the per-kind dispatch loop is the natural site to call `build_source_plan(conn)` and feed `tasks_by_kind()` into each fetcher.

### 2026-05-04 — Fetcher harness + orchestrator wiring (Phase 1 step 6.0)
- Step 6 split: 6.0 ships the harness + orchestrator wiring; 6.1–6.8 each ship one fetcher. This keeps the inspectable unit small and pins the contract every fetcher must satisfy.
- `worker/clawfeed_intel/fetchers/__init__.py` + `base.py` + `runner.py`:
  - `FetchedItem` dataclass: shape matches `db.upsert_raw_item` parameters via `upsert_kwargs()`. Producers must already have run `canonicalize_url` and `content_hash`; the runner does not redo that work.
  - `FetchOutcome` with `status` ∈ `{succeeded, failed, skipped}` plus items_seen/items_new/latency_ms/error.
  - `FetcherCallable = Callable[[ResolvedTask], Awaitable[list[FetchedItem]]]` — flat async-function contract, no class hierarchy. Module-level `FETCHER_REGISTRY` for production wiring.
  - `run_fetch_stage`: tasks of the same kind run concurrently via `asyncio.gather`. Per-task try/except → `failed` outcome with `_short_error(exc)`; sibling tasks of same kind continue. Per-item upsert failures logged but don't blackhole the batch. SQLite writes serialize naturally on the single-threaded event loop.
- `worker/clawfeed_intel/runs.py`: `Coverage` extended with `skipped_sources` and `plan_warnings` lists plus `record_skipped` / `record_plan_warning` helpers. Skipped counts as attempted but is held distinct from failed_sources.
- `worker/clawfeed_intel/db.py`: added `record_fetch_success` (resets consecutive_errors, clears last_error, optional cursor/metadata via COALESCE) and `record_fetch_failure` (bumps consecutive_errors, never overwrites last_success_at or cursor — flaky stretches retain the last-known-good resumption point).
- `worker/clawfeed_intel/pipeline/orchestrator.py`: `_drive_run` calls `build_source_plan(conn)` and `asyncio.run(run_fetch_stage(...))` during the `fetching` state. With no fetchers registered yet, every resolved task records `skipped: no fetcher for kind <X>`. Run still publishes the stub digest with full coverage block.
- `worker/tests/test_fetch_state.py` (8 tests) + `test_fetchers_runner.py` (13 tests, all using stub fetchers — no HTTP) + 2 new orchestrator tests covering the resolver-wiring + plan-warning paths. Existing orchestrator tests now monkeypatch `DEFAULT_CONFIG_PATH` so they're insulated from editorial-config edits. Total worker suite is now 141 tests passing in 1.13s. ruff check + format clean.
- Manual smoke confirmed: `clawfeed-intel run daily --window 24h` against a temp DB walks the full lifecycle, the resolver loads the editorial config, the harness records 7 skipped tasks (one per default-config source entry), and the digest publishes with `skipped_sources` populated. No fetcher modules registered — that's the next eight steps.

### 2026-05-04 — RSS / Atom fetcher (Phase 1 step 6.1)
- `worker/clawfeed_intel/fetchers/http.py`: shared `httpx.AsyncClient` factory (`build_client()` async-context-manager). Polite `User-Agent` `ClawFeed-Intel/<version> (+contact: <email>)` with email from `CLAWFEED_CONTACT_EMAIL` env (defaulting to `noreply@clawfeed.local`). Default timeouts (connect 5s, read 20s) and connection limits (20 max, 8 keepalive). Future fetchers reuse this verbatim.
- `worker/clawfeed_intel/fetchers/rss.py`: registers under `kind="rss"`. Two layers — pure `parse_feed_text` (feedparser + selectolax HTML strip + canonicalize_url + content_hash) plus async `fetch_rss` for the HTTP wrap. Atom `<content>` preferred over `<summary>`; `<published>` preferred over `<updated>`; tags surfaced in metadata; original HTML kept in `raw_payload` for forensics. Trafilatura full-text fallback fires when summary text is below 400 chars; `_trafilatura_extract` is best-effort (returns None on any failure → keep original summary). Per-entry conversion errors are logged and skipped without aborting the feed.
- `worker/clawfeed_intel/fetchers/__init__.py`: imports `rss` to trigger registration. Subsequent fetcher steps add an analogous import.
- `worker/tests/test_fetchers_rss.py`: 17 tests (7 pure-parse + 4 HTTP via `httpx.MockTransport` + 4 trafilatura-fallback paths + 2 registration sanity). No live network.
- `worker/tests/test_orchestrator.py`: extracted `_isolate_config` and new `_empty_fetcher_registry` helpers. The "skipped tasks" test now monkeypatches `runner.FETCHER_REGISTRY` to `{}` so it stays valid as more fetcher modules register at import. Note: had to patch the runner-module binding (not `base.FETCHER_REGISTRY`) because `from .base import` rebinds into the runner namespace at import time.
- 158 tests pass in 1.16s. ruff check + format clean.
- **Failure-mode verified by tests:** 4xx/5xx on the feed URL raises `httpx.HTTPStatusError` → runner converts to `failed` outcome → coverage records the source in `failed_sources`. Trafilatura article-fetch failure during enrichment is swallowed by `_trafilatura_extract` returning `None`. Per-entry parse failures are logged and skipped. Zero of these abort the run.

### 2026-05-04 — arXiv fetcher (Phase 1 step 6.2)
- `worker/clawfeed_intel/fetchers/arxiv.py`: registers `kind="arxiv"`. Hits `https://export.arxiv.org/api/query` with `OR`-joined `cat:<X>` filters from the task's categories, sorted `submittedDate desc`, `max_results=500`. Stdlib `xml.etree.ElementTree` for parsing — no defusedxml needed (arXiv is a trusted endpoint), no feedparser (its arxiv-namespace handling is version-flaky). Two layers preserved: pure `parse_atom_response` for fixture-based tests, async `fetch_arxiv` for HTTP.
- Per-entry conventions: `dedup_key = arxiv_dedup_key(<id>v<n>)` keeps revisions distinct (downstream clustering will merge them logically); abstract → `content` (no trafilatura — arXiv abstracts are already condensed); whitespace collapsed in title/summary (arXiv wraps both for line-length); authors joined `, ` for `author`, full list in `raw_payload`; metadata carries `arxiv_id`, `primary_category`, all `categories` (primary-first, deduped), `abs_url`, optional `pdf_url` / `doi`, and the `query_url` for traceability.
- Both modern (`2405.12345v1`) and legacy (`math/0506203v2`) ID formats handled. Naive timestamps treated as UTC (arXiv spec is UTC-only) rather than dropped.
- 20 new tests in `test_fetchers_arxiv.py` (13 pure-parse + 5 HTTP via MockTransport + 2 registration). 178 tests pass in 1.45s. ruff check + format clean.
- **Failure modes verified by tests:** 4xx/5xx → `HTTPStatusError` → runner records `failed`. Malformed XML → empty list, no raise. Per-entry conversion exceptions → logged and skipped. Non-arxiv task type → `TypeError` (a programmer error, not a fetcher-time failure). Zero of these abort the run.

### 2026-05-04 — Hacker News fetcher (Phase 1 step 6.3)
- `worker/clawfeed_intel/fetchers/hn.py`: registers `kind="hn"`. Per task: `topstories.json` / `beststories.json` / `newstories.json` / `showstories.json` / `askstories.json` from the Firebase API → truncate to `task.limit` (default 200) → fan out item fetches under `asyncio.Semaphore(10)` → optional `min_score` filter applied post-fetch (the list endpoint only returns IDs).
- Algolia HN Search is **deliberately not wired here** — architecture doc reserves it for topical search (Phase 7).
- Per-item failures (5xx, network error, JSON `null` for deleted items) are caught at the `gather` boundary and logged; one bad item must not abort the rest of the task. Verified by an explicit test.
- Per-item parse rules: `dedup_key = hn_dedup_key(<id>)` keeps two HN posts of the same external URL as distinct attention signals (cross-source dedup happens later via `content_hash`/clustering); `canonical_url` falls back to the discussion URL for Ask HN; HTML body (`text` field) decoded via selectolax; `published_at` from unix `time` epoch; `raw_payload` strips `kids`/`parts` to avoid bloating the row with comment trees.
- Window scoping deliberately **not applied** — `topstories` is an attention snapshot, not a 24h window. A week-old story currently #1 represents today's developer attention. Downstream relevance can use `metadata.list` and `score` instead of `published_at`.
- `worker/tests/test_fetchers_hn.py`: 27 new tests covering 11 pure-parse edge cases, 5 list-endpoint variants (parametrized), limit/min_score, null/dead/deleted handling, **load-bearing per-item-failure-doesn't-abort assertion**, list-endpoint 5xx propagation, and graceful degradation on unexpected list-endpoint shape.
- 205 tests pass in 1.43s. ruff check + format clean.
- **Failure modes verified by tests:** list endpoint 5xx → `HTTPStatusError` → runner records `failed`. Per-item 5xx → swallowed, logged, batch continues. JSON `null` item → skipped. Non-list-endpoint payload (defensive against API shape change) → empty list rather than raise. Comment / deleted / dead / untitled items → silently skipped. Non-hn task → `TypeError` (programmer error).

### 2026-05-04 — SEC EDGAR fetcher (Phase 1 step 6.4)
- `worker/clawfeed_intel/fetchers/sec.py`: registers `kind="sec_edgar"`. Hits the legacy `getcurrent` Atom endpoint (`https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=<F>&output=atom&count=100`) once per form in `task.forms`, fanned out concurrently. Stdlib `xml.etree.ElementTree` for parsing — same pattern as arXiv: trusted source + minimal-dep philosophy.
- `dedup_key = sec_dedup_key(accession_number)`. Accession is SEC's canonical filing identifier — globally unique across filers — so URL-based dedup would miss mirrored / re-rendered filing pages.
- Title shape `<FORM> - <COMPANY> (<CIK>) (Filer)` parsed via regex; on failure (EDGAR ever changes shape), the entry is kept with empty `company_name`/`cik` and the requested form used as `form_type`. We never drop an entry over title-parse failure.
- Metadata: `accession_number`, `filing_url`, `form_type` (from title), `requested_form` (only when title's form differs from request), `company_name`, `cik`, `query_url`. `published_at` normalized from `-04:00`/`-05:00` to UTC.
- **Partial-failure semantics**: when `forms=["D", "D/A"]` and one query fails, the survivor's items are returned; when *all* fail, the first exception propagates so the runner records `failed`. This balances "don't lose the data we did retrieve" with "don't silently mask total failure."
- `task.ciks` field reserved for Phase 7 (topical search filtering by filer); daily-brief use case is form-only, so the field is intentionally ignored with an inline comment.
- SEC compliance: their explicit ≤10 req/s ceiling is met trivially (1-2 requests per fetch); the contact-bearing User-Agent comes from `fetchers.http` and is verified by a dedicated test.
- `worker/tests/test_fetchers_sec.py`: 18 new tests (10 pure-parse + 6 MockTransport + 2 registration). 223 tests pass in 1.93s. ruff check + format clean.
- **Failure modes verified by tests:** all-form 5xx → `HTTPStatusError` propagates → runner records `failed`. Partial-form 5xx → survivors returned. Malformed XML → empty list. Per-entry parse exception → logged + skipped. Title-parse failure → entry kept with empty company/cik fallback. Non-sec task → `TypeError`.
