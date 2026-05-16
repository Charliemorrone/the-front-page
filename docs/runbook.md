# Personal Intelligence Brief — Operations Runbook

Operator-facing reference for the daily-brief pipeline. Use it to
diagnose a failed or degraded run, recover from common failure modes,
and verify the system is healthy before installing or changing the
scheduled job.

For the build state + decision history, see
[`current-project-status.md`](current-project-status.md). For the full
architecture, see
[`personal-intelligence-brief-architecture.md`](personal-intelligence-brief-architecture.md).

---

## What runs when

- **06:15 local each morning**: the macOS `launchd` LaunchAgent
  `local.clawfeed.daily-brief` fires. It runs:

  ```bash
  cd /Users/merlin/clawfeed && uv run clawfeed-intel run daily --window 24h
  ```

  stdout and stderr land in `data/logs/daily-brief.{out,err}.log`.

- **Pipeline**: fetch from all eight source families → normalize +
  upsert raw items → cluster (L1 canonical URL + L2 content
  fingerprint + L3 heuristic event similarity) → vMLX relevance
  filter → vMLX cluster summarize → Gemini CLI final compose
  (Tier 1) with vMLX fallback (Tier 2) and deterministic render
  (Tier 3) → publish digest.

- **Audit trail** lives in SQLite: `intel_runs` for run lifecycle,
  `raw_items` for fetched items, `item_clusters` + `item_summaries`
  for the processed evidence, `llm_calls` for every LLM dispatch,
  `digests` for the published brief.

---

## Quick diagnostic commands

```bash
# Is the system runnable right now?
uv run clawfeed-intel doctor

# Would the next scheduled run succeed?
uv run clawfeed-intel run daily --dry-run

# Is the LaunchAgent registered with launchd?
uv run clawfeed-intel cron status

# What would cleanup remove?
uv run clawfeed-intel cleanup
```

All four are read-only or print-only. Run any of them at any time.

---

## Reading a published brief

Every digest carries a `metadata` JSON blob. The fields that matter
for diagnosis:

| Field | What it means |
|---|---|
| `composition_provider` | Which Tier produced the brief (see table below). |
| `composition_model` | Specific model that ran the compose call. |
| `coverage.sources_attempted` | Source plan size at run time. |
| `coverage.sources_succeeded` | How many fetchers completed without raising. |
| `coverage.failed_sources` | Names of sources that errored. Inspect these first when fetch coverage looks thin. |
| `coverage.skipped_sources` | Sources whose fetcher isn't registered for the configured kind. |
| `coverage.plan_warnings` | YAML / DB warnings the resolver raised without aborting. |
| `coverage.raw_items` | How many normalized items entered the pipeline. |
| `coverage.clusters` | Distinct clusters after L1+L2+L3 dedup. |
| `coverage.kept_clusters` | Clusters the relevance filter accepted. |
| `coverage.failed_filter_batches` | Filter calls that errored after the bounded JSON repair. |
| `coverage.summarized_clusters` / `failed_summary_clusters` | Per-cluster summary outcomes. |
| `local_models.{filter,summary}` | Which vMLX models actually ran the stages. |

### `composition_provider` tag table

| Tag | Meaning | Action |
|---|---|---|
| `gemini_cli` | Tier 1 frontier compose succeeded. | None — happy path. |
| `vmlx_fallback` | Either Tier 1 routed to vMLX directly (no `gemini_cli` provider in YAML), or Tier 2 fell back to vMLX after a Gemini failure. | Check the `llm_calls` table for the failed Tier-1 call to see what broke. If persistent across multiple days, the Gemini CLI integration is unhealthy — see "Gemini CLI stalls" below. |
| `local_stub_failed` | Both Tier 1 and Tier 2 (when configured) failed. Brief was rendered deterministically from the structured cluster summaries. | The vMLX preflight should have caught the underlying issue; check `llm_calls` and the worker logs for the specific failure. |
| `local_stub_empty` | Zero summarized clusters reached compose — fetch fully degraded, filter rejected everything, or every cluster's summary call failed. | Check `coverage.failed_sources`, `failed_filter_batches`, `failed_summary_clusters` to localize the upstream failure. |

---

## Failure modes and recovery

### vMLX completely unreachable

**Symptom.** `clawfeed-intel doctor` reports `[FAIL] health: ConnectError`. The daily run aborts at preflight with exit 3 — no `intel_runs` row, no published digest. launchd's error log shows `preflight: vMLX is not ready — aborting before run`.

**Diagnosis.** Is the vMLX process running?

```bash
curl -sS http://127.0.0.1:8080/health
ps aux | grep -i mlx
```

**Recovery.** Restart the vMLX server. The launch command is:

```bash
python3 -m mlx_lm server --model mlx-community/Qwen3-8B-4bit \
  --host 127.0.0.1 --port 8080 \
  --chat-template-args '{"enable_thinking":false}'
```

(See `~/.claude/projects/-Users-merlin-clawfeed/memory/vmlx_environment.md` for the verified process line.) Once `curl /health` returns `{"status": "ok"}`, re-run `clawfeed-intel doctor` to confirm. Then re-fire the brief manually:

```bash
uv run clawfeed-intel run daily --window 24h
```

### vMLX up but the model isn't loaded

**Symptom.** `doctor` shows `health` + `models` pass but `chat:relevance_filter` fails with a timeout or model-not-found error.

**Diagnosis.** Check the loaded model registry:

```bash
curl -sS http://127.0.0.1:8080/v1/models | jq '.data[].id'
```

The model named in `config/model-routing.yaml` for the failing stage must appear. vMLX hot-loads cached models on demand, so a missing model usually means the HuggingFace cache doesn't have it.

**Recovery.** Either download the missing model (`huggingface-cli download <model>`) or edit `config/model-routing.yaml` to point the stage at a model that's already cached. Both `Qwen3.5-27B-4bit` and `Qwen3-30B-A3B-4bit` are typically present on this machine — verified in the memory file.

### Gemini CLI auth expired

**Context.** The `gemini_cli` provider is signed into the operator's **Gemini Pro subscription** via the CLI's OAuth flow. There is no API key. The CLI manages its own refresh-token lifecycle internally; "auth expired" means the refresh token has lapsed and an interactive re-login is required.

**Symptom.** Published brief has `composition_provider: "vmlx_fallback"` despite the routing config showing `final_compose: gemini_cli`. The `llm_calls` row for the failed Tier-1 call has `error` containing `GeminiCliExitError` with stderr mentioning auth / login / token / OAuth.

**Diagnosis.** Manually invoke the Gemini CLI:

```bash
/opt/homebrew/bin/node /opt/homebrew/bin/gemini --version
echo "Reply with PONG." | /opt/homebrew/bin/node /opt/homebrew/bin/gemini -p "" --approval-mode plan -m gemini-3-pro-preview
```

If the second command produces an auth error, the OAuth refresh has lapsed.

**Recovery.** Re-authenticate the Gemini CLI interactively against the operator's Gemini Pro account:

```bash
/opt/homebrew/bin/node /opt/homebrew/bin/gemini
```

Follow the OAuth flow in the terminal/browser. The CLI persists the refreshed token; no worker code change is needed. Then re-run the dry-run preflight to confirm:

```bash
uv run clawfeed-intel run daily --dry-run
```

**Do NOT switch to a pay-per-token Gemini API key as a fix.** The architecture-doc decision (Decision 4 amendment, 2026-05-15) is explicit: this integration uses the Pro subscription via the CLI, not the API. Quota-exceeded errors are handled by the Tier-2 vMLX fallback; a persistent quota issue is a "the Pro plan ceiling was wrong" signal, not a "we should adopt API billing" signal.

### Gemini CLI stalls mid-response (Tier-1 → Tier-2 fallback)

**Symptom.** Published brief has `composition_provider: "vmlx_fallback"`. The `llm_calls` row for the Tier-1 call has `error: "GeminiCliStallError: gemini stream went idle for >60s"` and `status: "failed"`. A subsequent `llm_calls` row records the Tier-2 vMLX call succeeding.

**Diagnosis.** A single stall on a given day is unremarkable — Gemini Pro 3 occasionally has tail-latency issues that the provider's `stream-json` idle detection trips on. The brief still published with citations preserved (via local vMLX prose); the only loss is some prose polish.

**Recovery (if persistent).** If `vmlx_fallback` appears on multiple consecutive days, increase the per-stage tolerance in `config/model-routing.yaml`:

```yaml
providers:
  gemini_cli:
    idle_timeout_seconds: 120    # was 60
    hard_timeout_seconds: 600    # was 300
```

Then re-run `clawfeed-intel run daily --dry-run` to confirm the routing reloads cleanly. If stalls persist beyond config tuning, file a bug against the Gemini CLI itself — the worker's mitigation budget is exhausted.

### Both compose tiers fail (Tier-3 deterministic brief)

**Symptom.** Published brief has `composition_provider: "local_stub_failed"`. The H1 line ends with `(degraded)` and the body has a `_Final composition failed; this brief was rendered directly from the structured cluster summaries…_` notice.

**Diagnosis.** Two LLM calls failed in succession. Check the `llm_calls` table:

```bash
sqlite3 data/digest.db "SELECT created_at, stage, provider, model, status, error FROM llm_calls WHERE run_id = (SELECT MAX(id) FROM intel_runs WHERE status='published') AND stage='final_compose' ORDER BY id;"
```

The two rows show what each tier produced. Common causes:
- vMLX wedged mid-run (memory pressure, model crash) — restart vMLX.
- Both providers hit the same upstream issue (DNS, network) — verify connectivity.
- The cluster set was malformed in a way that broke prompt construction — check the worker logs around the run.

**Recovery.** The brief published in Tier-3 form, so today's read is still valuable. Fix the underlying provider issue, then manually re-fire the run to overwrite with a Tier-1 / Tier-2 brief:

```bash
uv run clawfeed-intel run daily --window 24h
```

### Source rate-limited or timing out

**Symptom.** `coverage.failed_sources` lists one or more source names. The fetcher's HTTP error is recorded in the worker stderr log.

**Diagnosis.** Per source-family expectations:
- **Reddit**: ~60 req/min unauthenticated; private subs return 403.
- **GitHub**: 60 req/hr unauthenticated, 5000/hr authenticated. Check `GITHUB_TOKEN` env var.
- **SEC EDGAR**: ≤10 req/s, strict User-Agent requirement.
- **GDELT**: degrades to empty rather than failing in many cases.
- **arXiv**: 429s under heavy load.

**Recovery.** A single source failure does not fail the run — the brief publishes with reduced coverage. If a source family fails consistently:

1. Set the relevant API token (`GITHUB_TOKEN`) or contact email (`CLAWFEED_CONTACT_EMAIL` env var).
2. Reduce the source's request volume by editing its YAML config.
3. Mark the source `is_active=0` in the `sources` table to skip it entirely.

### LLM emits malformed JSON (filter / summary stages)

**Symptom.** `coverage.failed_filter_batches > 0` or `coverage.failed_summary_clusters > 0`. The `llm_calls` table shows `error: LLMSchemaError` for the affected calls.

**Diagnosis.** The LLM client already attempts one bounded JSON repair before recording the failure. A persistent failure means the model is consistently producing invalid output for the prompt shape. Inspect the prompt + the captured response hashes:

```bash
sqlite3 data/digest.db "SELECT stage, provider, model, input_hash, output_hash, error FROM llm_calls WHERE status='failed' ORDER BY id DESC LIMIT 10;"
```

**Recovery.** Typically a model swap fixes it — the routing schema's `relevance_filter` / `cluster_summary` stages can be pointed at a different cached vMLX model without code change. If the schema itself needs to be more permissive, the 9c / 10a lesson applies: prefer making fields optional over forcing repair retries. See `worker/clawfeed_intel/llm/schemas.py`.

### Source plan resolver raises

**Symptom.** Daily run aborts (or dry-run exits 3) with `source_plan: resolver raised`. The error names a `ValueError` or `ValidationError`.

**Diagnosis.** Either `config/intel-sources.yaml` has a structural error, or the `sources` table contains a row whose `config` JSON doesn't match what the resolver expects. The PlanWarning soft-failure path catches single-entry issues; a top-level YAML structural problem is a hard raise.

**Recovery.** Validate the YAML:

```bash
python -c "import yaml; yaml.safe_load(open('config/intel-sources.yaml'))"
```

If YAML is fine, the issue is in the DB. Check recent edits to dashboard-managed sources. Per-source soft-failures land in `coverage.plan_warnings` — the dry-run prints them so you can see what's flagged.

### DB locked / unavailable

**Symptom.** Worker fails with `sqlite3.OperationalError: database is locked` or `unable to open database file`.

**Diagnosis.** WAL is on; long-running readers (e.g. the ClawFeed Node server holding a transaction) shouldn't lock writers, but a writer holding a transaction past `busy_timeout` will. Check:

```bash
fuser data/digest.db
lsof data/digest.db
```

**Recovery.** Identify and end the blocking process. If `data/digest.db` is missing or corrupt, restore from the most recent backup (manual operation — automated backup is not yet in scope; see "Open risks" in the status doc).

### launchd job not firing

**Symptom.** No new digest at the expected daily time. `clawfeed-intel cron status` reports `installed` but no recent runs.

**Diagnosis.** Check whether launchd actually fired the agent:

```bash
launchctl print gui/$(id -u)/local.clawfeed.daily-brief
```

Look for the `state =` line (should be `running` or `waiting`) and the `next firing =` timestamp.

If the agent was disabled by macOS (because of repeated crashes, for instance), `state = exited` or the agent is missing from the listing.

**Recovery.** Re-bootstrap:

```bash
uv run clawfeed-intel cron uninstall --remove
uv run clawfeed-intel cron install --install
```

Then verify with `cron status` again. If the agent keeps getting disabled, the underlying job is crashing — check `data/logs/daily-brief.err.log` for the failure trace.

### Empty brief (zero clusters reached compose)

**Symptom.** Published brief has `composition_provider: "local_stub_empty"`, an "Executive Read" section that says "No cluster summaries reached the composition stage for this run", and only a Coverage section.

**Diagnosis.** Walk the coverage block backwards:
- `kept_clusters: 0` → relevance filter rejected everything. Inspect the filter's verdicts in `item_clusters`.
- `clusters: 0` → no items entered clustering. Inspect `raw_items` for the run.
- `raw_items: 0` → every fetcher failed or skipped. Inspect `failed_sources` and `skipped_sources`.

**Recovery.** Depends on which step is empty. If sources failed, see "Source rate-limited" above. If the relevance filter wholesale rejected, the prompt or model may be too restrictive — try the cluster summaries on a different cached model to see whether the content is actually thin or just the filter is.

---

## Recovery procedures

### Restart vMLX

```bash
# Stop (if running)
pkill -f "mlx_lm server"

# Start (verified launch line from memory file)
python3 -m mlx_lm server --model mlx-community/Qwen3-8B-4bit \
  --host 127.0.0.1 --port 8080 \
  --chat-template-args '{"enable_thinking":false}' &

# Verify
curl -sS http://127.0.0.1:8080/health
```

### Refresh Gemini CLI OAuth

```bash
# Interactive — follow the OAuth flow in the terminal/browser
/opt/homebrew/bin/node /opt/homebrew/bin/gemini

# Confirm a non-interactive call works
echo "Reply with PONG." | /opt/homebrew/bin/node /opt/homebrew/bin/gemini -p "" -m gemini-3-pro-preview --approval-mode plan
```

### Re-install the LaunchAgent

```bash
uv run clawfeed-intel cron uninstall --remove
uv run clawfeed-intel cron install --install
uv run clawfeed-intel cron status
```

### Manually re-fire a daily run

```bash
# Skip cron, run immediately
uv run clawfeed-intel run daily --window 24h
```

The published digest overwrites whatever existed for the same window —
the dashboard will reflect the new brief.

### Prune old data

```bash
# Preview what would be deleted
uv run clawfeed-intel cleanup

# Actually delete (per architecture-doc defaults: raw_items 90 days, llm_calls 30)
uv run clawfeed-intel cleanup --apply

# Custom windows
uv run clawfeed-intel cleanup --raw-items-keep-days 30 --llm-calls-keep-days 14 --apply
```

`item_clusters` and `item_summaries` are preserved indefinitely per the
architecture-doc retention policy — orphaned clusters (whose member
raw_items aged out) still carry their headline + summary text so
historical briefs remain coherent.

---

## Useful one-off queries

Most-recent successful runs:

```bash
sqlite3 data/digest.db "
SELECT id, run_type, status, started_at, finished_at,
       json_extract(metadata, '$.composition_provider') AS tier
FROM intel_runs ORDER BY id DESC LIMIT 10;
"
```

Failed LLM calls in the last 24 hours:

```bash
sqlite3 data/digest.db "
SELECT created_at, stage, provider, model, error
FROM llm_calls
WHERE status = 'failed' AND created_at >= datetime('now','-1 day')
ORDER BY created_at DESC;
"
```

Coverage of the most recent published brief:

```bash
sqlite3 data/digest.db "
SELECT json_extract(metadata, '$.coverage')
FROM digests
WHERE type = 'daily'
ORDER BY id DESC LIMIT 1;
" | jq
```

---

## When to file a bug vs adjust config

| Situation | Fix |
|---|---|
| Daily run takes too long (>2 hours) | Switch to a smaller model in `model-routing.yaml`. |
| Gemini stalls > 1×/week | Tune `idle_timeout_seconds` / `hard_timeout_seconds`. |
| Source consistently rate-limited | Set `GITHUB_TOKEN` or `CLAWFEED_CONTACT_EMAIL`; reduce request volume in YAML. |
| Coverage thin for a known-active topic | Add more sources to the relevant category in `intel-sources.yaml`. |
| Brief prose quality drops day-over-day | Check whether `composition_provider` flipped from `gemini_cli` to `vmlx_fallback`. |
| Worker crashes mid-pipeline | File a bug — the failure-mode contract is "degrade coverage, don't crash." |
| Cleanup CLI removes more / less than expected | Check the cutoff timestamp printed by dry-run vs the actual row timestamps; lexicographic ISO comparison is strict. |
