-- Migration 010: Personal Intelligence Brief pipeline
--
-- Adds the durable backbone for the daily/topic intelligence pipeline:
--   intel_runs        -- run lifecycle (daily | topic)
--   intel_jobs        -- async job queue
--   raw_items         -- normalized fetched items (one row per distinct dedup_key)
--   run_raw_items     -- many-to-many: which raw items participated in which run
--   item_clusters     -- event clusters within a run
--   cluster_items     -- many-to-many: cluster <-> raw item
--   item_summaries    -- per-cluster grounded summary (input to final composer)
--   llm_calls         -- audit log of every model call
--   source_fetch_state -- per-(source, fetcher) cursor + error tracking
--   source_categories -- many-to-many: ClawFeed source <-> editorial category

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
CREATE INDEX IF NOT EXISTS idx_raw_items_content_hash ON raw_items(content_hash);

CREATE TABLE IF NOT EXISTS run_raw_items (
  run_id INTEGER NOT NULL REFERENCES intel_runs(id) ON DELETE CASCADE,
  raw_item_id INTEGER NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
  PRIMARY KEY (run_id, raw_item_id)
);
CREATE INDEX IF NOT EXISTS idx_run_raw_items_raw ON run_raw_items(raw_item_id);

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

CREATE TABLE IF NOT EXISTS cluster_items (
  cluster_id INTEGER NOT NULL REFERENCES item_clusters(id) ON DELETE CASCADE,
  raw_item_id INTEGER NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
  PRIMARY KEY (cluster_id, raw_item_id)
);
CREATE INDEX IF NOT EXISTS idx_cluster_items_raw ON cluster_items(raw_item_id);

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

CREATE TABLE IF NOT EXISTS source_categories (
  source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  category TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (source_id, category)
);
CREATE INDEX IF NOT EXISTS idx_source_categories_category ON source_categories(category);
