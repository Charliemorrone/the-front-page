-- Migration 011: GitHub repository observations for velocity tracking
--
-- The intelligence pipeline needs evidence-backed "repos gaining traction"
-- signal — not "appeared on Trending today". That requires storing
-- observations across runs and computing star/fork deltas over time.
--
-- This table is the storage substrate. One row per (repo, observation),
-- written each time the GitHub fetcher sees a repo in trending or search.
-- The fetcher computes velocity by reading recent rows for the repo.
--
-- Why a dedicated table rather than overloading source_fetch_state.metadata:
-- the natural identity of an observation is the repo (`owner/repo`), not
-- the source row that surfaced it — the same repo can be discovered by
-- both `github_trending` (no language filter) and `github_search`
-- (`topic:llm`), and by both YAML-origin and DB-origin tasks. Velocity
-- belongs to the repo, not to any one source. source_fetch_state is also
-- ``source_id NOT NULL`` which would exclude YAML-origin tasks.
--
-- Retention: pruned periodically by Phase 6 (architecture doc: "Raw
-- content has TTL cleanup"). Keep ≥30 days for short-window velocity.

CREATE TABLE IF NOT EXISTS github_repo_observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name TEXT NOT NULL,
  observed_at TEXT NOT NULL DEFAULT (datetime('now')),
  stars INTEGER NOT NULL,
  forks INTEGER,
  watchers INTEGER,
  open_issues INTEGER,
  language TEXT,
  topics TEXT NOT NULL DEFAULT '[]',
  pushed_at TEXT,
  discovered_via TEXT NOT NULL CHECK(discovered_via IN ('trending', 'search'))
);

-- Read pattern: "give me the recent observations for repo X". The compound
-- index on (full_name, observed_at DESC) covers this directly.
CREATE INDEX IF NOT EXISTS idx_gh_obs_repo_at
  ON github_repo_observations(full_name, observed_at DESC);

-- Read pattern: "which repos did we observe in the last 7 days". Useful for
-- batch velocity computation and retention pruning.
CREATE INDEX IF NOT EXISTS idx_gh_obs_at
  ON github_repo_observations(observed_at);
