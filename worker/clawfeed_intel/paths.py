"""Filesystem layout for the worker.

The worker shares the SQLite file with the ClawFeed Node server. We compute
paths relative to the repo root rather than hardcoding /Users/... so the same
checkout works on any machine.
"""

from __future__ import annotations

import os
from pathlib import Path

# worker/clawfeed_intel/paths.py -> worker/clawfeed_intel -> worker -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
DB_PATH = Path(os.environ.get("DIGEST_DB", DATA_DIR / "digest.db"))

CONFIG_DIR = REPO_ROOT / "config"
PROMPTS_DIR = REPO_ROOT / "prompts"
MIGRATIONS_DIR = REPO_ROOT / "migrations"
