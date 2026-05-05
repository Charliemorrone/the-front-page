"""Shared pytest fixtures.

The :func:`temp_db` fixture creates a fresh SQLite file in a per-test temp
directory and applies the project's migration files to it. Tests never touch
the real ``data/digest.db``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from clawfeed_intel.paths import MIGRATIONS_DIR


def _split_statements(sql: str) -> list[str]:
    """Split a SQL file into executable statements.

    Naive split-on-semicolon is adequate because none of our migrations use
    semicolons inside string literals, triggers, or BEGIN/END blocks. Lines
    that are entirely ``--`` comments or whitespace are skipped so that a
    file's leading comment header does not produce an empty statement.
    """
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


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """A fresh SQLite database with all migrations applied."""
    db_path = tmp_path / "digest.db"
    _apply_migrations(db_path)
    return db_path
