"""Run-level data model: coverage accounting and digest metadata.

These shapes are populated incrementally by pipeline stages and serialized into
``intel_runs.metadata`` and ``digests.metadata``. Match the JSON shape in the
architecture doc so downstream tooling (UI, audits) can rely on it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Coverage:
    """Per-run accounting of what the pipeline saw and skipped.

    Mutated in place by stages; serialized into the digest metadata at publish.
    """

    sources_attempted: int = 0
    sources_succeeded: int = 0
    raw_items: int = 0
    clusters: int = 0
    kept_clusters: int = 0
    failed_sources: list[str] = field(default_factory=list)
    skipped_sources: list[str] = field(default_factory=list)
    plan_warnings: list[str] = field(default_factory=list)

    def record_success(self, source_id: str, items: int) -> None:
        self.sources_attempted += 1
        self.sources_succeeded += 1
        self.raw_items += items

    def record_failure(self, source_id: str) -> None:
        self.sources_attempted += 1
        if source_id not in self.failed_sources:
            self.failed_sources.append(source_id)

    def record_skipped(self, source_id: str, reason: str) -> None:
        """A task we wanted to run but couldn't dispatch (e.g. no fetcher yet).

        Counts as attempted so coverage stays honest about intent, but lives
        on its own list so the brief can distinguish "we tried and the source
        broke" from "we didn't try because the harness was incomplete."
        """
        self.sources_attempted += 1
        entry = f"{source_id}: {reason}" if reason else source_id
        if entry not in self.skipped_sources:
            self.skipped_sources.append(entry)

    def record_plan_warning(self, message: str) -> None:
        if message not in self.plan_warnings:
            self.plan_warnings.append(message)


@dataclass
class RunMetadata:
    """Top-level metadata stamped onto runs and digests.

    Mirrors the JSON example in
    ``docs/personal-intelligence-brief-architecture.md`` ("Daily brief
    digests.metadata example").
    """

    brief_kind: str
    run_id: int
    window_start: str
    window_end: str
    composition_provider: str | None = None
    composition_model: str | None = None
    local_models: dict[str, str] = field(default_factory=dict)
    coverage: Coverage = field(default_factory=Coverage)

    def as_json(self) -> dict[str, Any]:
        return asdict(self)
