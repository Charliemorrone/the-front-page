"""Source-plan resolver for the daily brief.

Combines two source registries into one per-category fetch plan:

- ``config/intel-sources.yaml`` — editorial categories plus structured
  non-user-facing sources (arXiv categories, GDELT queries, SEC EDGAR forms,
  GitHub topic searches).
- ClawFeed's ``sources`` table — user-managed RSS feeds, websites, subreddits,
  HN definitions, GitHub Trending lists — joined to a category through the
  ``source_categories`` table introduced in migration 010.

The resolver is purely deterministic: no LLM is involved. The LLM-driven
source planner described in the architecture doc applies to *topical search*
(Phase 7), where the input is a free-form query rather than a fixed category
tag. Daily plans are stable, inspectable, and reproducible.

Failure model — single-entry problems degrade coverage instead of failing the
run, matching the hard requirement that *"failed sources degrade coverage;
they do not fail the run"*:

- Missing config file → empty YAML side, single warning, no raise.
- Empty/missing ``categories:`` block → empty YAML side, no warning.
- One malformed source entry inside a category → that entry is dropped,
  warning is recorded with category + offending fragment, sibling entries and
  other categories survive.
- Unknown ``sources.type`` from the DB → warning + skip, doesn't poison the
  category.
- YAML file present but unreadable / not valid YAML → raise. This is a
  deploy-time error, not a degradation, and we want it loud.

The orchestrator (step 6, fetchers) calls :func:`build_source_plan` early in
the fetching stage, dispatches tasks to fetchers grouped by ``kind``, and
appends ``warnings`` to the run's coverage notes.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .paths import CONFIG_DIR

DEFAULT_CONFIG_PATH = CONFIG_DIR / "intel-sources.yaml"


# ── Typed source tasks ────────────────────────────────────────────────────────
#
# One model per fetcher family. ``kind`` is the discriminator. Field shapes are
# what the eight fetchers actually need to do their work — keep them lean and
# add fetcher-specific options here as the fetchers land.


class _TaskBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RssTask(_TaskBase):
    kind: Literal["rss"]
    url: str = Field(min_length=1)


class ArxivTask(_TaskBase):
    kind: Literal["arxiv"]
    categories: list[str] = Field(min_length=1)


class HnTask(_TaskBase):
    kind: Literal["hn"]
    list: Literal["top", "best", "new", "show", "ask"]
    min_score: int | None = None
    limit: int | None = None


class RedditTask(_TaskBase):
    kind: Literal["reddit"]
    subreddit: str = Field(min_length=1)
    sort: Literal["hot", "new", "top", "rising"] = "hot"
    limit: int | None = None


class GdeltTask(_TaskBase):
    kind: Literal["gdelt"]
    query: str = Field(min_length=1)


class SecEdgarTask(_TaskBase):
    kind: Literal["sec_edgar"]
    forms: list[str] = Field(min_length=1)
    ciks: list[str] = Field(default_factory=list)


class GithubSearchTask(_TaskBase):
    kind: Literal["github_search"]
    query: str = Field(min_length=1)


class GithubTrendingTask(_TaskBase):
    kind: Literal["github_trending"]
    language: str | None = None


class WebsiteTask(_TaskBase):
    kind: Literal["website"]
    url: str = Field(min_length=1)


# ── Phase 7 (topical search) task kinds ──────────────────────────────────────
#
# These kinds are produced by the topic orchestrator (Phase 7e) from a
# :class:`SearchPlan`, not by the daily YAML resolver. They're absent from
# ``_DB_TYPE_TO_KIND`` on purpose — they aren't dashboard-managed sources.


class HnAlgoliaTask(_TaskBase):
    """One HN Algolia search request (Phase 7c).

    Algolia takes a single query string per request, so the topic
    orchestrator dispatches one :class:`HnAlgoliaTask` per query
    variant from the :class:`SearchPlan`. The fetcher emits
    ``source_type="hn"`` (same as the daily Firebase fetcher) so the
    same HN item discovered via either path dedupes naturally on
    ``UNIQUE(source_type, dedup_key)``; ``metadata.discovered_via``
    distinguishes ``"algolia"`` from ``"firebase"``.

    ``window_start_epoch`` is the unix epoch (UTC seconds) below which
    items are filtered out via Algolia's ``numericFilters`` param. The
    topic orchestrator supplies this from the run's window; the fetcher
    composes the actual ``created_at_i>{epoch}`` filter string.
    """

    kind: Literal["hn_algolia"]
    query: str = Field(min_length=1)
    tags: str = "story"  # "story" | "comment" | "(story,comment)" — Algolia syntax
    window_start_epoch: int | None = None
    hits_per_page: int = Field(default=50, ge=1, le=1000)


class RawCacheTask(_TaskBase):
    """Search the existing ``raw_items`` cache for a topic (Phase 7c).

    Surfaces items that prior daily runs already collected, matched
    against any of the ``query_variants`` via case-insensitive LIKE on
    title + canonical_url + content. The fetcher emits each matched
    raw_item's original ``source_type`` and ``dedup_key`` unchanged so
    the runner's :func:`db.upsert_raw_item` no-ops the row and adds the
    topic-run linkage via ``run_raw_items``. Same item discovered by
    multiple variants dedupes naturally via SQL ``DISTINCT``.

    ``window_start`` (ISO UTC) bounds the matched items by
    ``published_at`` (falling back to ``fetched_at`` when null) so a
    "Khosla Ventures last 30 days" topic doesn't surface 2-year-old
    articles. ``limit`` caps the total result count.
    """

    kind: Literal["raw_cache"]
    query_variants: list[str] = Field(min_length=1)
    window_start: str | None = None
    limit: int = Field(default=200, ge=1, le=2000)


SourceTaskUnion = (
    RssTask
    | ArxivTask
    | HnTask
    | RedditTask
    | GdeltTask
    | SecEdgarTask
    | GithubSearchTask
    | GithubTrendingTask
    | WebsiteTask
    | HnAlgoliaTask
    | RawCacheTask
)
SourceTask = Annotated[SourceTaskUnion, Field(discriminator="kind")]

_TASK_ADAPTER: TypeAdapter[SourceTaskUnion] = TypeAdapter(SourceTask)

KNOWN_TASK_KINDS: frozenset[str] = frozenset(
    {
        "rss",
        "arxiv",
        "hn",
        "reddit",
        "gdelt",
        "sec_edgar",
        "github_search",
        "github_trending",
        "website",
        # Phase 7 topical-search kinds. Not in _DB_TYPE_TO_KIND —
        # produced by the topic orchestrator, not by dashboard sources.
        "hn_algolia",
        "raw_cache",
    }
)


# Maps the ClawFeed ``sources.type`` column to a fetcher kind. Twitter types
# are intentionally absent — out of scope for v1 per the architecture doc.
# ``digest_feed`` is a ClawFeed-internal type (subscribe to another digest
# feed) and never participates in intel fetching.
_DB_TYPE_TO_KIND: dict[str, str] = {
    "rss": "rss",
    "atom": "rss",
    "website": "website",
    "hackernews": "hn",
    "hn": "hn",
    "reddit": "reddit",
    "github_trending": "github_trending",
}


# ── Resolved plan shapes ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProfileConfig:
    timezone: str = "UTC"
    daily_window_hours: int = 24
    default_language: str = "en"


@dataclass(frozen=True)
class ResolvedTask:
    """One concrete fetcher task plus where it came from.

    Provenance lets later stages explain in coverage *why* a source was
    queried (a specific dashboard row vs an editorial config entry), and lets
    fetcher failures be attributed back to the right row.
    """

    task: SourceTaskUnion
    category: str
    origin: Literal["yaml", "db"]
    source_id: int | None
    source_name: str

    @property
    def kind(self) -> str:
        return self.task.kind


@dataclass(frozen=True)
class PlanWarning:
    """A single soft failure noted while resolving the plan."""

    origin: Literal["config", "yaml", "db"]
    category: str | None
    message: str


@dataclass
class CategoryPlan:
    name: str
    description: str = ""
    include_rules: list[str] = field(default_factory=list)
    exclude_rules: list[str] = field(default_factory=list)
    tasks: list[ResolvedTask] = field(default_factory=list)


@dataclass
class SourcePlan:
    profile: ProfileConfig
    categories: list[CategoryPlan]
    dynamic_search: list[str]
    warnings: list[PlanWarning]

    def category(self, name: str) -> CategoryPlan | None:
        for cat in self.categories:
            if cat.name == name:
                return cat
        return None

    def tasks_by_kind(self) -> dict[str, list[ResolvedTask]]:
        """Group resolved tasks across all categories by fetcher kind.

        Fetchers consume one bucket each; grouping here keeps the dispatch
        loop in the orchestrator trivial.
        """
        out: dict[str, list[ResolvedTask]] = {kind: [] for kind in KNOWN_TASK_KINDS}
        for cat in self.categories:
            for resolved in cat.tasks:
                out[resolved.kind].append(resolved)
        return out


# ── Resolver entry point ──────────────────────────────────────────────────────


def build_source_plan(
    conn: sqlite3.Connection,
    *,
    config_path: Path | None = None,
) -> SourcePlan:
    """Resolve the daily fetch plan from YAML editorial config + DB tags.

    ``conn`` is read-only here; the resolver opens no transaction.

    The plan always has every YAML category as a :class:`CategoryPlan`, even
    when its source list is fully malformed — this preserves coverage
    accounting (the run can still record that the category was attempted) and
    keeps the per-category section in the final brief stable.
    """
    path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    warnings: list[PlanWarning] = []

    raw_config = _load_yaml_config(path, warnings)
    profile = _build_profile(raw_config.get("profile") or {})
    dynamic_search = _build_dynamic_search(raw_config.get("dynamic_search") or {})

    categories = _build_categories_from_yaml(raw_config.get("categories") or {}, warnings)
    _attach_db_sources(conn, categories, warnings)

    return SourcePlan(
        profile=profile,
        categories=categories,
        dynamic_search=dynamic_search,
        warnings=warnings,
    )


# ── YAML side ─────────────────────────────────────────────────────────────────


def _load_yaml_config(
    path: Path,
    warnings: list[PlanWarning],
) -> dict[str, Any]:
    if not path.exists():
        warnings.append(
            PlanWarning(
                origin="config",
                category=None,
                message=f"intel-sources config not found at {path}; YAML side is empty",
            )
        )
        return {}

    text = path.read_text(encoding="utf-8")
    # A YAML parse error is a deploy bug, not a runtime degradation. Let it
    # propagate — the orchestrator's outer try/except marks the run failed,
    # which is correct: we have no editorial intent to act on.
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(loaded).__name__}")
    return loaded


def _build_profile(raw: dict[str, Any]) -> ProfileConfig:
    return ProfileConfig(
        timezone=str(raw.get("timezone") or "UTC"),
        daily_window_hours=int(raw.get("daily_window_hours") or 24),
        default_language=str(raw.get("default_language") or "en"),
    )


def _build_dynamic_search(raw: dict[str, Any]) -> list[str]:
    enabled = raw.get("enabled_sources") or []
    if not isinstance(enabled, list):
        return []
    return [str(item) for item in enabled if item]


def _build_categories_from_yaml(
    raw: dict[str, Any],
    warnings: list[PlanWarning],
) -> list[CategoryPlan]:
    categories: list[CategoryPlan] = []
    if not isinstance(raw, dict):
        return categories

    for name, body in raw.items():
        if not isinstance(body, dict):
            warnings.append(
                PlanWarning(
                    origin="yaml",
                    category=str(name),
                    message=f"category body must be a mapping, got {type(body).__name__}; skipped",
                )
            )
            continue

        plan = CategoryPlan(
            name=str(name),
            description=str(body.get("description") or ""),
            include_rules=_string_list(body.get("include")),
            exclude_rules=_string_list(body.get("exclude")),
        )
        for entry in body.get("sources") or []:
            resolved = _resolve_yaml_entry(plan.name, entry, warnings)
            if resolved is not None:
                plan.tasks.append(resolved)
        categories.append(plan)

    return categories


def _resolve_yaml_entry(
    category: str,
    entry: Any,
    warnings: list[PlanWarning],
) -> ResolvedTask | None:
    if not isinstance(entry, dict):
        warnings.append(
            PlanWarning(
                origin="yaml",
                category=category,
                message=f"source entry must be a mapping, got {type(entry).__name__}; skipped",
            )
        )
        return None

    kind = entry.get("kind")
    if not kind:
        warnings.append(
            PlanWarning(
                origin="yaml",
                category=category,
                message=f"source entry missing 'kind': {entry!r}; skipped",
            )
        )
        return None
    if kind not in KNOWN_TASK_KINDS:
        warnings.append(
            PlanWarning(
                origin="yaml",
                category=category,
                message=f"unknown source kind {kind!r}; skipped",
            )
        )
        return None

    try:
        task = _TASK_ADAPTER.validate_python(entry)
    except ValidationError as exc:
        warnings.append(
            PlanWarning(
                origin="yaml",
                category=category,
                message=f"invalid {kind!r} entry: {_compact_validation_error(exc)}; skipped",
            )
        )
        return None

    return ResolvedTask(
        task=task,
        category=category,
        origin="yaml",
        source_id=None,
        source_name=f"{category}:{kind}",
    )


# ── DB side ───────────────────────────────────────────────────────────────────


def _attach_db_sources(
    conn: sqlite3.Connection,
    categories: list[CategoryPlan],
    warnings: list[PlanWarning],
) -> None:
    """Append tagged ClawFeed sources to their categories.

    A source can be tagged with multiple categories; the same row therefore
    appears as a task in each tagged category. ``source_id`` is preserved so
    fetcher failures can update ``source_fetch_state`` for the right row.
    """
    by_name = {cat.name: cat for cat in categories}
    rows = conn.execute(
        """
        SELECT s.id        AS source_id,
               s.name      AS source_name,
               s.type      AS source_type,
               s.config    AS source_config,
               sc.category AS category
          FROM sources s
          JOIN source_categories sc ON sc.source_id = s.id
         WHERE s.is_active = 1
         ORDER BY sc.category, s.id
        """
    ).fetchall()

    for row in rows:
        category = row["category"]
        plan = by_name.get(category)
        if plan is None:
            # Source tagged for a category the YAML doesn't know about. This
            # is a useful signal but shouldn't crash; the user may be
            # mid-edit. Materialize the category lazily so the source isn't
            # lost.
            plan = CategoryPlan(name=category)
            categories.append(plan)
            by_name[category] = plan

        resolved = _resolve_db_row(row, warnings)
        if resolved is not None:
            plan.tasks.append(resolved)


def _resolve_db_row(
    row: sqlite3.Row,
    warnings: list[PlanWarning],
) -> ResolvedTask | None:
    source_id = int(row["source_id"])
    source_name = row["source_name"] or f"source#{source_id}"
    source_type = (row["source_type"] or "").strip().lower()
    category = row["category"]

    kind = _DB_TYPE_TO_KIND.get(source_type)
    if kind is None:
        warnings.append(
            PlanWarning(
                origin="db",
                category=category,
                message=(
                    f"source #{source_id} ({source_name!r}) has unsupported type "
                    f"{source_type!r}; skipped"
                ),
            )
        )
        return None

    try:
        config = json.loads(row["source_config"] or "{}")
    except json.JSONDecodeError as exc:
        warnings.append(
            PlanWarning(
                origin="db",
                category=category,
                message=f"source #{source_id} config is not JSON: {exc}; skipped",
            )
        )
        return None
    if not isinstance(config, dict):
        warnings.append(
            PlanWarning(
                origin="db",
                category=category,
                message=f"source #{source_id} config is not an object; skipped",
            )
        )
        return None

    task_payload = _payload_from_db(kind, config)
    if task_payload is None:
        warnings.append(
            PlanWarning(
                origin="db",
                category=category,
                message=(
                    f"source #{source_id} ({source_name!r}) of kind {kind!r} is "
                    f"missing required config fields; skipped"
                ),
            )
        )
        return None

    try:
        task = _TASK_ADAPTER.validate_python(task_payload)
    except ValidationError as exc:
        warnings.append(
            PlanWarning(
                origin="db",
                category=category,
                message=(
                    f"source #{source_id} ({source_name!r}): "
                    f"{_compact_validation_error(exc)}; skipped"
                ),
            )
        )
        return None

    return ResolvedTask(
        task=task,
        category=category,
        origin="db",
        source_id=source_id,
        source_name=source_name,
    )


def _payload_from_db(kind: str, config: dict[str, Any]) -> dict[str, Any] | None:
    """Translate the ClawFeed-side ``sources.config`` JSON into a task payload.

    Returns ``None`` when the config doesn't carry the fields a fetcher needs,
    which surfaces as a warning rather than a validation crash.
    """
    if kind == "rss":
        url = config.get("url")
        if not url:
            return None
        return {"kind": "rss", "url": url}
    if kind == "website":
        url = config.get("url")
        if not url:
            return None
        return {"kind": "website", "url": url}
    if kind == "reddit":
        subreddit = config.get("subreddit")
        if not subreddit:
            return None
        payload: dict[str, Any] = {"kind": "reddit", "subreddit": subreddit}
        if (sort := config.get("sort")) in {"hot", "new", "top", "rising"}:
            payload["sort"] = sort
        if isinstance(config.get("limit"), int):
            payload["limit"] = config["limit"]
        return payload
    if kind == "hn":
        # ClawFeed stores the list selector under "filter" historically.
        list_value = config.get("list") or config.get("filter")
        if list_value not in {"top", "best", "new", "show", "ask"}:
            return None
        payload = {"kind": "hn", "list": list_value}
        if isinstance(config.get("min_score"), int):
            payload["min_score"] = config["min_score"]
        if isinstance(config.get("limit"), int):
            payload["limit"] = config["limit"]
        return payload
    if kind == "github_trending":
        language = config.get("language")
        # ClawFeed UI persists "all" to mean "any language"; normalize to None
        # so the fetcher only sees explicit language constraints.
        if isinstance(language, str) and language.lower() != "all":
            return {"kind": "github_trending", "language": language}
        return {"kind": "github_trending"}

    return None


# ── helpers ───────────────────────────────────────────────────────────────────


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _compact_validation_error(exc: ValidationError) -> str:
    """One-line summary of a Pydantic validation failure for warning messages."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{loc or '<root>'}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) or "validation failed"
