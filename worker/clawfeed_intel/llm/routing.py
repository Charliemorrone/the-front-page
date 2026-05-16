"""Per-stage LLM routing config.

Loaded from ``config/model-routing.yaml`` at startup. The architecture doc's
``Model Routing`` section is the source-of-truth for the YAML shape.

Validation posture mirrors :mod:`clawfeed_intel.sources`: tight pydantic
schemas with ``extra="forbid"`` so a YAML typo surfaces at load time, not
at call time. A non-mapping at root raises — that's a deploy bug, not a
degradation, and we want it loud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..paths import CONFIG_DIR

DEFAULT_CONFIG_PATH: Path = CONFIG_DIR / "model-routing.yaml"


class _ConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class VmlxProviderConfig(_ConfigBase):
    """vMLX OpenAI-compatible endpoint.

    ``api_key_env`` names an environment variable rather than carrying the
    key inline — secrets don't belong in YAML. vMLX accepts requests
    without an API key on loopback today, so the variable is optional.
    """

    base_url: str = Field(min_length=1)
    api_key_env: str | None = None


class GeminiCliProviderConfig(_ConfigBase):
    """Gemini CLI subprocess provider config (Step 12b).

    The provider invokes the Gemini CLI as a child process and reads
    stream-json events from its stdout. ``executable_path`` lets us
    bypass a broken PATH-resolved shebang (the local ``node@22`` is
    linked against an absent ``libsimdjson.30.dylib``); when omitted,
    we invoke ``script_path`` directly and rely on its shebang.

    **No ``api_key_env`` field, by design.** The CLI is signed into
    the operator's Gemini Pro subscription via OAuth at install time;
    the CLI manages refresh tokens internally. The worker never sees
    a Gemini API key — this is the subscription-via-CLI integration,
    not the pay-per-token Gemini API. See the architecture doc's
    Decision 4 amendment for why this is the explicit choice.

    Defaults mirror the dataclass in ``llm/gemini_cli.py``; both shapes
    coexist because the dataclass is the runtime contract of the async
    function and the pydantic model is the YAML-validation contract.
    Field names and defaults must stay in sync; a single ``from_yaml_*``
    converter lives on this class.
    """

    script_path: str = Field(min_length=1)
    executable_path: str | None = None
    approval_mode: str = "plan"
    output_format: str = "stream-json"
    idle_timeout_seconds: float = Field(default=60.0, gt=0)
    hard_timeout_seconds: float = Field(default=300.0, gt=0)
    retries: int = Field(default=1, ge=0)
    retry_backoff_seconds: float = Field(default=10.0, ge=0)


class ProvidersConfig(_ConfigBase):
    """Provider registry.

    Phase 1 shipped with vmlx only. Step 12b adds ``gemini_cli`` for
    the final-composition stage. Both are optional in the YAML so a
    deployment can ship either one alone; declaring a stage that
    references a missing provider fails at ``resolve`` time, not at
    YAML load.
    """

    vmlx: VmlxProviderConfig
    gemini_cli: GeminiCliProviderConfig | None = None


class FallbackConfig(_ConfigBase):
    """Secondary stage routing used when the primary provider fails.

    Compose-stage policy: Tier 1 = primary (``gemini_cli`` per the
    2026-05-15 amendment), Tier 2 = this fallback (local vMLX with
    the strongest cached model), Tier 3 = deterministic
    ``render_fallback_brief``. Tier 3 lives in the compose pure
    layer; Tier 2 is whatever this fallback config points at.

    Only ``provider`` + ``model`` are required. ``timeout_seconds``
    inherits the parent stage's value when omitted — the fallback is
    a different model, not a different patience budget.
    """

    provider: Literal["vmlx", "gemini_cli"]
    model: str = Field(min_length=1)
    timeout_seconds: float | None = Field(default=None, gt=0)


class StageConfig(_ConfigBase):
    """One stage's dispatch rules.

    ``batch_size`` is consumed by the relevance filter (step 9, default 12
    per the architecture doc). ``fallback`` is consumed by the compose
    stage (step 12b) to wire up the three-tier resilience chain; other
    stages may declare it but no current caller reads it for them.

    ``retries`` and ``retry_backoff_seconds`` are read by the
    ``gemini_cli`` provider only (the HTTP path uses its own
    tenacity-driven retry config on the client). The fields are
    optional + ignored for non-CLI providers so the YAML stays
    declarative.
    """

    provider: Literal["vmlx", "gemini_cli"]
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    retries: int | None = Field(default=None, ge=0)
    retry_backoff_seconds: float | None = Field(default=None, ge=0)
    fallback: FallbackConfig | None = None


class RoutingConfig(_ConfigBase):
    """Top-level routing config: providers + per-stage routing."""

    providers: ProvidersConfig
    stages: dict[str, StageConfig]

    def resolve(self, stage: str) -> StageConfig:
        """Return the stage config or raise ``KeyError`` with a helpful message."""
        try:
            return self.stages[stage]
        except KeyError:
            known = ", ".join(sorted(self.stages)) or "<none>"
            raise KeyError(f"unknown stage {stage!r}; known stages: {known}") from None


def load_routing(path: Path | None = None) -> RoutingConfig:
    """Load and validate the routing YAML.

    A non-mapping at root is a deploy bug — raise loudly rather than
    degrading. Pydantic ``ValidationError`` propagates from
    ``model_validate`` unchanged so the caller sees the field path.
    """
    config_path = path if path is not None else DEFAULT_CONFIG_PATH
    text = config_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{config_path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    return RoutingConfig.model_validate(raw)
