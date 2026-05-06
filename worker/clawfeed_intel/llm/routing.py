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


class ProvidersConfig(_ConfigBase):
    """Provider registry.

    Phase 1 ships with vmlx only. Step 11 adds an ``openclaw`` field here
    (WebSocket gateway transport, gpt-5.3-codex) for final composition.
    """

    vmlx: VmlxProviderConfig


class StageConfig(_ConfigBase):
    """One stage's dispatch rules.

    ``batch_size`` is consumed by the relevance filter (step 9, default 12
    per the architecture doc). Parsed here so adding the relevance filter
    doesn't need a schema bump.
    """

    # Widens to ``Literal["vmlx", "openclaw"]`` in step 11.
    provider: Literal["vmlx"]
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)
    batch_size: int | None = Field(default=None, gt=0)


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
