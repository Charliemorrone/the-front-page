"""Tests for the LLM routing config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from clawfeed_intel.llm import (
    DEFAULT_CONFIG_PATH,
    RoutingConfig,
    load_routing,
)


def _write_yaml(path: Path, config: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def base_config() -> dict[str, Any]:
    return {
        "providers": {
            "vmlx": {
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key_env": "VMLX_API_KEY",
            },
        },
        "stages": {
            "source_planning": {
                "provider": "vmlx",
                "model": "mlx-community/Qwen3-8B-4bit",
                "timeout_seconds": 90,
            },
            "relevance_filter": {
                "provider": "vmlx",
                "model": "mlx-community/Qwen3.5-27B-4bit",
                "timeout_seconds": 240,
                "batch_size": 12,
            },
        },
    }


# ── Default-path round-trip ───────────────────────────────────────────────


def test_default_config_path_points_at_shipped_yaml() -> None:
    """The shipped default config exists and is at the expected path."""
    assert DEFAULT_CONFIG_PATH.name == "model-routing.yaml"
    assert DEFAULT_CONFIG_PATH.exists()


def test_load_default_config_succeeds() -> None:
    """Phase 1 ships a YAML that validates and declares all four stages."""
    config = load_routing()
    assert isinstance(config, RoutingConfig)
    expected = {"source_planning", "relevance_filter", "cluster_summary", "final_compose"}
    assert expected <= set(config.stages)


def test_default_config_relevance_filter_has_batch_size() -> None:
    """Architecture-doc default is 12 per call; ship that value."""
    config = load_routing()
    assert config.resolve("relevance_filter").batch_size == 12


# ── Explicit-path loading ─────────────────────────────────────────────────


def test_load_explicit_path(tmp_path: Path, base_config: dict[str, Any]) -> None:
    path = _write_yaml(tmp_path / "routing.yaml", base_config)
    config = load_routing(path)
    assert config.providers.vmlx.base_url == "http://127.0.0.1:8080/v1"
    assert config.providers.vmlx.api_key_env == "VMLX_API_KEY"


def test_api_key_env_optional(tmp_path: Path, base_config: dict[str, Any]) -> None:
    """vMLX on loopback works without an API key — make the field optional."""
    del base_config["providers"]["vmlx"]["api_key_env"]
    path = _write_yaml(tmp_path / "routing.yaml", base_config)
    config = load_routing(path)
    assert config.providers.vmlx.api_key_env is None


def test_batch_size_optional(tmp_path: Path, base_config: dict[str, Any]) -> None:
    """Stages other than relevance_filter don't need batching."""
    config = load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))
    assert config.resolve("source_planning").batch_size is None


def test_strips_whitespace_on_strings(tmp_path: Path, base_config: dict[str, Any]) -> None:
    """``str_strip_whitespace`` defends against trailing newlines in YAML."""
    base_config["stages"]["source_planning"]["model"] = "  mlx-community/Qwen3-8B-4bit  "
    config = load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))
    assert config.resolve("source_planning").model == "mlx-community/Qwen3-8B-4bit"


# ── resolve() ─────────────────────────────────────────────────────────────


def test_resolve_returns_stage_config(tmp_path: Path, base_config: dict[str, Any]) -> None:
    config = load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))
    stage = config.resolve("relevance_filter")
    assert stage.model == "mlx-community/Qwen3.5-27B-4bit"
    assert stage.timeout_seconds == 240
    assert stage.batch_size == 12


def test_resolve_unknown_stage_raises(tmp_path: Path, base_config: dict[str, Any]) -> None:
    config = load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))
    with pytest.raises(KeyError, match="unknown stage 'made_up'"):
        config.resolve("made_up")


def test_resolve_unknown_stage_lists_known_stages(
    tmp_path: Path, base_config: dict[str, Any]
) -> None:
    """The error should help an engineer fix the typo without grepping the YAML."""
    config = load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))
    with pytest.raises(KeyError) as excinfo:
        config.resolve("typo")
    msg = str(excinfo.value)
    assert "relevance_filter" in msg and "source_planning" in msg


# ── Hard-failure surface ──────────────────────────────────────────────────


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_routing(tmp_path / "does-not-exist.yaml")


def test_top_level_not_mapping_raises(tmp_path: Path) -> None:
    """Sequence at root is a deploy-time YAML mistake — fail loudly."""
    path = tmp_path / "routing.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level YAML must be a mapping"):
        load_routing(path)


def test_top_level_scalar_raises(tmp_path: Path) -> None:
    path = tmp_path / "routing.yaml"
    path.write_text("just-a-string\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level YAML must be a mapping"):
        load_routing(path)


def test_extra_field_in_provider_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["providers"]["vmlx"]["unexpected"] = "value"
    path = _write_yaml(tmp_path / "routing.yaml", base_config)
    with pytest.raises(ValidationError):
        load_routing(path)


def test_extra_field_in_stage_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["stages"]["source_planning"]["mystery"] = 7
    path = _write_yaml(tmp_path / "routing.yaml", base_config)
    with pytest.raises(ValidationError):
        load_routing(path)


def test_extra_provider_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    """Until step 11 adds openclaw, declaring a new provider type is a typo."""
    base_config["providers"]["openclaw"] = {"base_url": "http://example"}
    path = _write_yaml(tmp_path / "routing.yaml", base_config)
    with pytest.raises(ValidationError):
        load_routing(path)


def test_unknown_provider_value_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    """A stage referencing a provider type we don't know about should fail loudly."""
    base_config["stages"]["source_planning"]["provider"] = "openclaw"
    path = _write_yaml(tmp_path / "routing.yaml", base_config)
    with pytest.raises(ValidationError):
        load_routing(path)


def test_missing_required_field_raises(tmp_path: Path, base_config: dict[str, Any]) -> None:
    del base_config["stages"]["source_planning"]["model"]
    path = _write_yaml(tmp_path / "routing.yaml", base_config)
    with pytest.raises(ValidationError):
        load_routing(path)


def test_empty_model_string_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["stages"]["source_planning"]["model"] = ""
    path = _write_yaml(tmp_path / "routing.yaml", base_config)
    with pytest.raises(ValidationError):
        load_routing(path)


def test_empty_base_url_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["providers"]["vmlx"]["base_url"] = ""
    path = _write_yaml(tmp_path / "routing.yaml", base_config)
    with pytest.raises(ValidationError):
        load_routing(path)


def test_zero_timeout_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["stages"]["source_planning"]["timeout_seconds"] = 0
    path = _write_yaml(tmp_path / "routing.yaml", base_config)
    with pytest.raises(ValidationError):
        load_routing(path)


def test_negative_batch_size_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["stages"]["relevance_filter"]["batch_size"] = -1
    path = _write_yaml(tmp_path / "routing.yaml", base_config)
    with pytest.raises(ValidationError):
        load_routing(path)


# ── Step 12b: gemini_cli provider + fallback shape ──────────────────────────


def test_gemini_cli_provider_loads(tmp_path: Path, base_config: dict[str, Any]) -> None:
    """The shipped default config declares the gemini_cli provider with
    every field the runtime needs to spawn the subprocess.
    """
    base_config["providers"]["gemini_cli"] = {
        "executable_path": "/usr/bin/node",
        "script_path": "/usr/bin/gemini",
        "approval_mode": "plan",
        "output_format": "stream-json",
        "idle_timeout_seconds": 30,
        "hard_timeout_seconds": 120,
        "retries": 2,
        "retry_backoff_seconds": 5,
    }
    config = load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))
    gem = config.providers.gemini_cli
    assert gem is not None
    assert gem.script_path == "/usr/bin/gemini"
    assert gem.executable_path == "/usr/bin/node"
    assert gem.idle_timeout_seconds == 30
    assert gem.hard_timeout_seconds == 120
    assert gem.retries == 2


def test_gemini_cli_provider_optional(tmp_path: Path, base_config: dict[str, Any]) -> None:
    """A vmlx-only deployment still loads cleanly without the provider."""
    config = load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))
    assert config.providers.gemini_cli is None


def test_stage_routes_to_gemini_cli_provider(tmp_path: Path, base_config: dict[str, Any]) -> None:
    """A stage may declare ``provider: gemini_cli`` after Step 12b."""
    base_config["providers"]["gemini_cli"] = {
        "script_path": "/usr/bin/gemini",
    }
    base_config["stages"]["final_compose"] = {
        "provider": "gemini_cli",
        "model": "gemini-2.5-pro",
        "timeout_seconds": 300,
    }
    config = load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))
    stage = config.resolve("final_compose")
    assert stage.provider == "gemini_cli"
    assert stage.model == "gemini-2.5-pro"


def test_stage_fallback_parses(tmp_path: Path, base_config: dict[str, Any]) -> None:
    """``FallbackConfig`` accepts provider + model with an optional timeout."""
    base_config["stages"]["final_compose"] = {
        "provider": "vmlx",
        "model": "primary-model",
        "timeout_seconds": 300,
        "fallback": {
            "provider": "vmlx",
            "model": "fallback-model",
            "timeout_seconds": 600,
        },
    }
    config = load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))
    fb = config.resolve("final_compose").fallback
    assert fb is not None
    assert fb.provider == "vmlx"
    assert fb.model == "fallback-model"
    assert fb.timeout_seconds == 600


def test_stage_fallback_timeout_optional(tmp_path: Path, base_config: dict[str, Any]) -> None:
    """``fallback.timeout_seconds`` omitted → ``None`` (compose layer inherits parent)."""
    base_config["stages"]["final_compose"] = {
        "provider": "vmlx",
        "model": "primary",
        "timeout_seconds": 300,
        "fallback": {"provider": "vmlx", "model": "fb"},
    }
    config = load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))
    assert config.resolve("final_compose").fallback.timeout_seconds is None


def test_stage_fallback_zero_timeout_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["stages"]["final_compose"] = {
        "provider": "vmlx",
        "model": "x",
        "timeout_seconds": 300,
        "fallback": {"provider": "vmlx", "model": "fb", "timeout_seconds": 0},
    }
    with pytest.raises(ValidationError):
        load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))


def test_stage_retries_field_optional_and_validated(
    tmp_path: Path, base_config: dict[str, Any]
) -> None:
    base_config["stages"]["final_compose"] = {
        "provider": "gemini_cli",
        "model": "gemini-2.5-pro",
        "timeout_seconds": 300,
        "retries": 2,
        "retry_backoff_seconds": 7.5,
    }
    base_config["providers"]["gemini_cli"] = {"script_path": "/x"}
    config = load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))
    stage = config.resolve("final_compose")
    assert stage.retries == 2
    assert stage.retry_backoff_seconds == 7.5


def test_stage_negative_retries_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["stages"]["final_compose"] = {
        "provider": "vmlx",
        "model": "x",
        "timeout_seconds": 300,
        "retries": -1,
    }
    with pytest.raises(ValidationError):
        load_routing(_write_yaml(tmp_path / "routing.yaml", base_config))


def test_default_config_routes_final_compose_to_gemini_cli() -> None:
    """The shipped default config (post-Step-12b) routes final_compose
    through ``gemini_cli`` with a vmlx fallback declared. Pinning this
    catches accidental regression of the production routing.
    """
    config = load_routing()
    stage = config.resolve("final_compose")
    assert stage.provider == "gemini_cli"
    assert stage.model == "gemini-2.5-pro"
    assert stage.fallback is not None
    assert stage.fallback.provider == "vmlx"
    assert config.providers.gemini_cli is not None
