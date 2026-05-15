"""LLM client chokepoint for the intelligence pipeline.

Every LLM HTTP call in the worker goes through this package. Pipeline
stages, fetchers, and tests must not invoke a model endpoint directly —
the indirection is what makes per-stage routing, retries, schema
validation, and ``llm_calls`` logging enforceable in one place.

Phase 1 step 8a ships the routing config + a happy-path dispatcher.
Step 8b adds retries, JSON-schema validation, repair fallback, and DB
logging. Step 11 adds the openclaw provider for final composition.
"""

from .client import CallResult, LLMClient, LLMSchemaError, RetryConfig
from .gemini_cli import (
    GeminiCliError,
    GeminiCliExitError,
    GeminiCliOutputError,
    GeminiCliProviderConfig as GeminiCliRuntimeConfig,
    GeminiCliResult,
    GeminiCliStallError,
    GeminiCliTimeoutError,
    gemini_cli_completion,
)
from .routing import (
    DEFAULT_CONFIG_PATH,
    FallbackConfig,
    GeminiCliProviderConfig,
    ProvidersConfig,
    RoutingConfig,
    StageConfig,
    VmlxProviderConfig,
    load_routing,
)
from .schemas import ClusterSummaryPayload, RelevanceBatchResponse, RelevanceVerdict

__all__ = [
    "CallResult",
    "ClusterSummaryPayload",
    "DEFAULT_CONFIG_PATH",
    "FallbackConfig",
    "GeminiCliError",
    "GeminiCliExitError",
    "GeminiCliOutputError",
    "GeminiCliProviderConfig",
    "GeminiCliResult",
    "GeminiCliRuntimeConfig",
    "GeminiCliStallError",
    "GeminiCliTimeoutError",
    "LLMClient",
    "LLMSchemaError",
    "ProvidersConfig",
    "RelevanceBatchResponse",
    "RelevanceVerdict",
    "RetryConfig",
    "RoutingConfig",
    "StageConfig",
    "VmlxProviderConfig",
    "gemini_cli_completion",
    "load_routing",
]
