"""Declarative backend configuration (discriminated by ``type``)."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from inferencelab.backends.simulated.engine import SimulatedEngineConfig
from inferencelab.schema import StrictModel

__all__ = [
    "BackendConfig",
    "FaultMode",
    "FaultWindow",
    "HTTPBackendConfig",
    "LatencySpec",
    "MockBackendConfig",
    "OpenAIBackendConfig",
    "RoutingAttributes",
    "SGLangBackendConfig",
    "SimulatedBackendConfig",
    "TensorRTLLMBackendConfig",
    "VLLMBackendConfig",
]

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


class RoutingAttributes(StrictModel):
    """Per-backend metadata used by routing policies (never by the backend itself)."""

    tier: int = Field(default=0, ge=0, description="0 = preferred, 1 = secondary, 2+ = fallback")
    cost_per_1m_tokens_usd: float | None = Field(
        default=None, ge=0, description="Relative cost for cost-aware routing (from measurements)"
    )
    max_outstanding: int | None = Field(
        default=None,
        ge=1,
        description="Soft capacity: in-flight requests before the backend is 'full'",
    )


class _BackendBase(StrictModel):
    name: str
    models: list[str] = Field(default_factory=lambda: ["default"], min_length=1)
    model_map: dict[str, str] = Field(
        default_factory=dict, description="logical model name -> name served by the backend"
    )
    deployment: str | None = Field(default=None, description="Deployment label for reporting")
    routing: RoutingAttributes = Field(default_factory=RoutingAttributes)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _NAME.match(value):
            raise ValueError("backend name must match [A-Za-z0-9._-] and be <= 63 chars")
        return value

    def served_model(self, logical: str) -> str:
        return self.model_map.get(logical, logical)


# ------------------------------------------------------------------ local backends
class LatencySpec(StrictModel):
    ttft_ms: float = Field(default=50.0, ge=0)
    tpot_ms: float = Field(default=10.0, ge=0)
    jitter: float = Field(default=0.0, ge=0, le=1, description="Uniform +/- fraction")


class FaultMode(StrEnum):
    ERROR = "error"  # HTTP 500 before the first token
    TIMEOUT = "timeout"  # hang before the first token
    SLOW = "slow"  # multiply TTFT/TPOT by slow_factor
    DISCONNECT = "disconnect"  # break the stream mid-way
    MALFORMED = "malformed"  # emit an unparseable chunk mid-way
    DOWN = "down"  # health check fails, connections refused


class FaultWindow(StrictModel):
    mode: FaultMode
    start_s: float = Field(default=0.0, ge=0, description="Seconds after backend start")
    end_s: float | None = Field(default=None, description="None = until the end of time")
    probability: float = Field(default=1.0, ge=0, le=1)
    slow_factor: float = Field(default=5.0, ge=1)

    def active(self, elapsed_s: float) -> bool:
        return self.start_s <= elapsed_s and (self.end_s is None or elapsed_s < self.end_s)


class MockBackendConfig(_BackendBase):
    """Deterministic mock: fixed TTFT/TPOT with seeded jitter and fault injection."""

    type: Literal["mock"] = "mock"
    latency: LatencySpec = Field(default_factory=LatencySpec)
    output_tokens: int | None = Field(default=None, ge=0, description="None = honour max_tokens")
    failure_probability: float = Field(default=0.0, ge=0, le=1)
    timeout_probability: float = Field(default=0.0, ge=0, le=1)
    disconnect_probability: float = Field(default=0.0, ge=0, le=1)
    malformed_probability: float = Field(default=0.0, ge=0, le=1)
    faults: list[FaultWindow] = Field(default_factory=list)
    max_concurrency: int | None = Field(default=None, ge=1)
    seed: int = 0


class SimulatedBackendConfig(_BackendBase):
    """CPU discrete-time model of a continuous-batching engine (not GPU physics)."""

    type: Literal["simulated"] = "simulated"
    engine: SimulatedEngineConfig = Field(default_factory=SimulatedEngineConfig)


# ------------------------------------------------------------------ HTTP backends
class HTTPBackendConfig(_BackendBase):
    base_url: str = Field(description="Root URL of the OpenAI-compatible server")
    api_key_env: str | None = Field(
        default=None, description="Name of the environment variable holding the API key"
    )
    connect_timeout_s: float = Field(default=5.0, gt=0)
    read_timeout_s: float = Field(default=600.0, gt=0)
    max_connections: int = Field(default=512, ge=1)
    verify_tls: bool = True
    health_path: str = "/health"
    metrics_path: str | None = "/metrics"
    request_usage_in_stream: bool = Field(
        default=True, description="Send stream_options.include_usage (engine-reported counts)"
    )

    @field_validator("base_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https"):
            raise ValueError("base_url must use http or https")
        if not parts.hostname:
            raise ValueError("base_url needs a host")
        if parts.username or parts.password:
            raise ValueError("credentials must not be embedded in base_url; use api_key_env")
        return value.rstrip("/")

    @property
    def host(self) -> str:
        return urlsplit(self.base_url).hostname or ""


class OpenAIBackendConfig(HTTPBackendConfig):
    type: Literal["openai"] = "openai"


class VLLMBackendConfig(HTTPBackendConfig):
    type: Literal["vllm"] = "vllm"


class SGLangBackendConfig(HTTPBackendConfig):
    type: Literal["sglang"] = "sglang"


class TensorRTLLMBackendConfig(HTTPBackendConfig):
    type: Literal["tensorrt_llm"] = "tensorrt_llm"
    # trtllm-serve's /metrics is JSON iteration stats; Prometheus text lives here and
    # requires `return_perf_metrics: true` in the server's extra-options YAML.
    metrics_path: str | None = "/prometheus/metrics"


BackendConfig = Annotated[
    MockBackendConfig
    | SimulatedBackendConfig
    | OpenAIBackendConfig
    | VLLMBackendConfig
    | SGLangBackendConfig
    | TensorRTLLMBackendConfig,
    Field(discriminator="type"),
]
