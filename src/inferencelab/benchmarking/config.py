"""Benchmark run configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from inferencelab.backends.config import BackendConfig
from inferencelab.domain.deployment import DeploymentSpec
from inferencelab.domain.hardware import HardwareRef
from inferencelab.domain.model import ModelSpec
from inferencelab.domain.slo import SLOSpec
from inferencelab.domain.workload import WorkloadSpec
from inferencelab.gateway.config import GatewayConfig
from inferencelab.schema import StrictModel

__all__ = [
    "GPU_ENGINE_TYPES",
    "LOCAL_TYPES",
    "BackendTargetConfig",
    "BenchmarkConfig",
    "GatewayTargetConfig",
    "HTTPTargetConfig",
    "TargetConfig",
    "classify_backend_types",
]

LOCAL_TYPES = frozenset({"mock", "simulated"})
GPU_ENGINE_TYPES = frozenset({"vllm", "sglang", "tensorrt_llm"})


def classify_backend_types(types: set[str]) -> bool | None:
    """Simulated-ness of a set of backend types, requiring positive evidence either way.

    True: every backend is mock/simulated. False: every backend is a GPU serving engine.
    None: anything else (e.g. a generic OpenAI-compatible endpoint) - results are then
    labelled UNVERIFIED rather than ever being presented as GPU measurements.
    """
    if types and types <= LOCAL_TYPES:
        return True
    if types and types <= GPU_ENGINE_TYPES:
        return False
    return None


class HTTPTargetConfig(StrictModel):
    type: Literal["http"] = "http"
    base_url: str
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    max_connections: int = Field(default=1024, ge=1)
    connect_timeout_s: float = Field(default=10.0, gt=0)
    metrics_url: str | None = Field(default=None, description="Engine /metrics to snapshot")
    engine: Literal["vllm", "sglang", "tensorrt_llm", "inferencelab"] | None = None


class BackendTargetConfig(StrictModel):
    """Benchmark one backend directly, in-process."""

    type: Literal["backend"] = "backend"
    backend: BackendConfig


class GatewayTargetConfig(StrictModel):
    """Benchmark the gateway data plane in-process (no HTTP hop)."""

    type: Literal["gateway"] = "gateway"
    gateway: GatewayConfig


TargetConfig = Annotated[
    HTTPTargetConfig | BackendTargetConfig | GatewayTargetConfig, Field(discriminator="type")
]


class BenchmarkConfig(StrictModel):
    name: str
    description: str = ""
    execution: Literal["local", "gpu"] = Field(
        default="local", description="'gpu' runs need an explicit --execute (cost guard)"
    )
    target: TargetConfig
    workload: WorkloadSpec
    slo: SLOSpec | None = None
    model: ModelSpec | None = None
    deployment: DeploymentSpec | None = None
    hardware: HardwareRef | None = None
    simulated: bool | None = Field(default=None, description="Override backend-type detection")
    virtual_time: bool = Field(
        default=False,
        description="Run under a discrete-event clock (in-process local targets only)",
    )
    capture_text: bool = Field(default=False, description="Persist prompt/response text (opt-in)")
    min_tail_samples: int = Field(default=5, ge=1)
    output_dir: str = "results"
    estimated_duration_s: float | None = Field(default=None, gt=0)
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> BenchmarkConfig:
        if self.virtual_time:
            if isinstance(self.target, HTTPTargetConfig):
                raise ValueError("virtual_time requires an in-process target (backend or gateway)")
            if self.local_backend_types() is False:
                raise ValueError("virtual_time is only meaningful with mock/simulated backends")
        in_process = isinstance(self.target, BackendTargetConfig | GatewayTargetConfig)
        if self.execution == "gpu" and in_process and self.local_backend_types():
            raise ValueError("execution 'gpu' with only mock/simulated backends is contradictory")
        return self

    def local_backend_types(self) -> bool | None:
        """True if all backends are mock/simulated, False if all are GPU engines, else None."""
        if isinstance(self.target, BackendTargetConfig):
            return classify_backend_types({self.target.backend.type})
        if isinstance(self.target, GatewayTargetConfig):
            return classify_backend_types({b.type for b in self.target.gateway.backends})
        return None

    @property
    def hardware_ref(self) -> HardwareRef | None:
        if self.deployment is not None:
            return self.deployment.hardware
        return self.hardware
