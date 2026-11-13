"""Deployment contract: one model served by one backend on specific hardware.

``Model`` ≠ ``Deployment``. The same checkpoint can be deployed as vLLM+BF16,
vLLM+AWQ or SGLang+AWQ; those are three deployments that must be benchmarked and
compared separately. Engine-neutral settings live here; backend launch builders
translate them into engine flags and reject combinations an engine cannot serve.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from inferencelab.domain.hardware import HardwareRef
from inferencelab.domain.model import Quantization
from inferencelab.schema import StrictModel

__all__ = [
    "BackendKind",
    "DeploymentRole",
    "DeploymentSpec",
    "LoRAAdapter",
    "LoRASpec",
    "Parallelism",
    "SpeculativeMethod",
    "SpeculativeSpec",
]

_SECRETISH = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|CREDENTIAL)", re.IGNORECASE)


class BackendKind(StrEnum):
    MOCK = "mock"
    SIMULATED = "simulated"
    OPENAI = "openai"
    VLLM = "vllm"
    SGLANG = "sglang"
    TENSORRT_LLM = "tensorrt_llm"

    @property
    def is_gpu_engine(self) -> bool:
        return self in (BackendKind.VLLM, BackendKind.SGLANG, BackendKind.TENSORRT_LLM)

    @property
    def is_local_simulation(self) -> bool:
        return self in (BackendKind.MOCK, BackendKind.SIMULATED)


class DeploymentRole(StrEnum):
    """Disaggregated serving splits prefill and decode onto different workers."""

    UNIFIED = "unified"
    PREFILL = "prefill"
    DECODE = "decode"


class Parallelism(StrictModel):
    tensor: int = Field(default=1, ge=1)
    pipeline: int = Field(default=1, ge=1)
    data: int = Field(default=1, ge=1)
    expert: bool = Field(default=False, description="Expert parallelism for MoE models")

    @property
    def gpus_required(self) -> int:
        return self.tensor * self.pipeline * self.data


class LoRAAdapter(StrictModel):
    name: str
    source: str = Field(description="HF repo id or path on the GPU host")
    revision: str | None = None


class LoRASpec(StrictModel):
    adapters: list[LoRAAdapter] = Field(min_length=1)
    max_loras: int = Field(default=1, ge=1, description="Adapters resident per batch")
    max_lora_rank: int = Field(default=16, ge=1)


class SpeculativeMethod(StrEnum):
    NGRAM = "ngram"
    DRAFT_MODEL = "draft_model"
    EAGLE = "eagle"
    EAGLE3 = "eagle3"
    MTP = "mtp"


class SpeculativeSpec(StrictModel):
    method: SpeculativeMethod
    draft_model: str | None = None
    num_speculative_tokens: int = Field(default=4, ge=1, le=32)

    @model_validator(mode="after")
    def _draft_required(self) -> SpeculativeSpec:
        needs_draft = {
            SpeculativeMethod.DRAFT_MODEL,
            SpeculativeMethod.EAGLE,
            SpeculativeMethod.EAGLE3,
        }
        if self.method in needs_draft and not self.draft_model:
            raise ValueError(f"speculative method {self.method.value!r} requires draft_model")
        return self


EngineArgValue = str | int | float | bool


class DeploymentSpec(StrictModel):
    name: str
    model: str = Field(description="Logical model name from the model registry")
    backend: BackendKind
    role: DeploymentRole = DeploymentRole.UNIFIED
    image: str | None = Field(default=None, description="Pinned container image (never ':latest')")
    hardware: HardwareRef
    quantization: Quantization = Quantization.BF16
    dtype: Literal["auto", "bfloat16", "float16"] = "auto"
    parallelism: Parallelism = Field(default_factory=Parallelism)
    max_context: int | None = Field(default=None, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1, description="Engine max running seqs")
    max_batched_tokens: int | None = Field(default=None, ge=1)
    gpu_memory_utilization: float | None = Field(default=None, gt=0.05, le=0.98)
    prefix_caching: bool | None = Field(default=None, description="None = engine default")
    chunked_prefill: bool | None = Field(default=None, description="None = engine default")
    lora: LoRASpec | None = None
    speculative: SpeculativeSpec | None = None
    engine_args: dict[str, EngineArgValue] = Field(
        default_factory=dict, description="Verbatim engine flags (escape hatch, recorded)"
    )
    served_model_name: str | None = None
    port: int = Field(default=8000, ge=1, le=65535)
    env: dict[str, str] = Field(default_factory=dict, description="Non-secret env vars")
    notes: str | None = None

    @field_validator("image")
    @classmethod
    def _image_pinned(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if "@sha256:" in value:
            return value
        name, _, tag = value.rpartition(":")
        if not name or "/" in tag or not tag:
            raise ValueError("image must carry an explicit version tag or digest")
        if tag == "latest":
            raise ValueError("':latest' is not reproducible; pin a version tag or digest")
        return value

    @field_validator("env")
    @classmethod
    def _no_secrets_in_env(cls, value: dict[str, str]) -> dict[str, str]:
        leaked = [k for k in value if _SECRETISH.search(k)]
        if leaked:
            raise ValueError(
                f"secret-like env vars {leaked} must not live in configs; "
                "provide them through the remote host's env file"
            )
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> DeploymentSpec:
        if self.backend.is_gpu_engine and self.hardware.count < self.parallelism.gpus_required:
            raise ValueError(
                f"parallelism needs {self.parallelism.gpus_required} GPUs "
                f"but hardware.count={self.hardware.count}"
            )
        if self.role is not DeploymentRole.UNIFIED and not self.backend.is_gpu_engine:
            raise ValueError("prefill/decode roles are only meaningful for GPU serving engines")
        return self

    @property
    def served_name(self) -> str:
        return self.served_model_name or self.model
