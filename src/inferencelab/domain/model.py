"""Model registry entries.

A *model* is a checkpoint (weights + tokenizer at a pinned revision). It is deliberately
separate from a *deployment* (how a backend serves that checkpoint on some hardware).
Weights are never stored in this repository; only references to them.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import Field, field_validator

from inferencelab.schema import StrictModel

__all__ = ["LicenseInfo", "ModelSpec", "Quantization"]

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA = re.compile(r"^[0-9a-f]{40}$")


class Quantization(StrEnum):
    """Weight format of a checkpoint or the serving precision of a deployment."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    AWQ = "awq"  # 4-bit weight-only (W4A16), group-wise scales
    GPTQ = "gptq"  # usually 4-bit weight-only
    INT4 = "int4"
    NVFP4 = "nvfp4"

    @property
    def approx_bits_per_weight(self) -> float:
        """Approximate storage bits per parameter, including group scales.

        Only used for a *lower-bound* weight-memory fit check before a GPU run; it is
        never used to predict throughput.
        """
        return {
            Quantization.BF16: 16.0,
            Quantization.FP16: 16.0,
            Quantization.FP8: 8.0,
            Quantization.INT8: 8.0,
            Quantization.AWQ: 4.25,
            Quantization.GPTQ: 4.25,
            Quantization.INT4: 4.25,
            Quantization.NVFP4: 4.5,
        }[self]


class LicenseInfo(StrictModel):
    id: str = Field(description="SPDX identifier or license name, e.g. Apache-2.0")
    url: str | None = None
    notes: str | None = None


class ModelSpec(StrictModel):
    name: str = Field(description="Logical model name used in requests and configs")
    family: str | None = None
    checkpoint: str = Field(description="Hugging Face repo id or local path reference")
    revision: str = Field(default="main", description="Commit SHA (preferred) or branch")
    tokenizer: str | None = None
    architecture: str | None = None
    parameter_count_b: float | None = Field(default=None, gt=0)
    context_length: int | None = Field(default=None, ge=1)
    quantization: Quantization = Quantization.BF16
    quantization_tool: str | None = Field(
        default=None,
        description="Toolchain that produced a pre-quantized checkpoint "
        "(e.g. autoawq, auto-gptq, modelopt, llm-compressor); engines differ in what they load",
    )
    license: LicenseInfo | None = None
    notes: str | None = None

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _NAME.match(value):
            raise ValueError("model name must be lowercase [a-z0-9._-]")
        return value

    @property
    def revision_pinned(self) -> bool:
        """True when ``revision`` is an immutable commit SHA rather than a branch."""
        return bool(_SHA.match(self.revision))

    @property
    def tokenizer_ref(self) -> str:
        return self.tokenizer or self.checkpoint

    def weight_memory_gb(self, quantization: Quantization | None = None) -> float | None:
        """Lower bound on weight memory (GB, 1e9 bytes). KV cache/activations excluded."""
        if self.parameter_count_b is None:
            return None
        q = quantization or self.quantization
        return self.parameter_count_b * q.approx_bits_per_weight / 8.0
