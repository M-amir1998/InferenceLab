"""Hardware profiles and pricing.

Hardware profiles hold *specifications only* (memory size, bandwidth, architecture).
Prices are never hard-coded: they change constantly, so they are supplied per
experiment together with their source and the date they were observed.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from inferencelab.schema import StrictModel

__all__ = ["HardwareProfile", "HardwareRef", "PricingSpec"]


class HardwareProfile(StrictModel):
    name: str
    vendor: str = "nvidia"
    architecture: str
    compute_capability: str | None = None
    vram_gb: float = Field(gt=0)
    memory_type: str | None = None
    memory_bandwidth_gbps: float | None = Field(default=None, gt=0)
    fp8_tensor_cores: bool = False
    form_factor: str | None = None
    interconnect: str | None = None
    source: str = Field(description="Where the specification came from (datasheet URL)")
    notes: str | None = None


class PricingSpec(StrictModel):
    hourly_cost_usd: float = Field(ge=0)
    unit: Literal["per_gpu_hour", "per_instance_hour"] = "per_instance_hour"
    price_source: str = Field(min_length=1, description="e.g. 'manual', provider name or URL")
    recorded_at: date | None = Field(default=None, description="Date the price was observed")

    def hourly_cost_for(self, gpu_count: int) -> float:
        return self.hourly_cost_usd * (gpu_count if self.unit == "per_gpu_hour" else 1)


class HardwareRef(StrictModel):
    """Hardware a deployment runs on, optionally with experiment-time pricing.

    Flat on purpose so configs read naturally::

        hardware:
          gpu: H100-80GB-SXM
          count: 1
          hourly_cost_usd: 2.50
          price_source: manual
          price_recorded_at: 2026-11-13
    """

    gpu: str = Field(description="Hardware profile name, or 'cpu' for local runs")
    count: int = Field(default=1, ge=0)
    hourly_cost_usd: float | None = Field(default=None, ge=0)
    price_unit: Literal["per_gpu_hour", "per_instance_hour"] = "per_instance_hour"
    price_source: str | None = None
    price_recorded_at: date | None = None

    @model_validator(mode="after")
    def _price_needs_source(self) -> HardwareRef:
        if self.hourly_cost_usd is not None and not self.price_source:
            raise ValueError(
                "hourly_cost_usd requires price_source (never record unsourced prices)"
            )
        return self

    @property
    def pricing(self) -> PricingSpec | None:
        if self.hourly_cost_usd is None or self.price_source is None:
            return None
        return PricingSpec(
            hourly_cost_usd=self.hourly_cost_usd,
            unit=self.price_unit,
            price_source=self.price_source,
            recorded_at=self.price_recorded_at,
        )
