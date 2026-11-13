"""Experiment matrices with an explicit run cap and a cost guard.

An experiment crosses deployments (or local backends) × workloads × load levels ×
repeats. The expansion is computed up front and **refused** if it exceeds
``max_runs`` — an accidental Cartesian product on paid GPUs is a real failure mode.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from inferencelab.domain.hardware import HardwareRef
from inferencelab.domain.slo import SLOSpec
from inferencelab.schema import StrictModel

__all__ = ["ExperimentSpec", "MatrixSpec"]


class MatrixSpec(StrictModel):
    deployments: list[str] = Field(
        default_factory=list, description="DeploymentSpec YAML paths (remote)"
    )
    backends: list[str] = Field(
        default_factory=list, description="Backend config YAML paths (local)"
    )
    workloads: list[str] = Field(min_length=1)
    concurrency: list[int] = Field(default_factory=list)
    request_rates: list[float] = Field(default_factory=list)
    repeats: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check(self) -> MatrixSpec:
        if bool(self.deployments) == bool(self.backends):
            raise ValueError(
                "matrix needs exactly one of 'deployments' (remote) or 'backends' (local)"
            )
        if any(c < 1 for c in self.concurrency) or any(r <= 0 for r in self.request_rates):
            raise ValueError("load levels must be positive")
        return self


class ExperimentSpec(StrictModel):
    name: str
    description: str = ""
    mode: Literal["local", "remote"]
    max_runs: int = Field(ge=1, description="Hard cap on the expanded matrix size")
    matrix: MatrixSpec
    exclude: list[dict[str, str | int | float]] = Field(default_factory=list)
    slo: SLOSpec | None = None
    models_dir: str = "../models"
    pricing: HardwareRef | None = Field(default=None, description="Overrides deployment pricing")
    max_run_duration_s: float = Field(default=900.0, gt=0, description="Per-run cap (enforced)")
    model_load_estimate_s: float = Field(default=600.0, ge=0)
    per_run_overhead_s: float = Field(default=30.0, ge=0)
    virtual_time: bool = Field(default=False, description="Local mode: run in simulated time")

    @model_validator(mode="after")
    def _mode(self) -> ExperimentSpec:
        if self.mode == "local" and not self.matrix.backends:
            raise ValueError("local experiments use matrix.backends")
        if self.mode == "remote" and not self.matrix.deployments:
            raise ValueError("remote experiments use matrix.deployments")
        if self.virtual_time and self.mode == "remote":
            raise ValueError("virtual_time is only available for local experiments")
        return self
