"""Performance profiles: small, reusable records of *measured* deployment behaviour.

A profile describes one deployment (hardware + model + backend + configuration) under
one workload shape, as a curve of load level → latency/throughput points. Capacity
planning, router simulation and cost optimisation read profiles, so they can run on a
laptop long after the GPU is gone. Profiles produced by the mock/simulated backends
carry ``simulated: true`` and must never be presented as GPU measurements.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from inferencelab.domain.slo import SLOSpec
from inferencelab.schema import MutableModel

__all__ = ["LoadLevel", "PercentileSet", "PerformanceProfile", "ProfilePoint"]


class LoadLevel(MutableModel):
    kind: Literal["concurrency", "request_rate"]
    value: float


class PercentileSet(MutableModel):
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    mean: float | None = None

    def get(self, percentile: float) -> float | None:
        return {50.0: self.p50, 90.0: self.p90, 95.0: self.p95, 99.0: self.p99}.get(
            float(percentile)
        )


class ProfilePoint(MutableModel):
    load: LoadLevel
    run_id: str
    requests: int
    request_throughput: float
    input_token_throughput: float
    output_token_throughput: float
    goodput_rps: float | None = None
    slo_attainment: float | None = None
    error_rate: float = 0.0
    ttft_ms: PercentileSet = Field(default_factory=PercentileSet)
    tpot_ms: PercentileSet = Field(default_factory=PercentileSet)
    itl_ms: PercentileSet = Field(default_factory=PercentileSet)
    e2e_ms: PercentileSet = Field(default_factory=PercentileSet)
    mean_input_tokens: float | None = None
    mean_output_tokens: float | None = None
    gpu_memory_peak_gb: float | None = None


class PerformanceProfile(MutableModel):
    schema_version: int = 1
    profile_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    simulated: bool
    hardware: dict[str, str | int | float | None] = Field(default_factory=dict)
    model: dict[str, str | int | float | None] = Field(default_factory=dict)
    backend: dict[str, str | int | float | None] = Field(default_factory=dict)
    deployment: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    workload: dict[str, str | int | float | None] = Field(default_factory=dict)
    slo: SLOSpec | None = None
    points: list[ProfilePoint] = Field(default_factory=list)
    memory: dict[str, float | str | None] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def sorted_points(self) -> list[ProfilePoint]:
        return sorted(self.points, key=lambda p: p.request_throughput)

    def merge(self, other: PerformanceProfile) -> PerformanceProfile:
        """Combine load points of the same deployment/workload (e.g. a concurrency sweep)."""
        for key in ("hardware", "model", "backend", "deployment"):
            if getattr(self, key) != getattr(other, key):
                raise ValueError(f"cannot merge profiles with different {key}")
        if self.simulated != other.simulated:
            raise ValueError("cannot merge simulated and measured profiles")
        merged = self.model_copy(deep=True)
        merged.points = sorted(
            [*self.points, *other.points], key=lambda p: (p.load.kind, p.load.value)
        )
        return merged
