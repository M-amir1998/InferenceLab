"""Profile-driven capacity planning.

Question: *how many replicas does this deployment need to serve R requests/s while
meeting the SLO?* The answer is derived only from **measured** profile points:

1. Evaluate every measured load point against the SLO objectives (its own percentiles).
2. The highest-throughput passing point is the per-replica sustainable rate. Optional
   linear interpolation towards the first failing point is available but off by
   default (conservative).
3. ``replicas = ceil(R / (sustainable × (1 − headroom))) + redundancy``.

Throughput is never estimated from parameter count or GPU datasheets. If no measured
point passes, or the passing range never reached saturation, the plan says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import Field

from inferencelab.domain.hardware import HardwareRef
from inferencelab.domain.slo import SLOSpec
from inferencelab.profiles.model import PerformanceProfile, ProfilePoint
from inferencelab.schema import StrictModel

__all__ = [
    "CapacityPlan",
    "CapacityPlanConfig",
    "CapacityTarget",
    "PointVerdict",
    "plan_capacity",
    "point_meets_slo",
]


class CapacityTarget(StrictModel):
    request_rate: float = Field(gt=0, description="Peak requests/s to serve")
    mean_input_tokens: float | None = Field(default=None, gt=0)
    mean_output_tokens: float | None = Field(default=None, gt=0)


class CapacityPlanConfig(StrictModel):
    profile: str
    target: CapacityTarget
    slo: SLOSpec
    headroom: float = Field(default=0.2, ge=0, lt=0.95)
    redundancy: int = Field(default=0, ge=0, description="Extra replicas (N+k)")
    interpolate: bool = False
    hardware: HardwareRef | None = Field(default=None, description="Pricing for cost estimates")
    max_workload_mismatch: float = Field(default=0.25, gt=0)


@dataclass(frozen=True, slots=True)
class PointVerdict:
    point: ProfilePoint
    meets: bool | None  # None = insufficient data at this point
    details: list[str]


@dataclass(slots=True)
class CapacityPlan:
    simulated: bool
    verdicts: list[PointVerdict]
    sustainable_rps_per_replica: float | None = None
    limiting_point: ProfilePoint | None = None
    required_replicas: int | None = None
    gpus_per_replica: int = 1
    total_gpus: int | None = None
    hourly_cost_usd: float | None = None
    cost_per_1m_output_tokens_usd: float | None = None
    warnings: list[str] = field(default_factory=list)


def point_meets_slo(point: ProfilePoint, slo: SLOSpec) -> PointVerdict:
    details: list[str] = []
    verdict: bool | None = True
    for obj in slo.objectives:
        series = {
            "ttft": point.ttft_ms,
            "tpot": point.tpot_ms,
            "itl": point.itl_ms,
            "e2e": point.e2e_ms,
        }[obj.metric]
        value = series.get(obj.percentile)
        if value is None:
            details.append(f"{obj.label}: no measured p{obj.percentile:g} at this load")
            if verdict is True:
                verdict = None
            continue
        ok = value <= obj.threshold_ms
        details.append(f"{obj.label}: {value:.1f} ms {'ok' if ok else 'VIOLATED'}")
        if not ok:
            verdict = False
    if slo.availability is not None:
        ok = point.error_rate <= 1 - slo.availability
        details.append(f"error rate {point.error_rate:.4f} {'ok' if ok else 'VIOLATED'}")
        if not ok:
            verdict = False
    return PointVerdict(point, verdict, details)


def _interpolated(best: ProfilePoint, failing: ProfilePoint, slo: SLOSpec) -> float:
    """Throughput where the binding objective crosses its threshold (linear)."""
    candidates = [failing.request_throughput]
    for obj in slo.objectives:
        attr = f"{obj.metric}_ms"
        vb = getattr(best, attr).get(obj.percentile)
        vf = getattr(failing, attr).get(obj.percentile)
        if vb is None or vf is None or vf <= obj.threshold_ms or vf <= vb:
            continue
        frac = (obj.threshold_ms - vb) / (vf - vb)
        candidates.append(
            best.request_throughput + frac * (failing.request_throughput - best.request_throughput)
        )
    return max(best.request_throughput, min(candidates))


def plan_capacity(profile: PerformanceProfile, config: CapacityPlanConfig) -> CapacityPlan:
    points = profile.sorted_points()
    verdicts = [point_meets_slo(p, config.slo) for p in points]
    gpus = int(profile.hardware.get("count") or 1) or 1
    plan = CapacityPlan(simulated=profile.simulated, verdicts=verdicts, gpus_per_replica=gpus)
    if profile.simulated:
        plan.warnings.append("profile comes from SIMULATED backends: not a GPU capacity estimate")
    if not points:
        plan.warnings.append("profile has no measured points")
        return plan
    passing = [v.point for v in verdicts if v.meets is True]
    if not passing:
        plan.warnings.append("no measured load level meets the SLO; measure lower load levels")
        return plan
    best = max(passing, key=lambda p: p.request_throughput)
    sustainable = best.request_throughput
    higher_fail = [
        v.point
        for v in verdicts
        if v.meets is False and v.point.request_throughput > best.request_throughput
    ]
    if config.interpolate and higher_fail:
        sustainable = _interpolated(
            best, min(higher_fail, key=lambda p: p.request_throughput), config.slo
        )
    if best is points[-1]:
        plan.warnings.append(
            "the highest measured load still meets the SLO: saturation was not reached, so "
            "per-replica capacity is a lower bound (measure higher concurrency)"
        )
    usable = sustainable * (1.0 - config.headroom)
    if usable <= 0:
        plan.warnings.append("sustainable throughput is zero")
        return plan
    replicas = math.ceil(config.target.request_rate / usable) + config.redundancy
    plan.sustainable_rps_per_replica = sustainable
    plan.limiting_point = best
    plan.required_replicas = replicas
    plan.total_gpus = replicas * gpus
    _workload_warnings(plan, profile, config)
    pricing = config.hardware.pricing if config.hardware is not None else None
    if pricing is not None:
        plan.hourly_cost_usd = replicas * pricing.hourly_cost_for(gpus)
        out_tokens = config.target.mean_output_tokens or best.mean_output_tokens
        if out_tokens:
            tokens_per_hour = config.target.request_rate * out_tokens * 3600.0
            plan.cost_per_1m_output_tokens_usd = plan.hourly_cost_usd / tokens_per_hour * 1e6
    return plan


def _workload_warnings(
    plan: CapacityPlan, profile: PerformanceProfile, config: CapacityPlanConfig
) -> None:
    for name, target, measured in (
        ("input", config.target.mean_input_tokens, profile.workload.get("mean_input_tokens")),
        ("output", config.target.mean_output_tokens, profile.workload.get("mean_output_tokens")),
    ):
        if target is None or not isinstance(measured, int | float) or measured <= 0:
            continue
        drift = abs(target - measured) / measured
        if drift > config.max_workload_mismatch:
            plan.warnings.append(
                f"target mean {name} tokens ({target:g}) differ {drift * 100:.0f}% from the "
                f"profiled workload ({measured:g}); re-profile with a matching workload"
            )
