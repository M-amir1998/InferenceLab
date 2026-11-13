"""Autoscaling policies (pure functions of an observation).

* ``queue_length``  KEDA-style: outstanding work (queued + in service) per replica.
* ``utilization``   HPA-style: ``ceil(replicas × demand / target)`` where demand is
                    offered work over capacity (can exceed 1, unlike busy-time
                    utilisation, so heavy overload scales up quickly).
* ``request_rate``  observed arrival rate / target rate per replica.
* ``slo_aware``     smallest N whose *predicted* latency (profile curve + queue drain)
                    stays within the latency budget.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from inferencelab.schema import StrictModel

__all__ = [
    "Observation",
    "PolicyConfig",
    "QueueLengthPolicy",
    "RequestRatePolicy",
    "SLOAwarePolicy",
    "UtilizationPolicy",
    "desired_replicas",
]


class _Common(StrictModel):
    min_replicas: int = Field(default=1, ge=0)
    max_replicas: int = Field(default=20, ge=1)
    evaluation_interval_s: float = Field(default=15.0, gt=0)
    scale_up_cooldown_s: float = Field(default=30.0, ge=0)
    scale_down_stabilization_s: float = Field(default=300.0, ge=0)


class QueueLengthPolicy(_Common):
    kind: Literal["queue_length"] = "queue_length"
    target_outstanding_per_replica: float = Field(default=8.0, gt=0)


class UtilizationPolicy(_Common):
    kind: Literal["utilization"] = "utilization"
    target_utilization: float = Field(default=0.7, gt=0, le=1)


class RequestRatePolicy(_Common):
    kind: Literal["request_rate"] = "request_rate"
    target_rps_per_replica: float = Field(gt=0)


class SLOAwarePolicy(_Common):
    kind: Literal["slo_aware"] = "slo_aware"
    latency_budget_ms: float = Field(gt=0)
    headroom: float = Field(default=0.1, ge=0, lt=1)
    lookahead_s: float | None = Field(
        default=None,
        ge=0,
        description="Forecast horizon for the arrival rate (default: replica startup delay)",
    )


PolicyConfig = Annotated[
    QueueLengthPolicy | UtilizationPolicy | RequestRatePolicy | SLOAwarePolicy,
    Field(discriminator="kind"),
]


@dataclass(frozen=True, slots=True)
class Observation:
    arrival_rate: float  # smoothed req/s
    queue: float
    in_service: float
    demand_utilization: float
    current_replicas: int  # ready + starting
    ready_replicas: int
    capacity_rps_per_replica: float
    arrival_trend: float = 0.0  # d(rate)/dt in req/s per second (smoothed)
    startup_delay_s: float = 0.0  # time for a new replica to become ready


LatencyFn = Callable[[float], float]  # per-replica req/s -> predicted latency (ms)


def desired_replicas(
    policy: PolicyConfig, obs: Observation, latency_at: LatencyFn | None = None
) -> int:
    if isinstance(policy, QueueLengthPolicy):
        want = math.ceil((obs.queue + obs.in_service) / policy.target_outstanding_per_replica)
    elif isinstance(policy, UtilizationPolicy):
        base = max(obs.ready_replicas, 1)
        want = math.ceil(base * obs.demand_utilization / policy.target_utilization)
    elif isinstance(policy, RequestRatePolicy):
        want = math.ceil(obs.arrival_rate / policy.target_rps_per_replica)
    else:
        want = _slo_replicas(policy, obs, latency_at)
    return max(policy.min_replicas, min(policy.max_replicas, want))


def _slo_replicas(policy: SLOAwarePolicy, obs: Observation, latency_at: LatencyFn | None) -> int:
    budget = policy.latency_budget_ms * (1.0 - policy.headroom)
    # Replicas ordered now are ready only after the startup delay: plan for the rate then.
    horizon = obs.startup_delay_s if policy.lookahead_s is None else policy.lookahead_s
    rate = max(obs.arrival_rate, obs.arrival_rate + obs.arrival_trend * horizon)
    for n in range(max(policy.min_replicas, 1), policy.max_replicas + 1):
        per_replica = rate / n
        if per_replica >= obs.capacity_rps_per_replica:
            continue  # this N cannot even keep up
        drain_ms = obs.queue / (n * obs.capacity_rps_per_replica - rate) * 1000.0
        base = latency_at(per_replica) if latency_at is not None else 0.0
        if base + drain_ms <= budget:
            return n
    return policy.max_replicas
