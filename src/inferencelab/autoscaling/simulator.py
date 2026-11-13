"""Discrete-time autoscaling simulator (CPU only, deterministic).

A fluid queue model of a replica fleet: each tick, arrivals join a shared queue and
ready replicas serve up to ``capacity_rps`` each. Replicas take ``startup_delay_s`` to
become ready (GPU model load) and are billed while starting. Latency per tick is the
profile's measured latency at the current per-replica load plus the time to drain the
queue (Little's law).

This is a *policy comparison tool*: it shows how policies trade SLO violations against
replica-hours under the same trace. It is not a model of Kubernetes, cold-start
variance or network effects, and its numbers must not be read as a validated forecast.
"""

from __future__ import annotations

import bisect
import csv
import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from inferencelab.autoscaling.policies import Observation, PolicyConfig, desired_replicas
from inferencelab.domain.hardware import PricingSpec
from inferencelab.profiles.model import PerformanceProfile
from inferencelab.schema import StrictModel

__all__ = [
    "AutoscaleConfig",
    "AutoscaleResult",
    "ReplicaModel",
    "Tick",
    "latency_curve",
    "simulate",
]


# ------------------------------------------------------------------------ traces
class PiecewiseTrace(StrictModel):
    kind: Literal["piecewise"] = "piecewise"
    points: list[tuple[float, float]] = Field(min_length=1, description="(t_s, req/s), linear")

    def rate(self, t: float) -> float:
        pts = self.points
        if t <= pts[0][0]:
            return pts[0][1]
        for (t0, r0), (t1, r1) in itertools.pairwise(pts):
            if t0 <= t <= t1:
                return r0 + (r1 - r0) * (t - t0) / (t1 - t0) if t1 > t0 else r1
        return pts[-1][1]


class DiurnalTrace(StrictModel):
    kind: Literal["diurnal"] = "diurnal"
    base_rps: float = Field(ge=0)
    peak_rps: float = Field(ge=0)
    period_s: float = Field(gt=0)
    phase_s: float = 0.0

    def rate(self, t: float) -> float:
        x = (1 - math.cos(2 * math.pi * (t - self.phase_s) / self.period_s)) / 2
        return self.base_rps + (self.peak_rps - self.base_rps) * x


class FileTrace(StrictModel):
    """CSV with ``t_s,rps`` rows (linear interpolation)."""

    kind: Literal["file"] = "file"
    path: str

    def load(self, base_dir: Path | None) -> PiecewiseTrace:
        path = Path(self.path)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        with path.open(encoding="utf-8") as fh:
            rows = [(float(r["t_s"]), float(r["rps"])) for r in csv.DictReader(fh)]
        return PiecewiseTrace(points=rows)


TraceShape = Annotated[PiecewiseTrace | DiurnalTrace | FileTrace, Field(discriminator="kind")]


# ------------------------------------------------------------------------ config
class ReplicaModel(StrictModel):
    capacity_rps: float | None = Field(
        default=None, gt=0, description="Sustainable req/s per replica"
    )
    base_latency_ms: float | None = Field(default=None, gt=0)
    profile: str | None = Field(
        default=None, description="Profile JSON to derive the latency curve"
    )
    latency_metric: Literal["ttft", "e2e"] = "ttft"
    latency_percentile: float = 95.0
    startup_delay_s: float = Field(default=180.0, ge=0)
    pricing: PricingSpec | None = None
    gpus_per_replica: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _source(self) -> ReplicaModel:
        if self.capacity_rps is None and self.profile is None:
            raise ValueError("replica model needs capacity_rps or a profile")
        return self


class AutoscaleConfig(StrictModel):
    name: str
    duration_s: float = Field(gt=0)
    tick_s: float = Field(default=1.0, gt=0)
    trace: TraceShape
    replica: ReplicaModel
    policy: PolicyConfig
    initial_replicas: int = Field(default=1, ge=0)
    latency_slo_ms: float = Field(
        gt=0, description="Per-tick latency above this counts as violation"
    )
    arrivals: Literal["fluid", "poisson"] = "fluid"
    seed: int = 0
    rate_smoothing_alpha: float = Field(default=0.3, gt=0, le=1)


# ------------------------------------------------------------------------ results
@dataclass(frozen=True, slots=True)
class Tick:
    t: float
    offered_rps: float
    arrivals: float
    served: float
    queue: float
    ready: int
    starting: int
    latency_ms: float
    violated: float


@dataclass(slots=True)
class AutoscaleResult:
    name: str
    policy: str
    ticks: list[Tick]
    total_arrivals: float = 0.0
    total_served: float = 0.0
    violating_requests: float = 0.0
    replica_hours: float = 0.0
    cost_usd: float | None = None
    scale_events: int = 0
    max_queue: float = 0.0
    final_queue: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def violation_fraction(self) -> float:
        return self.violating_requests / self.total_served if self.total_served else 0.0

    @property
    def mean_replicas(self) -> float:
        return (
            sum(t.ready + t.starting for t in self.ticks) / len(self.ticks) if self.ticks else 0.0
        )


def latency_curve(
    profile: PerformanceProfile, metric: str, percentile: float
) -> tuple[list[float], list[float]]:
    """(request throughput, latency ms) pairs from measured profile points."""
    xs: list[float] = []
    ys: list[float] = []
    for p in profile.sorted_points():
        value = getattr(p, f"{metric}_ms").get(percentile)
        if value is None:
            value = getattr(p, f"{metric}_ms").mean
        if value is not None:
            xs.append(p.request_throughput)
            ys.append(value)
    return xs, ys


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    if not xs:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1


def simulate(
    config: AutoscaleConfig,
    *,
    profile: PerformanceProfile | None = None,
    base_dir: Path | None = None,
) -> AutoscaleResult:
    trace = config.trace.load(base_dir) if isinstance(config.trace, FileTrace) else config.trace
    rm = config.replica
    xs: list[float] = []
    ys: list[float] = []
    if profile is not None:
        xs, ys = latency_curve(profile, rm.latency_metric, rm.latency_percentile)
    capacity = rm.capacity_rps or (max(xs) if xs else None)
    if capacity is None:
        raise ValueError(
            "cannot derive replica capacity: provide capacity_rps or a measured profile"
        )
    base_latency = rm.base_latency_ms or (ys[0] if ys else 0.0)

    def latency_at(per_replica_rps: float) -> float:
        return _interp(xs, ys, per_replica_rps) if xs else base_latency

    # Little's law needs the full request time (not TTFT) to estimate work in service.
    exs, eys = latency_curve(profile, "e2e", 50.0) if profile is not None else ([], [])

    def service_time_s(per_replica_rps: float) -> float:
        return (_interp(exs, eys, per_replica_rps) if exs else base_latency) / 1000.0

    rng = random.Random(config.seed)
    policy = config.policy
    result = AutoscaleResult(config.name, policy.kind, [])
    if profile is not None and profile.simulated:
        result.notes.append("latency curve comes from SIMULATED backends")
    ready = config.initial_replicas
    starting: list[float] = []  # ready-at times
    queue = 0.0
    smoothed_rate: float | None = None
    trend = 0.0
    last_scale_up = -math.inf
    down_window: list[tuple[float, int]] = []
    hourly = rm.pricing.hourly_cost_for(rm.gpus_per_replica) if rm.pricing else None
    cost = 0.0
    dt = config.tick_s
    n_ticks = math.ceil(config.duration_s / dt)
    for i in range(n_ticks):
        t = i * dt
        ready += sum(1 for s in starting if s <= t)
        starting = [s for s in starting if s > t]
        rate = max(0.0, trace.rate(t))
        expected = rate * dt
        poisson = config.arrivals == "poisson"
        arrivals = float(_poisson(rng, expected)) if poisson else expected
        cap = ready * capacity * dt
        backlog = queue + arrivals
        served = min(backlog, cap)
        queue = backlog - served
        per_replica = served / dt / ready if ready else 0.0
        if ready == 0:
            latency = math.inf if backlog > 0 else 0.0
        else:
            drain = queue / (ready * capacity) * 1000.0
            latency = latency_at(per_replica) + drain
        violated = served if latency > config.latency_slo_ms else 0.0
        if ready == 0 and arrivals > 0:
            violated = 0.0  # nothing served; the queue carries the debt forward
        result.ticks.append(
            Tick(t, rate, arrivals, served, queue, ready, len(starting), latency, violated)
        )
        result.total_arrivals += arrivals
        result.total_served += served
        result.violating_requests += violated
        result.max_queue = max(result.max_queue, queue)
        result.replica_hours += (ready + len(starting)) * dt / 3600.0
        if hourly is not None:
            cost += (ready + len(starting)) * hourly * dt / 3600.0
        a = config.rate_smoothing_alpha
        observed = arrivals / dt
        previous = smoothed_rate
        smoothed_rate = (
            observed if smoothed_rate is None else (1 - a) * smoothed_rate + a * observed
        )
        if previous is not None:
            trend = (1 - a) * trend + a * (smoothed_rate - previous) / dt
        if i % max(1, round(policy.evaluation_interval_s / dt)) != 0:
            continue
        obs = Observation(
            arrival_rate=smoothed_rate,
            queue=queue,
            in_service=served / dt * service_time_s(per_replica),
            demand_utilization=(arrivals + queue) / (max(ready, 1) * capacity * dt),
            current_replicas=ready + len(starting),
            ready_replicas=ready,
            capacity_rps_per_replica=capacity,
            arrival_trend=trend,
            startup_delay_s=rm.startup_delay_s,
        )
        want = desired_replicas(policy, obs, latency_at if xs else None)
        current = ready + len(starting)
        down_window.append((t, want))
        down_window = [
            (ts, w) for ts, w in down_window if ts > t - policy.scale_down_stabilization_s
        ]
        if want > current and t - last_scale_up >= policy.scale_up_cooldown_s:
            starting.extend([t + rm.startup_delay_s] * (want - current))
            last_scale_up = t
            result.scale_events += 1
        else:
            stable = max(w for _, w in down_window)  # HPA-style: highest recent recommendation
            if stable < current:
                remove = current - stable
                cancel = min(remove, len(starting))
                starting = starting[: len(starting) - cancel]
                ready -= remove - cancel
                result.scale_events += 1
    result.final_queue = queue
    result.cost_usd = cost if hourly is not None else None
    if queue > 0:
        result.notes.append(f"{queue:.0f} requests still queued at the end of the trace")
    return result


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 50:  # normal approximation for large means
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1
