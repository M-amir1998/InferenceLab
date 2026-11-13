"""Expand an experiment into concrete runs and estimate duration/cost before executing."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

from inferencelab.config.loader import load_model_file
from inferencelab.cost.model import estimate_cost_upper_bound
from inferencelab.domain.deployment import DeploymentSpec
from inferencelab.domain.hardware import HardwareRef
from inferencelab.domain.workload import (
    ClosedLoopArrival,
    ConstantRateArrival,
    PoissonArrival,
    WorkloadSpec,
)
from inferencelab.experiments.spec import ExperimentSpec

__all__ = ["ExperimentPlan", "ExperimentTooLargeError", "PlannedRun", "expand_experiment"]


class ExperimentTooLargeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlannedRun:
    index: int
    target: str  # deployment or backend name
    target_path: Path
    workload: WorkloadSpec
    workload_path: Path
    load_kind: str  # "concurrency" | "request_rate" | "as_configured"
    load_value: float | None
    repeat: int
    estimated_duration_s: float | None

    @property
    def label(self) -> str:
        load = "" if self.load_value is None else f"-{self.load_kind[0]}{self.load_value:g}"
        rep = f"-r{self.repeat}" if self.repeat else ""
        return f"{self.target}-{self.workload.name}{load}{rep}"


@dataclass(slots=True)
class ExperimentPlan:
    spec: ExperimentSpec
    base_dir: Path
    runs: list[PlannedRun]
    deployments: dict[str, DeploymentSpec] = field(default_factory=dict)
    estimated_duration_s: float | None = None
    duration_upper_bound_s: float = 0.0
    estimated_cost_upper_bound_usd: float | None = None
    pricing: HardwareRef | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def requires_execute_flag(self) -> bool:
        return self.spec.mode == "remote"


def _apply_load(
    workload: WorkloadSpec, kind: str, value: float | None, cap_s: float
) -> WorkloadSpec:
    update: dict[str, object] = {}
    if kind == "concurrency" and value is not None:
        update["arrival"] = ClosedLoopArrival(concurrency=int(value))
    elif kind == "request_rate" and value is not None:
        update["arrival"] = PoissonArrival(rate=value)
    if workload.duration_s is None or workload.duration_s > cap_s:
        update["duration_s"] = cap_s  # hard per-run cap
    return workload.model_copy(update=update)


def _estimate(workload: WorkloadSpec) -> float | None:
    """Known run duration (open-loop schedules); None for closed loop (server-dependent)."""
    arrival = workload.arrival
    if isinstance(arrival, ClosedLoopArrival):
        return None
    if isinstance(arrival, ConstantRateArrival | PoissonArrival) and workload.num_requests:
        by_count = workload.num_requests / arrival.rate
        return min(by_count, workload.duration_s) if workload.duration_s else by_count
    return workload.duration_s


def expand_experiment(spec: ExperimentSpec, base_dir: Path) -> ExperimentPlan:
    targets = spec.matrix.deployments or spec.matrix.backends
    loads: list[tuple[str, float | None]] = [
        ("concurrency", float(c)) for c in spec.matrix.concurrency
    ]
    loads += [("request_rate", r) for r in spec.matrix.request_rates]
    if not loads:
        loads = [("as_configured", None)]
    combos = list(
        itertools.product(targets, spec.matrix.workloads, loads, range(spec.matrix.repeats))
    )
    workloads = {w: load_model_file(base_dir / w, WorkloadSpec) for w in spec.matrix.workloads}
    names: dict[str, str] = {}
    deployments: dict[str, DeploymentSpec] = {}
    for t in targets:
        path = base_dir / t
        if spec.mode == "remote":
            dep = load_model_file(path, DeploymentSpec)
            deployments[dep.name] = dep
            names[t] = dep.name
        else:
            names[t] = path.stem
    runs: list[PlannedRun] = []
    for target, wpath, (kind, value), repeat in combos:
        tname = names[target]
        wl = workloads[wpath]
        row = {
            "target": tname,
            "workload": wl.name,
            "load": value if value is not None else "",
            "repeat": repeat,
        }
        if any(all(row.get(k) == v for k, v in ex.items()) for ex in spec.exclude):
            continue
        applied = _apply_load(wl, kind, value, spec.max_run_duration_s)
        runs.append(
            PlannedRun(
                index=len(runs),
                target=tname,
                target_path=base_dir / target,
                workload=applied,
                workload_path=base_dir / wpath,
                load_kind=kind,
                load_value=value,
                repeat=repeat,
                estimated_duration_s=_estimate(applied),
            )
        )
    if len(runs) > spec.max_runs:
        raise ExperimentTooLargeError(
            f"experiment expands to {len(runs)} runs, above max_runs={spec.max_runs}; "
            "narrow the matrix, add excludes or raise max_runs deliberately"
        )
    plan = ExperimentPlan(spec=spec, base_dir=base_dir, runs=runs, deployments=deployments)
    known = [r.estimated_duration_s for r in runs if r.estimated_duration_s is not None]
    n_targets = len({r.target for r in runs})
    load_s = spec.model_load_estimate_s * n_targets if spec.mode == "remote" else 0.0
    overhead = spec.per_run_overhead_s * len(runs)
    plan.estimated_duration_s = sum(known) + load_s + overhead if len(known) == len(runs) else None
    plan.duration_upper_bound_s = spec.max_run_duration_s * len(runs) + load_s + overhead
    if len(known) != len(runs):
        plan.warnings.append(
            "closed-loop run durations depend on server speed; using the per-run cap "
            f"({spec.max_run_duration_s:g}s) for the upper bound"
        )
    pricing = spec.pricing
    if pricing is None and deployments:
        priced = [d.hardware for d in deployments.values() if d.hardware.pricing is not None]
        pricing = priced[0] if priced else None
    plan.pricing = pricing
    if pricing is not None and pricing.pricing is not None:
        plan.estimated_cost_upper_bound_usd = estimate_cost_upper_bound(
            pricing.pricing,
            gpu_count=max(1, pricing.count),
            hours=plan.duration_upper_bound_s / 3600.0,
        )
    elif spec.mode == "remote":
        plan.warnings.append("no hourly price configured: cost cannot be estimated")
    return plan
