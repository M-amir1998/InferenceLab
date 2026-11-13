"""Cost model, profiles, capacity planning, autoscaling and experiment expansion."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from inferencelab.autoscaling.policies import (
    Observation,
    QueueLengthPolicy,
    RequestRatePolicy,
    SLOAwarePolicy,
    UtilizationPolicy,
    desired_replicas,
)
from inferencelab.autoscaling.simulator import AutoscaleConfig, latency_curve, simulate
from inferencelab.capacity.planner import (
    CapacityPlanConfig,
    CapacityTarget,
    plan_capacity,
    point_meets_slo,
)
from inferencelab.cost.model import compute_cost, estimate_cost_upper_bound
from inferencelab.domain.hardware import PricingSpec
from inferencelab.domain.slo import SLOSpec
from inferencelab.experiments.plan import ExperimentTooLargeError, expand_experiment
from inferencelab.experiments.spec import ExperimentSpec
from inferencelab.profiles.model import LoadLevel, PercentileSet, PerformanceProfile, ProfilePoint
from inferencelab.profiles.registry import ProfileRegistry

SLO = SLOSpec.model_validate({"ttft_p95_ms": 500})


def point(c: int, rps: float, ttft95: float | None, *, e2e50: float = 1000.0) -> ProfilePoint:
    return ProfilePoint(
        load=LoadLevel(kind="concurrency", value=c),
        run_id=f"run-{c}",
        requests=200,
        request_throughput=rps,
        input_token_throughput=rps * 100,
        output_token_throughput=rps * 50,
        ttft_ms=PercentileSet(p50=(ttft95 or 0) / 2, p95=ttft95),
        e2e_ms=PercentileSet(p50=e2e50, mean=e2e50),
        mean_output_tokens=50,
    )


def profile(points: list[ProfilePoint], simulated: bool = True) -> PerformanceProfile:
    return PerformanceProfile(
        profile_id="p",
        simulated=simulated,
        hardware={"gpu": "L4", "count": 1},
        points=points,
        workload={"mean_input_tokens": 100, "mean_output_tokens": 50},
    )


# ---------------------------------------------------------------- cost
def test_cost_report_all_in_figures() -> None:
    pricing = PricingSpec(
        hourly_cost_usd=2.0,
        unit="per_gpu_hour",
        price_source="manual",
        recorded_at=date(2026, 11, 13),
    )
    report = compute_cost(
        pricing,
        gpu_count=2,
        duration_s=1800,
        input_tokens=1_000_000,
        output_tokens=500_000,
        successful_requests=1000,
        slo_compliant_requests=500,
    )
    assert report.effective_hourly_cost_usd == 4.0 and report.total_cost_usd == pytest.approx(2.0)
    assert report.cost_per_1m_input_tokens_usd == pytest.approx(2.0)
    assert report.cost_per_1m_output_tokens_usd == pytest.approx(4.0)
    assert report.cost_per_slo_compliant_request_usd == pytest.approx(0.004)
    assert (
        compute_cost(
            pricing,
            gpu_count=1,
            duration_s=0,
            input_tokens=0,
            output_tokens=0,
            successful_requests=0,
            slo_compliant_requests=None,
        ).cost_per_1m_input_tokens_usd
        is None
    )
    with pytest.raises(ValueError):
        compute_cost(
            pricing,
            gpu_count=1,
            duration_s=-1,
            input_tokens=0,
            output_tokens=0,
            successful_requests=0,
            slo_compliant_requests=None,
        )
    assert estimate_cost_upper_bound(pricing, gpu_count=4, hours=2) == 16.0


# ---------------------------------------------------------------- profiles
def test_profile_registry_roundtrip_and_merge(tmp_path: Path) -> None:
    reg = ProfileRegistry(tmp_path)
    p1 = profile([point(1, 1.0, 100)])
    reg.save(p1)
    loaded = reg.load("p")
    assert loaded.points[0].request_throughput == 1.0
    merged = p1.merge(profile([point(8, 6.0, 300)]))
    assert [pt.load.value for pt in merged.points] == [1, 8]
    with pytest.raises(ValueError):
        p1.merge(profile([point(2, 2.0, 1)], simulated=False))
    assert reg.find(gpu="L4") and not reg.find(gpu="H100-80GB-SXM")
    with pytest.raises(ValueError):
        reg.load("../evil")
    assert ProfileRegistry(tmp_path / "missing").all() == []


# ---------------------------------------------------------------- capacity
def test_capacity_plan_uses_highest_passing_point() -> None:
    prof = profile([point(1, 1.0, 100), point(8, 6.0, 300), point(32, 10.0, 900)])
    cfg = CapacityPlanConfig(
        profile="x", target=CapacityTarget(request_rate=30), slo=SLO, headroom=0.2, redundancy=1
    )
    plan = plan_capacity(prof, cfg)
    assert plan.sustainable_rps_per_replica == 6.0
    assert plan.required_replicas == 7 + 1  # ceil(30 / 4.8) + 1
    assert any("SIMULATED" in w for w in plan.warnings)
    interp = plan_capacity(prof, cfg.model_copy(update={"interpolate": True}))
    assert 6.0 < (interp.sustainable_rps_per_replica or 0) < 10.0


def test_capacity_plan_edge_cases() -> None:
    cfg = CapacityPlanConfig(
        profile="x", target=CapacityTarget(request_rate=5, mean_output_tokens=500), slo=SLO
    )
    none_pass = plan_capacity(profile([point(1, 1.0, 900)]), cfg)
    assert none_pass.required_replicas is None and any(
        "no measured" in w for w in none_pass.warnings
    )
    never_saturated = plan_capacity(profile([point(1, 1.0, 100), point(2, 2.0, 120)]), cfg)
    assert any("saturation" in w for w in never_saturated.warnings)
    assert any("differ" in w for w in never_saturated.warnings)  # output tokens 500 vs 50
    assert point_meets_slo(point(1, 1.0, None), SLO).meets is None


def test_capacity_cost_with_pricing() -> None:
    cfg = CapacityPlanConfig.model_validate(
        {
            "profile": "x",
            "target": {"request_rate": 6},
            "slo": {"ttft_p95_ms": 500},
            "headroom": 0.0,
            "hardware": {"gpu": "L4", "count": 1, "hourly_cost_usd": 1.0, "price_source": "manual"},
        }
    )
    plan = plan_capacity(profile([point(1, 3.0, 100), point(4, 6.0, 900)]), cfg)
    assert plan.required_replicas == 2 and plan.hourly_cost_usd == 2.0
    assert plan.cost_per_1m_output_tokens_usd == pytest.approx(2.0 / (6 * 50 * 3600) * 1e6)


# ---------------------------------------------------------------- autoscaling
def obs(**kw: float) -> Observation:
    base = {
        "arrival_rate": 50.0,
        "queue": 0.0,
        "in_service": 0.0,
        "demand_utilization": 1.0,
        "current_replicas": 2,
        "ready_replicas": 2,
        "capacity_rps_per_replica": 10.0,
    }
    base.update(kw)
    return Observation(**base)  # type: ignore[arg-type]


def test_policy_formulas_and_clamping() -> None:
    assert desired_replicas(RequestRatePolicy(target_rps_per_replica=10), obs()) == 5
    assert (
        desired_replicas(UtilizationPolicy(target_utilization=0.5), obs(demand_utilization=1.0))
        == 4
    )
    assert (
        desired_replicas(
            QueueLengthPolicy(target_outstanding_per_replica=10), obs(queue=35, in_service=10)
        )
        == 5
    )
    assert desired_replicas(RequestRatePolicy(target_rps_per_replica=1, max_replicas=7), obs()) == 7
    assert (
        desired_replicas(RequestRatePolicy(target_rps_per_replica=1000, min_replicas=2), obs()) == 2
    )


def test_slo_aware_policy_looks_ahead() -> None:
    policy = SLOAwarePolicy(latency_budget_ms=500, headroom=0.0)
    flat = desired_replicas(policy, obs(arrival_rate=50.0), latency_at=lambda rps: rps * 40)
    rising = desired_replicas(
        policy,
        obs(arrival_rate=50.0, arrival_trend=0.1, startup_delay_s=300),
        latency_at=lambda rps: rps * 40,
    )
    assert flat == 6  # 50/6=8.3 rps -> 333 ms; 5 replicas -> 10 rps (at capacity, infeasible)
    assert rising > flat  # plans for 80 req/s expected after the startup delay


def test_simulator_compares_policies_deterministically() -> None:
    base = {
        "name": "t",
        "duration_s": 600,
        "tick_s": 5,
        "trace": {"kind": "piecewise", "points": [[0, 5], [300, 40], [600, 5]]},
        "replica": {
            "capacity_rps": 10,
            "base_latency_ms": 100,
            "startup_delay_s": 60,
            "pricing": {"hourly_cost_usd": 3.6, "price_source": "manual"},
        },
        "latency_slo_ms": 400,
        "initial_replicas": 1,
    }
    rr = simulate(
        AutoscaleConfig.model_validate(
            {**base, "policy": {"kind": "request_rate", "target_rps_per_replica": 7}}
        )
    )
    again = simulate(
        AutoscaleConfig.model_validate(
            {**base, "policy": {"kind": "request_rate", "target_rps_per_replica": 7}}
        )
    )
    assert (
        rr.violation_fraction == again.violation_fraction
        and rr.replica_hours == again.replica_hours
    )
    assert max(t.ready for t in rr.ticks) >= 5  # scaled for the 40 req/s peak
    assert rr.cost_usd == pytest.approx(rr.replica_hours * 3.6)
    assert rr.total_arrivals == pytest.approx(sum(t.arrivals for t in rr.ticks))
    static = simulate(
        AutoscaleConfig.model_validate(
            {
                **base,
                "policy": {
                    "kind": "request_rate",
                    "target_rps_per_replica": 1000,
                    "max_replicas": 1,
                },
            }
        )
    )
    assert static.violation_fraction > rr.violation_fraction  # one replica cannot cope


def test_simulator_with_profile_and_poisson() -> None:
    prof = profile([point(1, 2.0, 50, e2e50=500), point(8, 9.0, 300, e2e50=1500)])
    xs, ys = latency_curve(prof, "ttft", 95)
    assert xs == [2.0, 9.0] and ys == [50, 300]
    cfg = AutoscaleConfig.model_validate(
        {
            "name": "p",
            "duration_s": 300,
            "trace": {"kind": "diurnal", "base_rps": 2, "peak_rps": 20, "period_s": 300},
            "replica": {"profile": "x", "startup_delay_s": 30},
            "policy": {"kind": "slo_aware", "latency_budget_ms": 400},
            "latency_slo_ms": 400,
            "arrivals": "poisson",
            "seed": 3,
        }
    )
    result = simulate(cfg, profile=prof)
    assert any("SIMULATED" in n for n in result.notes)
    assert result.mean_replicas >= 1


def test_replica_model_requires_capacity_source() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        AutoscaleConfig.model_validate(
            {
                "name": "x",
                "duration_s": 10,
                "trace": {"kind": "piecewise", "points": [[0, 1]]},
                "replica": {},
                "policy": {"kind": "utilization"},
                "latency_slo_ms": 1,
            }
        )


# ---------------------------------------------------------------- experiments
def _write(tmp: Path, name: str, text: str) -> str:
    (tmp / name).write_text(text, encoding="utf-8")
    return name


def test_experiment_expansion_cap_excludes_and_estimates(tmp_path: Path) -> None:
    _write(tmp_path, "b1.yaml", "type: mock\nname: b1\n")
    _write(tmp_path, "b2.yaml", "type: mock\nname: b2\n")
    _write(tmp_path, "w.yaml", "name: w\nnum_requests: 100\narrival: {kind: poisson, rate: 10}\n")
    spec = ExperimentSpec.model_validate(
        {
            "name": "e",
            "mode": "local",
            "max_runs": 3,
            "matrix": {
                "backends": ["b1.yaml", "b2.yaml"],
                "workloads": ["w.yaml"],
                "request_rates": [5, 10],
            },
        }
    )
    with pytest.raises(ExperimentTooLargeError, match="4 runs"):
        expand_experiment(spec, tmp_path)
    narrowed = spec.model_copy(update={"exclude": [{"target": "b2", "load": 10}]})
    plan = expand_experiment(narrowed, tmp_path)
    assert [r.label for r in plan.runs] == ["b1-w-r5", "b1-w-r10", "b2-w-r5"]
    assert plan.runs[0].estimated_duration_s == pytest.approx(20.0)  # 100 requests / 5 req/s
    assert plan.estimated_duration_s is not None and not plan.requires_execute_flag


def test_remote_experiment_cost_upper_bound(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    _write(
        tmp_path,
        "dep.yaml",
        "name: d\nmodel: m\nbackend: vllm\nimage: vllm/vllm-openai:v0.33.0\nhardware: {gpu: L4, count: 1, hourly_cost_usd: 2.0, price_source: manual}\n",
    )
    _write(tmp_path, "w.yaml", "name: w\nnum_requests: 10\n")
    spec = ExperimentSpec.model_validate(
        {
            "name": "r",
            "mode": "remote",
            "max_runs": 10,
            "max_run_duration_s": 600,
            "model_load_estimate_s": 600,
            "per_run_overhead_s": 0,
            "matrix": {"deployments": ["dep.yaml"], "workloads": ["w.yaml"], "concurrency": [1, 8]},
        }
    )
    plan = expand_experiment(spec, tmp_path)
    assert plan.requires_execute_flag and plan.estimated_duration_s is None
    assert plan.duration_upper_bound_s == 600 * 2 + 600
    assert plan.estimated_cost_upper_bound_usd == pytest.approx(2.0 * 1800 / 3600)
    assert all(r.workload.duration_s == 600 for r in plan.runs)  # per-run cap enforced


def test_experiment_spec_validation() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="exactly one"):
        ExperimentSpec.model_validate(
            {"name": "x", "mode": "local", "max_runs": 1, "matrix": {"workloads": ["w"]}}
        )
    with pytest.raises(pydantic.ValidationError, match="local experiments"):
        ExperimentSpec.model_validate(
            {
                "name": "x",
                "mode": "local",
                "max_runs": 1,
                "matrix": {"deployments": ["d"], "workloads": ["w"]},
            }
        )
    with pytest.raises(pydantic.ValidationError, match="virtual_time"):
        ExperimentSpec.model_validate(
            {
                "name": "x",
                "mode": "remote",
                "max_runs": 1,
                "virtual_time": True,
                "matrix": {"deployments": ["d"], "workloads": ["w"]},
            }
        )
