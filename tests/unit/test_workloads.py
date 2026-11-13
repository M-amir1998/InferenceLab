from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from inferencelab.domain.workload import (
    BucketLength,
    BurstArrival,
    ClosedLoopArrival,
    ConstantRateArrival,
    FixedLength,
    LengthBucket,
    PoissonArrival,
    StepArrival,
    StepStage,
    UniformLength,
    WorkloadSpec,
)
from inferencelab.workloads.arrivals import arrival_offsets
from inferencelab.workloads.generator import WorkloadPlan, synthetic_text
from inferencelab.workloads.traces import load_trace


def test_length_distributions() -> None:
    rng = random.Random(0)
    assert FixedLength(value=7).sample(rng) == 7
    u = UniformLength(min=3, max=5)
    assert {u.sample(rng) for _ in range(200)} == {3, 4, 5} and u.mean == 4
    b = BucketLength(
        buckets=[LengthBucket(min=1, max=1, weight=3), LengthBucket(min=100, max=100, weight=1)]
    )
    samples = [b.sample(rng) for _ in range(4000)]
    assert 0.70 < samples.count(1) / len(samples) < 0.80
    assert b.mean == pytest.approx((3 * 1 + 1 * 100) / 4)
    with pytest.raises(ValueError):
        UniformLength(min=5, max=1)


def test_constant_and_burst_arrivals_are_exact() -> None:
    rng = random.Random(0)
    assert list(arrival_offsets(ConstantRateArrival(rate=4), rng, duration_s=1.0)) == [
        0.0,
        0.25,
        0.5,
        0.75,
    ]
    burst = list(arrival_offsets(BurstArrival(burst_size=3, interval_s=2.0), rng, duration_s=4.0))
    assert burst == [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]


def test_poisson_rate_is_statistically_correct() -> None:
    offsets = list(arrival_offsets(PoissonArrival(rate=20), random.Random(1), duration_s=500))
    assert offsets == sorted(offsets)
    assert len(offsets) / 500 == pytest.approx(20, rel=0.05)


def test_step_arrivals_follow_stages() -> None:
    step = StepArrival(stages=[StepStage(rate=1, duration_s=10), StepStage(rate=4, duration_s=10)])
    offsets = list(arrival_offsets(step, random.Random(0)))
    assert sum(1 for t in offsets if t < 10) == 10 and sum(1 for t in offsets if t >= 10) == 40
    with pytest.raises(ValueError):
        list(arrival_offsets(ClosedLoopArrival(concurrency=1), random.Random(0)))


def test_synthetic_text_token_counts() -> None:
    text = synthetic_text(random.Random(0), 50)
    assert len(text.split()) == 50


def _spec(**kw: object) -> WorkloadSpec:
    base: dict[str, object] = {"name": "w", "num_requests": 10}
    base.update(kw)
    return WorkloadSpec.model_validate(base)


def test_plan_is_deterministic_and_respects_request_budget() -> None:
    spec = _spec(num_requests=25, input_tokens={"kind": "uniform", "min": 10, "max": 50})
    a = list(WorkloadPlan(spec).sessions())
    b = list(WorkloadPlan(spec).sessions())
    assert a == b and len(a) == 25
    assert all(10 <= t.input_tokens <= 50 for s in a for t in s.turns)


def test_shared_prefix_groups_and_keys() -> None:
    spec = _spec(
        num_requests=40,
        input_tokens={"kind": "fixed", "value": 120},
        shared_prefix={"num_prefixes": 3, "prefix_tokens": 100},
    )
    plan = WorkloadPlan(spec)
    sessions = list(plan.sessions())
    groups = {s.prefix_group for s in sessions}
    assert groups == {0, 1, 2}
    assert all(s.prefix_key == f"prefix-{s.prefix_group}" for s in sessions)
    assert all(s.turns[0].input_tokens == 20 for s in sessions)  # suffix only
    assert len((plan.prefix_text(0) or "").split()) == 100 and plan.prefix_text(None) is None


def test_sessions_turn_budget_and_ids() -> None:
    spec = _spec(
        num_requests=10, sessions={"turns": {"kind": "fixed", "value": 4}, "think_time_s": 1.0}
    )
    sessions = list(WorkloadPlan(spec).sessions())
    assert [len(s.turns) for s in sessions] == [4, 4, 2]
    assert sessions[0].session_id == "s000000" and sessions[0].think_time_s == 1.0


def test_tenant_priority_and_cancel_assignment() -> None:
    spec = _spec(
        num_requests=400,
        tenants=[{"tenant_id": "a", "weight": 1}, {"tenant_id": "b", "weight": 1}],
        priorities=[{"priority": "high", "weight": 1}],
        cancel_probability=0.5,
        cancel_after_tokens=3,
    )
    sessions = list(WorkloadPlan(spec).sessions())
    tenants = [s.tenant_id for s in sessions]
    assert 0.4 < tenants.count("a") / len(tenants) < 0.6
    assert {s.priority for s in sessions} == {"high"}
    cancels = [t.cancel_after_tokens for s in sessions for t in s.turns]
    assert 0.4 < sum(c is not None for c in cancels) / len(cancels) < 0.6


def test_duration_bounded_open_loop_plan() -> None:
    spec = WorkloadSpec.model_validate(
        {"name": "w", "duration_s": 2.0, "arrival": {"kind": "constant", "rate": 5}}
    )
    sessions = list(WorkloadPlan(spec).sessions())
    assert len(sessions) == 10 and sessions[-1].start_offset_s == pytest.approx(1.8)


def test_trace_loading_and_replay(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "timestamp_s": 0.0,
                    "conversation_id": "c1",
                },
                {"input_tokens": 20, "output_tokens": 6, "timestamp_s": 1.5, "prefix_key": "doc"},
            )
        )
        + "\n\n",
        encoding="utf-8",
    )
    spec = WorkloadSpec.model_validate(
        {"name": "w", "trace": {"path": "t.jsonl", "time_scale": 2.0}, "arrival": {"kind": "trace"}}
    )
    sessions = list(WorkloadPlan(spec, base_dir=tmp_path).sessions())
    assert [s.start_offset_s for s in sessions] == [0.0, 3.0]
    assert [s.turns[0].max_tokens for s in sessions] == [5, 6]
    assert sessions[0].session_id == "c1" and sessions[1].prefix_key == "doc"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"input_tokens": 5, "output_tokens": 5, "prompt": "secret"}', "text fields"),
        ('{"input_tokens": 0, "output_tokens": 5}', ">= 1"),
        ('{"output_tokens": 5}', "invalid trace entry"),
        ("not json", "invalid JSON"),
        ("[1]", "JSON object"),
    ],
)
def test_trace_validation(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_trace(path)


def test_trace_timestamps_must_be_ordered(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(
        '{"input_tokens": 1, "output_tokens": 1, "timestamp_s": 2}\n{"input_tokens": 1, "output_tokens": 1, "timestamp_s": 1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        load_trace(path)
