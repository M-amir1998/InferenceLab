from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from inferencelab.backends.config import LatencySpec, MockBackendConfig
from inferencelab.backends.mock.backend import MockBackend
from inferencelab.benchmarking.loadgen import LoadGenerator, assign_phases
from inferencelab.benchmarking.targets import BackendTarget, EventKind, TargetEvent, TargetRequest
from inferencelab.clock import VirtualClock
from inferencelab.domain.errors import QueueFullError
from inferencelab.domain.response import TokenCountSource
from inferencelab.domain.result import Phase, RequestStatus
from inferencelab.domain.workload import WarmupSpec, WorkloadSpec
from inferencelab.workloads.generator import WorkloadPlan


def _mock(clock: VirtualClock, **kw: object) -> BackendTarget:
    cfg = MockBackendConfig.model_validate(
        {"name": "m", "latency": {"ttft_ms": 50, "tpot_ms": 10}, **kw}
    )
    return BackendTarget(MockBackend(cfg, clock=clock))


def _plan(**kw: object) -> WorkloadPlan:
    base: dict[str, object] = {
        "name": "w",
        "input_tokens": {"kind": "fixed", "value": 20},
        "output_tokens": {"kind": "fixed", "value": 5},
    }
    base.update(kw)
    return WorkloadPlan(WorkloadSpec.model_validate(base))


async def test_closed_loop_exact_count_and_concurrency(vclock: VirtualClock) -> None:
    lg = LoadGenerator(
        _mock(vclock),
        _plan(num_requests=40, arrival={"kind": "closed_loop", "concurrency": 4}),
        clock=vclock,
    )
    result = await vclock.run(lg.run())
    assert len(result.records) == 40
    assert all(r.status is RequestStatus.OK and r.output_tokens == 5 for r in result.records)
    assert all(r.token_source is TokenCountSource.REPORTED for r in result.records)
    # 40 requests / 4 workers, each 50 ms + 4*10 ms = 90 ms
    assert result.duration_s == pytest.approx(10 * 0.09)
    r0 = result.records[0]
    assert r0.ttft_s == pytest.approx(0.05) and r0.tpot_s == pytest.approx(0.01)
    assert len(r0.chunk_times) == 5 and r0.prompt_text is None and r0.prompt_fingerprint


async def test_open_loop_follows_schedule_exactly(vclock: VirtualClock) -> None:
    plan = _plan(duration_s=2.0, arrival={"kind": "constant", "rate": 10})
    result = await vclock.run(LoadGenerator(_mock(vclock), plan, clock=vclock).run())
    assert len(result.records) == 20
    sched = sorted(r.scheduled_at for r in result.records)
    assert sched == pytest.approx([i * 0.1 for i in range(20)])
    assert all(r.schedule_lag_s == pytest.approx(0.0) for r in result.records)


async def test_open_loop_drops_instead_of_delaying_when_saturated(vclock: VirtualClock) -> None:
    plan = _plan(duration_s=1.0, arrival={"kind": "constant", "rate": 20}, max_in_flight=2)
    target = _mock(vclock, latency=LatencySpec(ttft_ms=500, tpot_ms=100))
    result = await vclock.run(LoadGenerator(target, plan, clock=vclock).run())
    statuses = [r.status for r in result.records]
    assert len(statuses) == 20
    assert statuses.count(RequestStatus.DROPPED) > 0
    assert all(r.sent_at is None for r in result.records if r.status is RequestStatus.DROPPED)


async def test_client_cancellation_after_n_tokens(vclock: VirtualClock) -> None:
    plan = _plan(
        num_requests=10,
        cancel_probability=1.0,
        cancel_after_tokens=2,
        output_tokens={"kind": "fixed", "value": 20},
    )
    result = await vclock.run(LoadGenerator(_mock(vclock), plan, clock=vclock).run())
    assert all(
        r.status is RequestStatus.CANCELLED and len(r.chunk_times) == 2 for r in result.records
    )


async def test_request_timeout_and_errors_are_recorded(vclock: VirtualClock) -> None:
    hang = _plan(num_requests=3, request_timeout_s=1.0)
    res = await vclock.run(
        LoadGenerator(_mock(vclock, timeout_probability=1.0), hang, clock=vclock).run()
    )
    assert {r.status for r in res.records} == {RequestStatus.TIMEOUT}
    assert {r.error_kind for r in res.records} == {"client_timeout"}
    err = await vclock.run(
        LoadGenerator(
            _mock(vclock, failure_probability=1.0), _plan(num_requests=3), clock=vclock
        ).run()
    )
    assert {(r.status, r.error_kind, r.http_status) for r in err.records} == {
        (RequestStatus.ERROR, "backend_error", 502)
    }


async def test_rejections_are_classified() -> None:
    clock = VirtualClock()

    class Rejecting:
        description = "rejecting"

        async def stream(self, request: TargetRequest) -> AsyncGenerator[TargetEvent, None]:
            raise QueueFullError("full")
            yield TargetEvent(EventKind.FINAL)  # pragma: no cover

        async def aclose(self) -> None:
            return None

    res = await clock.run(LoadGenerator(Rejecting(), _plan(num_requests=2), clock=clock).run())
    assert {r.status for r in res.records} == {RequestStatus.REJECTED}


async def test_multi_turn_sessions_carry_history(vclock: VirtualClock) -> None:
    seen: list[int] = []

    class Recording:
        description = "recording"

        async def stream(self, request: TargetRequest) -> AsyncGenerator[TargetEvent, None]:
            seen.append(len(request.messages))
            assert request.conversation_id is not None
            yield TargetEvent(EventKind.CHUNK, text="answer", tokens=1)
            yield TargetEvent(EventKind.FINAL)

        async def aclose(self) -> None:
            return None

    plan = _plan(
        num_requests=3, sessions={"turns": {"kind": "fixed", "value": 3}, "system_prompt_tokens": 5}
    )
    res = await vclock.run(LoadGenerator(Recording(), plan, clock=vclock, capture_text=True).run())
    assert seen == [2, 4, 6]  # system + user, then +assistant +user per turn
    assert [r.turn for r in res.records] == [0, 1, 2]
    assert (
        res.records[0].output_text == "answer"
        and res.records[0].token_source is TokenCountSource.ESTIMATED
    )


def test_assign_phases_by_count_and_time() -> None:
    from inferencelab.domain.result import RequestRecord

    records = [
        RequestRecord(
            index=i,
            request_id=str(i),
            phase=Phase.MEASURE,
            status=RequestStatus.OK,
            scheduled_at=float(i),
            sent_at=float(i),
            first_token_at=None,
            completed_at=float(i),
        )
        for i in range(10)
    ]
    phased = assign_phases(records, WarmupSpec(requests=2), WarmupSpec(requests=3))
    assert [r.phase for r in phased].count(Phase.WARMUP) == 2
    assert [r.phase for r in phased].count(Phase.COOLDOWN) == 3
    timed = assign_phases(records, WarmupSpec(seconds=4.0), WarmupSpec(seconds=2.0))
    assert [r.index for r in timed if r.phase is Phase.MEASURE] == [4, 5, 6, 7]
    assert assign_phases([], WarmupSpec(), WarmupSpec()) == []
