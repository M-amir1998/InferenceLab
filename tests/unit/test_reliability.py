from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from inferencelab.clock import VirtualClock
from inferencelab.domain.errors import (
    BackendHTTPError,
    BackendTimeoutError,
    ErrorKind,
    StreamInterruptedError,
    TimeoutPhase,
)
from inferencelab.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from inferencelab.reliability.retry import RetryBudget, RetryConfig, RetryPolicy
from inferencelab.reliability.timeouts import next_with_timeout


def breaker(clock: VirtualClock, **kw: object) -> tuple[CircuitBreaker, list[tuple[str, str]]]:
    transitions: list[tuple[str, str]] = []
    cfg = CircuitBreakerConfig.model_validate(
        {"failure_threshold": 3, "open_duration_s": 10, "failure_rate_threshold": None, **kw}
    )
    return CircuitBreaker(
        cfg, clock, on_transition=lambda a, b: transitions.append((a.value, b.value))
    ), transitions


def fail(cb: CircuitBreaker, n: int = 1) -> None:
    for _ in range(n):
        permit = cb.acquire()
        assert permit is not None
        permit.failure()


# ---------------------------------------------------------------- circuit breaker
def test_closed_to_open_after_consecutive_failures() -> None:
    clock = VirtualClock()
    cb, transitions = breaker(clock)
    fail(cb, 2)
    ok = cb.acquire()
    assert ok is not None
    ok.success()  # resets the consecutive count
    fail(cb, 2)
    assert cb.state is CircuitState.CLOSED
    fail(cb)
    assert cb.state is CircuitState.OPEN and transitions == [("closed", "open")]
    assert cb.acquire() is None and cb.is_open()
    assert cb.retry_after_s == pytest.approx(10.0)


def test_open_to_half_open_to_closed() -> None:
    clock = VirtualClock()
    cb, transitions = breaker(clock, success_threshold=2, half_open_max_calls=1)
    fail(cb, 3)
    clock.advance(9.99)
    assert cb.state is CircuitState.OPEN
    clock.advance(0.02)
    assert cb.state is CircuitState.HALF_OPEN
    probe = cb.acquire()
    assert probe is not None and probe.probe
    assert cb.acquire() is None  # only one concurrent probe
    probe.success()
    second = cb.acquire()
    assert second is not None
    second.success()
    assert cb.state is CircuitState.CLOSED
    assert transitions == [("closed", "open"), ("open", "half_open"), ("half_open", "closed")]


def test_half_open_failure_reopens() -> None:
    clock = VirtualClock()
    cb, _ = breaker(clock)
    fail(cb, 3)
    clock.advance(10)
    probe = cb.acquire()
    assert probe is not None
    probe.failure()
    assert cb.state is CircuitState.OPEN
    assert cb.retry_after_s == pytest.approx(10.0)


def test_stale_results_from_previous_generation_are_ignored() -> None:
    clock = VirtualClock()
    cb, _ = breaker(clock)
    slow = cb.acquire()  # admitted while CLOSED
    fail(cb, 3)  # opens
    clock.advance(10)
    assert cb.state is CircuitState.HALF_OPEN
    assert slow is not None
    slow.failure()  # stale: must not re-open the half-open circuit
    assert cb.state is CircuitState.HALF_OPEN
    slow.failure()  # idempotent


def test_ignored_results_do_not_count() -> None:
    clock = VirtualClock()
    cb, _ = breaker(clock)
    for _ in range(10):
        permit = cb.acquire()
        assert permit is not None
        permit.ignore()
    assert cb.state is CircuitState.CLOSED


def test_failure_rate_trips_with_enough_calls() -> None:
    clock = VirtualClock()
    cb, _ = breaker(
        clock, failure_threshold=100, failure_rate_threshold=0.5, window_size=10, min_calls=10
    )
    for i in range(9):
        permit = cb.acquire()
        assert permit is not None
        permit.failure() if i % 2 == 1 else permit.success()
    assert cb.state is CircuitState.CLOSED  # 4/9 failures, window not yet full
    last = cb.acquire()
    assert last is not None
    last.failure()  # 5/10 = 50% >= threshold
    assert cb.state is CircuitState.OPEN


def test_disabled_breaker_always_admits() -> None:
    clock = VirtualClock()
    cb, _ = breaker(clock, enabled=False)
    fail(cb, 50)
    assert cb.acquire() is not None and cb.state is CircuitState.CLOSED


# ---------------------------------------------------------------- retries
def test_retry_rules_distinguish_output_started() -> None:
    policy = RetryPolicy(RetryConfig(max_attempts=3))
    err = BackendHTTPError(500, "x")
    assert policy.should_retry(err, attempts_made=1, output_started=False, streaming=True)
    assert not policy.should_retry(err, attempts_made=3, output_started=False, streaming=True)
    assert not policy.should_retry(
        BackendHTTPError(400, "x"), attempts_made=1, output_started=False, streaming=True
    )
    mid = StreamInterruptedError("cut")
    assert not policy.should_retry(mid, attempts_made=1, output_started=True, streaming=True)
    assert policy.should_retry(mid, attempts_made=1, output_started=True, streaming=False)
    strict = RetryPolicy(RetryConfig(retry_non_streaming_after_output=False))
    assert not strict.should_retry(mid, attempts_made=1, output_started=True, streaming=False)


def test_backoff_is_bounded_and_jittered_deterministically() -> None:
    cfg = RetryConfig(backoff_initial_s=0.1, backoff_max_s=0.3, backoff_multiplier=2, jitter=0.0)
    policy = RetryPolicy(cfg)
    assert [policy.backoff_s(n) for n in (0, 1, 2, 3, 4)] == [0.0, 0.1, 0.2, 0.3, 0.3]
    jittered = RetryPolicy(RetryConfig(jitter=0.5), seed=1)
    again = RetryPolicy(RetryConfig(jitter=0.5), seed=1)
    assert [jittered.backoff_s(2) for _ in range(5)] == [again.backoff_s(2) for _ in range(5)]


def test_retry_budget_caps_retries() -> None:
    clock = VirtualClock()
    budget = RetryBudget(ratio=0.1, min_retries_per_s=0.0, clock=clock)
    spent = 0
    while budget.try_spend():
        spent += 1
    assert spent == 10  # initial reserve
    for _ in range(100):
        budget.record_request()
    assert sum(budget.try_spend() for _ in range(50)) == 10  # 10% of 100 requests
    assert RetryBudget(None, 0.0, clock).try_spend()


# ---------------------------------------------------------------- timeouts
async def _gen(delays: list[float], clock: VirtualClock) -> AsyncIterator[int]:
    for i, d in enumerate(delays):
        await clock.sleep(d)
        yield i


async def test_next_with_timeout_phases(vclock: VirtualClock) -> None:
    async def first_token() -> None:
        await next_with_timeout(
            _gen([5.0], vclock), clock=vclock, timeout_s=1.0, phase=TimeoutPhase.FIRST_TOKEN
        )

    with pytest.raises(BackendTimeoutError) as exc:
        await vclock.run(first_token())
    assert exc.value.kind is ErrorKind.TIMEOUT_FIRST_TOKEN


async def test_deadline_wins_over_phase_timeout(vclock: VirtualClock) -> None:
    async def go() -> None:
        await next_with_timeout(
            _gen([5.0], vclock),
            clock=vclock,
            timeout_s=10.0,
            phase=TimeoutPhase.IDLE,
            deadline=vclock.now() + 2.0,
        )

    with pytest.raises(BackendTimeoutError) as exc:
        await vclock.run(go())
    assert exc.value.kind is ErrorKind.TIMEOUT_OVERALL


async def test_next_with_timeout_passes_values_and_end(vclock: VirtualClock) -> None:
    async def go() -> list[int]:
        stream = _gen([0.1, 0.1], vclock)
        out = [
            await next_with_timeout(stream, clock=vclock, timeout_s=1.0, phase=TimeoutPhase.IDLE)
            for _ in range(2)
        ]
        with pytest.raises(StopAsyncIteration):
            await next_with_timeout(stream, clock=vclock, timeout_s=1.0, phase=TimeoutPhase.IDLE)
        with pytest.raises(BackendTimeoutError):
            await next_with_timeout(
                stream, clock=vclock, timeout_s=1.0, phase=TimeoutPhase.IDLE, deadline=vclock.now()
            )
        return out

    assert await vclock.run(go()) == [0, 1]
