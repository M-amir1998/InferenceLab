from __future__ import annotations

import asyncio
import math

import pytest

from inferencelab.admission.controller import AdmissionController, QueueConfig, SheddingPolicy
from inferencelab.clock import VirtualClock
from inferencelab.domain.errors import LoadShedError, QueueFullError, RateLimitedError
from inferencelab.domain.request import Priority
from inferencelab.limits.tenant import TenantLimiter, TenantLimitsConfig, TenantQuota
from inferencelab.limits.token_bucket import TokenBucket


# ---------------------------------------------------------------- token bucket
def test_token_bucket_refill_and_capacity() -> None:
    clock = VirtualClock()
    bucket = TokenBucket(capacity=5, refill_per_s=2, clock=clock)
    assert all(bucket.try_consume() for _ in range(5))
    assert not bucket.try_consume()
    assert bucket.time_until(1) == pytest.approx(0.5)
    clock.advance(1.0)
    assert bucket.tokens == pytest.approx(2.0)
    clock.advance(100.0)
    assert bucket.tokens == pytest.approx(5.0)  # never above capacity
    bucket.refund(10)
    assert bucket.tokens == pytest.approx(5.0)
    assert bucket.time_until(6) == math.inf


def test_token_bucket_rejects_bad_arguments() -> None:
    clock = VirtualClock()
    with pytest.raises(ValueError):
        TokenBucket(0, 1, clock)
    with pytest.raises(ValueError):
        TokenBucket(1, -1, clock)
    bucket = TokenBucket(1, 1, clock)
    with pytest.raises(ValueError):
        bucket.try_consume(-1)
    with pytest.raises(ValueError):
        bucket.refund(-1)


# ---------------------------------------------------------------- tenants
def limiter(clock: VirtualClock, **quota: object) -> TenantLimiter:
    cfg = TenantLimitsConfig(
        tenants={"free": TenantQuota.model_validate(quota)},
        default=TenantQuota(requests_per_second=100),
    )
    return TenantLimiter(cfg, clock)


def test_requests_per_second_quota_with_retry_after() -> None:
    clock = VirtualClock()
    lim = limiter(clock, requests_per_second=2, burst=2)
    lim.acquire("free", 1).release()
    lim.acquire("free", 1).release()
    with pytest.raises(RateLimitedError) as exc:
        lim.acquire("free", 1)
    assert exc.value.retry_after_s == pytest.approx(0.5)
    clock.advance(0.5)
    lim.acquire("free", 1).release()


def test_concurrency_quota_and_release() -> None:
    clock = VirtualClock()
    lim = limiter(clock, max_concurrent=1)
    lease = lim.acquire("free", 1)
    with pytest.raises(RateLimitedError, match="concurrent"):
        lim.acquire("free", 1)
    lease.release()
    lease.release()  # idempotent
    assert lim.in_flight("free") == 0
    lim.acquire("free", 1)


def test_token_quota_reserves_and_refunds() -> None:
    clock = VirtualClock()
    lim = limiter(clock, tokens_per_minute=1000, requests_per_second=10)
    lease = lim.acquire("free", 800)
    with pytest.raises(RateLimitedError, match="tokens/min"):
        lim.acquire("free", 300)
    lease.release(actual_tokens=100)  # refund unused 700
    lim.acquire("free", 300)
    with pytest.raises(RateLimitedError, match="burst"):
        lim.acquire("free", 5000)


def test_tenants_are_isolated_and_unknown_tenants_get_default() -> None:
    clock = VirtualClock()
    lim = limiter(clock, requests_per_second=1, burst=1)
    lim.acquire("free", 1)
    with pytest.raises(RateLimitedError):
        lim.acquire("free", 1)
    for _ in range(50):
        lim.acquire("other", 1).release()
    no_default = TenantLimiter(TenantLimitsConfig(), clock)
    assert no_default.acquire(None, 10**9).reserved_tokens == 0  # unlimited


def test_tracked_tenant_state_is_bounded() -> None:
    clock = VirtualClock()
    cfg = TenantLimitsConfig(default=TenantQuota(requests_per_second=1), max_tracked_tenants=10)
    lim = TenantLimiter(cfg, clock)
    for i in range(100):
        lim.acquire(f"t{i}", 1).release()
    assert len(lim._states) <= 10


# ---------------------------------------------------------------- admission
async def test_admission_queues_by_priority_then_fifo(vclock: VirtualClock) -> None:
    ctrl = AdmissionController(QueueConfig(max_concurrency=1, max_pending_requests=10), vclock)
    order: list[str] = []

    async def worker(name: str, prio: Priority, hold: float) -> None:
        permit = await ctrl.acquire(prio)
        order.append(name)
        await vclock.sleep(hold)
        permit.release()

    async def main() -> None:
        first = asyncio.ensure_future(worker("first", Priority.NORMAL, 1.0))
        await asyncio.sleep(0)
        await asyncio.gather(
            first,
            worker("low", Priority.LOW, 0.1),
            worker("normal-a", Priority.NORMAL, 0.1),
            worker("high", Priority.HIGH, 0.1),
            worker("normal-b", Priority.NORMAL, 0.1),
        )

    await vclock.run(main())
    assert order == ["first", "high", "normal-a", "normal-b", "low"]
    assert ctrl.in_flight == 0 and ctrl.pending == 0


async def test_queue_full_rejects_newest(vclock: VirtualClock) -> None:
    ctrl = AdmissionController(QueueConfig(max_concurrency=1, max_pending_requests=1), vclock)

    async def main() -> None:
        held = await ctrl.acquire()
        waiter = asyncio.ensure_future(ctrl.acquire())
        await asyncio.sleep(0)
        with pytest.raises(QueueFullError):
            await ctrl.acquire()
        held.release()
        (await waiter).release()

    await vclock.run(main())


async def test_reject_lowest_priority_evicts_low_waiter(vclock: VirtualClock) -> None:
    cfg = QueueConfig(
        max_concurrency=1, max_pending_requests=1, shedding=SheddingPolicy.REJECT_LOWEST_PRIORITY
    )
    ctrl = AdmissionController(cfg, vclock)

    async def main() -> None:
        held = await ctrl.acquire()
        low = asyncio.ensure_future(ctrl.acquire(Priority.LOW))
        await asyncio.sleep(0)
        high = asyncio.ensure_future(ctrl.acquire(Priority.HIGH))
        with pytest.raises(LoadShedError):
            await low
        with pytest.raises(QueueFullError):  # equal priority cannot evict
            await ctrl.acquire(Priority.HIGH)
        held.release()
        (await high).release()

    await vclock.run(main())


async def test_max_queue_wait_sheds(vclock: VirtualClock) -> None:
    ctrl = AdmissionController(
        QueueConfig(max_concurrency=1, max_pending_requests=5, max_queue_wait_s=2.0), vclock
    )

    async def main() -> None:
        held = await ctrl.acquire()
        with pytest.raises(LoadShedError):
            await ctrl.acquire()
        assert ctrl.pending == 0
        held.release()

    await vclock.run(main())
    assert vclock.now() == pytest.approx(2.0)


async def test_slo_aware_shedding_uses_observed_service_time(vclock: VirtualClock) -> None:
    cfg = QueueConfig(
        max_concurrency=1,
        max_pending_requests=100,
        shedding=SheddingPolicy.SLO_AWARE,
        slo_queue_budget_s=1.5,
    )
    ctrl = AdmissionController(cfg, vclock)

    async def main() -> None:
        p = await ctrl.acquire()
        await vclock.sleep(1.0)
        p.release()  # observed service time = 1 s
        assert ctrl.predicted_wait_s(3) == pytest.approx(3.0)
        held = await ctrl.acquire()
        w1 = asyncio.ensure_future(ctrl.acquire())  # predicted wait 1 s: admitted to queue
        await asyncio.sleep(0)
        with pytest.raises(LoadShedError, match="predicted"):
            await ctrl.acquire()  # predicted wait 2 s > 1.5 s budget
        held.release()
        (await w1).release()

    await vclock.run(main())


async def test_cancelled_waiter_is_removed(vclock: VirtualClock) -> None:
    ctrl = AdmissionController(QueueConfig(max_concurrency=1, max_pending_requests=5), vclock)

    async def main() -> None:
        held = await ctrl.acquire()
        waiter = asyncio.ensure_future(ctrl.acquire())
        await asyncio.sleep(0)
        assert ctrl.pending == 1
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert ctrl.pending == 0
        held.release()
        assert ctrl.in_flight == 0

    await vclock.run(main())
