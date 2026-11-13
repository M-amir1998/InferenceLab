from __future__ import annotations

import asyncio

import pytest

from inferencelab.clock import SystemClock, VirtualClock


async def test_virtual_sleep_advances_time_without_waiting(vclock: VirtualClock) -> None:
    async def work() -> float:
        await vclock.sleep(5.0)
        await vclock.sleep(2.5)
        return vclock.now()

    assert await vclock.run(work()) == pytest.approx(7.5)


async def test_concurrent_sleepers_wake_in_deadline_order(vclock: VirtualClock) -> None:
    order: list[str] = []

    async def sleeper(name: str, delay: float) -> None:
        await vclock.sleep(delay)
        order.append(f"{name}@{vclock.now():g}")

    async def main() -> None:
        await asyncio.gather(sleeper("b", 2.0), sleeper("a", 1.0), sleeper("c", 3.0))

    await vclock.run(main())
    assert order == ["a@1", "b@2", "c@3"]


async def test_virtual_timeout_raises_timeout_error(vclock: VirtualClock) -> None:
    async def work() -> None:
        async with vclock.timeout(1.0):
            await vclock.sleep(10.0)

    with pytest.raises(TimeoutError):
        await vclock.run(work())
    assert vclock.now() == pytest.approx(1.0)


async def test_virtual_timeout_not_triggered_when_body_finishes(vclock: VirtualClock) -> None:
    async def work() -> str:
        async with vclock.timeout(5.0):
            await vclock.sleep(1.0)
        await vclock.sleep(10.0)  # the cancelled timer must not fire later
        return "done"

    assert await vclock.run(work()) == "done"


async def test_timeout_none_means_unbounded(vclock: VirtualClock) -> None:
    async def work() -> float:
        async with vclock.timeout(None):
            await vclock.sleep(100.0)
        return vclock.now()

    assert await vclock.run(work()) == pytest.approx(100.0)


async def test_external_cancellation_is_not_converted_to_timeout(vclock: VirtualClock) -> None:
    async def inner() -> None:
        async with vclock.timeout(10.0):
            await vclock.sleep(5.0)

    async def main() -> None:
        task = asyncio.ensure_future(inner())
        await vclock.sleep(1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    await vclock.run(main())


async def test_deadlock_detection(vclock: VirtualClock) -> None:
    async def stuck() -> None:
        await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match="deadlock"):
        await vclock.run(stuck())


def test_advance_rejects_negative() -> None:
    with pytest.raises(ValueError):
        VirtualClock().advance(-1)


async def test_max_virtual_time_guard(vclock: VirtualClock) -> None:
    async def forever() -> None:
        while True:
            await vclock.sleep(1.0)

    with pytest.raises(RuntimeError, match="max_virtual_time"):
        await vclock.run(forever(), max_virtual_time=5.0)


async def test_system_clock_is_monotonic_and_sleeps() -> None:
    clock = SystemClock()
    t0 = clock.now()
    await clock.sleep(0)
    async with clock.timeout(1.0):
        await clock.sleep(0.001)
    assert clock.now() >= t0
