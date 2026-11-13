"""Time abstraction.

Every component that measures or waits on time receives a :class:`Clock`.

* :class:`SystemClock` is used in production: ``time.perf_counter`` (monotonic,
  high-resolution) plus ``asyncio`` sleeping and timeouts.
* :class:`VirtualClock` is a discrete-event clock for tests and offline simulation.
  Virtual time only moves when every task is blocked, at which point it jumps to the
  next pending timer. Timeouts, rate limiters, circuit breakers, arrival processes and
  the simulated engine therefore behave deterministically, and tests never sleep.

Code that runs under a :class:`VirtualClock` must wait *only* through the clock
(``clock.sleep`` / ``clock.timeout``), never through ``asyncio.sleep`` with a non-zero
delay or ``asyncio.wait_for``.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, Protocol, TypeVar, runtime_checkable

__all__ = ["Clock", "SystemClock", "VirtualClock"]

T = TypeVar("T")


@runtime_checkable
class Clock(Protocol):
    """Monotonic time source with cooperative sleeping and timeouts."""

    def now(self) -> float:
        """Monotonic time in seconds. Only differences are meaningful."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the current task for ``seconds`` (non-positive values just yield)."""
        ...

    def timeout(self, seconds: float | None) -> AbstractAsyncContextManager[object]:
        """Context manager raising :class:`TimeoutError` if the body exceeds ``seconds``."""
        ...


class SystemClock:
    """Real time: ``perf_counter`` for measurement, asyncio for waiting."""

    __slots__ = ()

    def now(self) -> float:
        return time.perf_counter()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))

    def timeout(self, seconds: float | None) -> AbstractAsyncContextManager[object]:
        # asyncio.timeout uses the loop clock; durations are relative so the
        # different time bases of loop.time() and perf_counter() do not matter.
        return asyncio.timeout(None if seconds is None else max(0.0, seconds))


class _Timer:
    __slots__ = ("callback", "cancelled")

    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        self.cancelled = False


class VirtualClock:
    """Deterministic discrete-event clock.

    ``advance(dt)`` moves time forward and fires due timers in deadline order.
    ``run(coro)`` drives a coroutine to completion, jumping time to the next timer
    whenever the event loop has no runnable callbacks (all tasks are blocked).
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._timers: list[tuple[float, int, _Timer]] = []
        self._seq = itertools.count()

    def now(self) -> float:
        return self._now

    def _schedule(self, deadline: float, callback: Callable[[], None]) -> _Timer:
        timer = _Timer(callback)
        heapq.heappush(self._timers, (deadline, next(self._seq), timer))
        return timer

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def _wake() -> None:
            if not fut.done():
                fut.set_result(None)

        timer = self._schedule(self._now + seconds, _wake)
        try:
            await fut
        finally:
            timer.cancelled = True

    def timeout(self, seconds: float | None) -> AbstractAsyncContextManager[object]:
        return _VirtualTimeout(self, seconds)

    def next_deadline(self) -> float | None:
        while self._timers and self._timers[0][2].cancelled:
            heapq.heappop(self._timers)
        return self._timers[0][0] if self._timers else None

    def advance(self, seconds: float) -> None:
        """Move virtual time forward by ``seconds``, firing timers that become due."""
        if seconds < 0:
            raise ValueError("cannot move virtual time backwards")
        target = self._now + seconds
        while self._timers and self._timers[0][0] <= target:
            deadline, _, timer = heapq.heappop(self._timers)
            if timer.cancelled:
                continue
            self._now = max(self._now, deadline)
            timer.callback()
        self._now = target

    async def run(self, awaitable: Awaitable[T], *, max_virtual_time: float | None = None) -> T:
        """Run ``awaitable`` to completion under virtual time.

        Raises ``RuntimeError`` on deadlock (task blocked with no pending timers) or
        when ``max_virtual_time`` would be exceeded.
        """
        task: asyncio.Future[T] = asyncio.ensure_future(awaitable)
        limit = None if max_virtual_time is None else self._now + max_virtual_time
        try:
            while not task.done():
                await _settle(task)
                if task.done():
                    break
                deadline = self.next_deadline()
                if deadline is None:
                    raise RuntimeError(
                        "VirtualClock deadlock: work is blocked and no timers remain"
                    )
                if limit is not None and deadline > limit:
                    raise RuntimeError(f"VirtualClock exceeded max_virtual_time={max_virtual_time}")
                self.advance(max(0.0, deadline - self._now))
            return task.result()
        finally:
            if not task.done():
                task.cancel()


async def _settle(task: asyncio.Future[Any], max_spins: int = 100_000) -> None:
    """Yield until no callbacks other than ours are runnable (or ``task`` finishes)."""
    loop = asyncio.get_running_loop()
    ready = getattr(loop, "_ready", None)  # CPython BaseEventLoop run queue
    quiet = 0
    for _ in range(max_spins):
        await asyncio.sleep(0)
        if task.done():
            return
        if ready is None:
            quiet += 1
            if quiet >= 64:  # conservative fallback for exotic loops
                return
            continue
        # Our own wake-up was just consumed; an empty queue means everything else
        # is blocked on futures that only timers (or nothing) can resolve.
        if len(ready) == 0:
            quiet += 1
            if quiet >= 2:
                return
        else:
            quiet = 0


class _VirtualTimeout:
    """``asyncio.timeout`` equivalent driven by a :class:`VirtualClock`."""

    def __init__(self, clock: VirtualClock, seconds: float | None) -> None:
        self._clock = clock
        self._seconds = seconds
        self._task: asyncio.Task[object] | None = None
        self._timer: _Timer | None = None
        self._expired = False
        self._cancelling = 0

    async def __aenter__(self) -> None:
        if self._seconds is None:
            return
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - always inside a task
            raise RuntimeError("timeout() must be used inside a task")
        self._task = task
        self._cancelling = task.cancelling()
        self._timer = self._clock._schedule(
            self._clock.now() + max(0.0, self._seconds), self._expire
        )

    def _expire(self) -> None:
        if self._task is not None and not self._task.done():
            self._expired = True
            self._task.cancel()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._timer is not None:
            self._timer.cancelled = True
        if self._expired and self._task is not None:
            remaining = self._task.uncancel()
            if remaining <= self._cancelling and exc_type is asyncio.CancelledError:
                raise TimeoutError from exc
