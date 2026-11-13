"""Admission control: bounded concurrency, a bounded priority queue and load shedding.

At most ``max_concurrency`` requests execute against backends; up to
``max_pending_requests`` more wait in a priority queue (higher priority first, FIFO
within a priority). Nothing is ever queued without bound.

Shedding policies when the queue is full (or a wait would be pointless):

* ``reject_newest``           reject the arriving request (HTTP 503, ``queue_full``)
* ``reject_lowest_priority``  evict the lowest-priority, newest waiter if the arrival
                              outranks it; otherwise reject the arrival
* ``slo_aware``               additionally reject on arrival when the *predicted* queue
                              wait exceeds ``slo_queue_budget_s`` — failing fast beats
                              answering after the client's SLO has already been missed

The predicted wait uses Little's law with an EWMA of observed service time; with no
observations yet, nothing is shed on prediction (no data, no claims).
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import Field

from inferencelab.clock import Clock
from inferencelab.domain.errors import LoadShedError, QueueFullError
from inferencelab.domain.request import Priority
from inferencelab.schema import StrictModel

__all__ = ["AdmissionController", "AdmissionPermit", "QueueConfig", "SheddingPolicy"]


class SheddingPolicy(StrEnum):
    REJECT_NEWEST = "reject_newest"
    REJECT_LOWEST_PRIORITY = "reject_lowest_priority"
    SLO_AWARE = "slo_aware"


class QueueConfig(StrictModel):
    max_concurrency: int = Field(default=256, ge=1)
    max_pending_requests: int = Field(default=1000, ge=0)
    shedding: SheddingPolicy = SheddingPolicy.REJECT_NEWEST
    max_queue_wait_s: float | None = Field(default=None, gt=0)
    slo_queue_budget_s: float | None = Field(default=None, gt=0)
    service_time_ewma_alpha: float = Field(default=0.2, gt=0, le=1)


@dataclass(eq=False, slots=True)
class _Waiter:
    priority: Priority
    seq: int
    enqueued_at: float
    future: asyncio.Future[None] = field(repr=False)
    removed: bool = False


class AdmissionPermit:
    __slots__ = ("_controller", "_released", "granted_at", "queue_s")

    def __init__(self, controller: AdmissionController, granted_at: float, queue_s: float) -> None:
        self._controller = controller
        self.granted_at = granted_at
        self.queue_s = queue_s
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._controller._release(self._controller.clock.now() - self.granted_at)


class AdmissionController:
    def __init__(
        self,
        config: QueueConfig,
        clock: Clock,
        *,
        on_change: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config = config
        self.clock = clock
        self._on_change = on_change
        self._in_flight = 0
        self._pending = 0
        self._heap: list[tuple[int, int, _Waiter]] = []
        self._seq = itertools.count()
        self._service_s: float | None = None

    # ------------------------------------------------------------------ observers
    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def pending(self) -> int:
        return self._pending

    def predicted_wait_s(self, position: int) -> float | None:
        """Little's law: wait ≈ position × service_time / concurrency."""
        if self._service_s is None:
            return None
        return position * self._service_s / self.config.max_concurrency

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change(self._in_flight, self._pending)

    # ------------------------------------------------------------------ admission
    async def acquire(self, priority: Priority = Priority.NORMAL) -> AdmissionPermit:
        cfg = self.config
        now = self.clock.now()
        if self._in_flight < cfg.max_concurrency and self._pending == 0:
            self._in_flight += 1
            self._notify()
            return AdmissionPermit(self, now, 0.0)
        if self._pending >= cfg.max_pending_requests and not (
            cfg.shedding is SheddingPolicy.REJECT_LOWEST_PRIORITY and self._evict_below(priority)
        ):
            raise QueueFullError(
                f"admission queue full ({cfg.max_pending_requests} pending)", retry_after_s=1.0
            )
        if cfg.shedding is SheddingPolicy.SLO_AWARE and cfg.slo_queue_budget_s is not None:
            predicted = self.predicted_wait_s(self._pending + 1)
            if predicted is not None and predicted > cfg.slo_queue_budget_s:
                raise LoadShedError(
                    f"predicted queue wait {predicted:.2f}s exceeds SLO budget "
                    f"{cfg.slo_queue_budget_s:.2f}s",
                    retry_after_s=predicted,
                )
        waiter = _Waiter(priority, next(self._seq), now, asyncio.get_running_loop().create_future())
        heapq.heappush(self._heap, (-int(priority), waiter.seq, waiter))
        self._pending += 1
        self._notify()
        try:
            async with self.clock.timeout(cfg.max_queue_wait_s):
                await waiter.future
        except TimeoutError:
            if not self._granted(waiter):
                self._remove(waiter)
                raise LoadShedError(
                    f"queued longer than {cfg.max_queue_wait_s}s", retry_after_s=1.0
                ) from None
        except asyncio.CancelledError:
            if self._granted(waiter):
                self._release(None)  # we hold a slot we will never use
            else:
                self._remove(waiter)
            raise
        granted = self.clock.now()
        return AdmissionPermit(self, granted, granted - waiter.enqueued_at)

    @staticmethod
    def _granted(waiter: _Waiter) -> bool:
        fut = waiter.future
        return fut.done() and not fut.cancelled() and fut.exception() is None

    def _remove(self, waiter: _Waiter) -> None:
        if not waiter.removed:
            waiter.removed = True
            self._pending -= 1
            self._notify()

    def _evict_below(self, priority: Priority) -> bool:
        """Evict the lowest-priority (newest) waiter below ``priority``; True on success."""
        victim: _Waiter | None = None
        for _, _, w in self._heap:
            if w.removed or w.future.done() or w.priority >= priority:
                continue
            if victim is None or (w.priority, -w.seq) < (victim.priority, -victim.seq):
                victim = w
        if victim is None:
            return False
        self._remove(victim)
        victim.future.set_exception(LoadShedError("evicted by a higher-priority request"))
        return True

    def _release(self, hold_s: float | None) -> None:
        self._in_flight -= 1
        if hold_s is not None:
            a = self.config.service_time_ewma_alpha
            self._service_s = (
                hold_s if self._service_s is None else (1 - a) * self._service_s + a * hold_s
            )
        while self._heap and self._in_flight < self.config.max_concurrency:
            _, _, waiter = heapq.heappop(self._heap)
            if waiter.removed or waiter.future.done():
                continue
            waiter.removed = True
            self._pending -= 1
            self._in_flight += 1
            waiter.future.set_result(None)
        self._notify()
