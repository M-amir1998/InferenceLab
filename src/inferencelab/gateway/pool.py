"""Backend pool: backends + their circuit breakers, runtime stats and health state."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from inferencelab.backends.base import BackendHealth, HealthStatus, InferenceBackend
from inferencelab.backends.config import BackendConfig
from inferencelab.clock import Clock
from inferencelab.gateway.config import HealthCheckConfig
from inferencelab.observability.prometheus import GatewayMetrics
from inferencelab.reliability.circuit_breaker import CircuitBreaker
from inferencelab.routing.base import BackendView
from inferencelab.routing.stats import BackendStats

__all__ = ["BackendHandle", "BackendPool"]

log = logging.getLogger(__name__)


@dataclass(eq=False)
class BackendHandle:
    name: str
    backend: InferenceBackend
    config: BackendConfig
    breaker: CircuitBreaker
    stats: BackendStats
    healthy: bool = True
    checked: bool = False
    last_health: BackendHealth | None = None
    _fail_streak: int = field(default=0, repr=False)
    _ok_streak: int = field(default=0, repr=False)

    @property
    def deployment(self) -> str | None:
        return self.config.deployment

    def serves(self, model: str) -> bool:
        return model in self.config.models

    def view(self) -> BackendView:
        routing = self.config.routing
        return BackendView(
            name=self.name,
            tier=routing.tier,
            healthy=self.healthy,
            circuit_open=self.breaker.is_open(),
            outstanding=self.stats.outstanding,
            max_outstanding=routing.max_outstanding,
            ewma_ttft_s=self.stats.ewma_ttft_s,
            ewma_tpot_s=self.stats.ewma_tpot_s,
            peak_latency_s=self.stats.peak.value,
            cost_per_1m_tokens_usd=routing.cost_per_1m_tokens_usd,
            capabilities=self.backend.capabilities,
        )

    def record_health(self, health: BackendHealth, cfg: HealthCheckConfig) -> None:
        """Hysteresis: N consecutive failures mark unhealthy, M successes mark healthy."""
        self.last_health = health
        self.checked = True
        if health.healthy:
            self._ok_streak += 1
            self._fail_streak = 0
            if not self.healthy and self._ok_streak >= cfg.healthy_threshold:
                self.healthy = True
                log.info("backend %s is healthy again", self.name)
        else:
            self._fail_streak += 1
            self._ok_streak = 0
            if self.healthy and self._fail_streak >= cfg.unhealthy_threshold:
                self.healthy = False
                log.warning("backend %s marked unhealthy: %s", self.name, health.detail)


class BackendPool:
    def __init__(
        self,
        handles: Sequence[BackendHandle],
        *,
        clock: Clock,
        health: HealthCheckConfig,
        metrics: GatewayMetrics | None = None,
    ) -> None:
        self._handles = {h.name: h for h in handles}
        self._clock = clock
        self._health = health
        self._metrics = metrics
        self._task: asyncio.Task[None] | None = None
        for handle in handles:
            self._publish(handle)

    @property
    def handles(self) -> list[BackendHandle]:
        return list(self._handles.values())

    def get(self, name: str) -> BackendHandle | None:
        return self._handles.get(name)

    def models(self) -> list[str]:
        return sorted({m for h in self._handles.values() for m in h.config.models})

    def candidates(self, model: str) -> list[BackendHandle]:
        return [h for h in self._handles.values() if h.serves(model)]

    def any_ready(self) -> bool:
        return any(h.healthy and not h.breaker.is_open() for h in self._handles.values())

    def _publish(self, handle: BackendHandle) -> None:
        if self._metrics is not None:
            self._metrics.backend_healthy.labels(handle.name).set(1 if handle.healthy else 0)
            self._metrics.circuit_state.labels(handle.name).set(handle.breaker.state.gauge_value)
            self._metrics.backend_outstanding.labels(handle.name).set(handle.stats.outstanding)

    async def _check(self, handle: BackendHandle) -> None:
        try:
            async with self._clock.timeout(self._health.timeout_s):
                health = await handle.backend.health()
        except TimeoutError:
            health = BackendHealth(HealthStatus.UNHEALTHY, detail="health check timed out")
        except Exception as exc:  # health checks must never crash the loop
            health = BackendHealth(HealthStatus.UNHEALTHY, detail=f"{type(exc).__name__}: {exc}")
        handle.record_health(health, self._health)
        self._publish(handle)

    async def check_all(self) -> None:
        await asyncio.gather(*(self._check(h) for h in self._handles.values()))

    async def _loop(self) -> None:
        while True:
            await self.check_all()
            await self._clock.sleep(self._health.interval_s)

    def start(self) -> None:
        if self._health.enabled and self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._loop(), name="health-checks")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def aclose(self) -> None:
        await self.stop()
        for handle in self._handles.values():
            with contextlib.suppress(Exception):
                await handle.backend.aclose()
