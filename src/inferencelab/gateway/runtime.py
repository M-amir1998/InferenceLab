"""Wire a :class:`GatewayConfig` into live components (pool, router, admission, service)."""

from __future__ import annotations

from collections.abc import Mapping

from inferencelab.admission.controller import AdmissionController
from inferencelab.backends.base import InferenceBackend
from inferencelab.backends.factory import create_backend
from inferencelab.clock import Clock, SystemClock
from inferencelab.gateway.config import GatewayConfig
from inferencelab.gateway.pool import BackendHandle, BackendPool
from inferencelab.gateway.service import InferenceService
from inferencelab.limits.tenant import TenantLimiter
from inferencelab.observability.prometheus import GatewayMetrics
from inferencelab.reliability.circuit_breaker import CircuitBreaker, CircuitState
from inferencelab.reliability.retry import RetryBudget, RetryPolicy
from inferencelab.routing.factory import build_router
from inferencelab.routing.stats import BackendStats

__all__ = ["GatewayRuntime"]


class GatewayRuntime:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        clock: Clock | None = None,
        backends: Mapping[str, InferenceBackend] | None = None,
        metrics: GatewayMetrics | None = None,
    ) -> None:
        """Build components. ``backends`` overrides config-created instances (tests, demos)."""
        self.config = config
        self.clock = clock or SystemClock()
        self.metrics = metrics or GatewayMetrics()
        overrides = dict(backends or {})
        handles: list[BackendHandle] = []
        for bcfg in config.backends:
            backend = overrides.pop(bcfg.name, None) or create_backend(bcfg, clock=self.clock)
            handles.append(
                BackendHandle(
                    name=bcfg.name,
                    backend=backend,
                    config=bcfg,
                    breaker=CircuitBreaker(
                        config.circuit_breaker,
                        self.clock,
                        name=bcfg.name,
                        on_transition=self._transition_hook(bcfg.name),
                    ),
                    stats=BackendStats(
                        self.clock,
                        alpha=config.routing.ewma_alpha,
                        peak_tau_s=config.routing.peak_ewma_decay_s,
                    ),
                )
            )
        if overrides:
            raise ValueError(f"backend overrides without config entries: {sorted(overrides)}")
        self.pool = BackendPool(
            handles, clock=self.clock, health=config.health, metrics=self.metrics
        )
        self.router = build_router(config.routing)
        self.admission = AdmissionController(config.queue, self.clock, on_change=self._queue_hook)
        limiter = TenantLimiter(config.tenants, self.clock) if config.tenants is not None else None
        thresholds = config.slo.effective_thresholds() if config.slo is not None else None
        self.service = InferenceService(
            pool=self.pool,
            router=self.router,
            admission=self.admission,
            retry=RetryPolicy(config.retry, seed=config.routing.seed),
            retry_budget=RetryBudget(
                config.retry.budget_ratio, config.retry.budget_min_retries_per_s, self.clock
            ),
            timeouts=config.timeouts,
            metrics=self.metrics,
            clock=self.clock,
            limiter=limiter,
            slo_thresholds=thresholds,
        )

    def _transition_hook(self, name: str):  # type: ignore[no-untyped-def]
        def hook(old: CircuitState, new: CircuitState) -> None:
            self.metrics.circuit_state.labels(name).set(new.gauge_value)
            self.metrics.circuit_transitions.labels(name, new.value).inc()

        return hook

    def _queue_hook(self, in_flight: int, pending: int) -> None:
        self.metrics.in_flight.set(in_flight)
        self.metrics.queue_depth.set(pending)

    async def start(self) -> None:
        if self.config.health.enabled:
            await self.pool.check_all()  # know backend state before serving
        self.pool.start()

    async def stop(self) -> None:
        await self.pool.aclose()
