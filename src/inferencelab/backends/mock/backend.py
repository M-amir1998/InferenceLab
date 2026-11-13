"""Deterministic mock backend with fault injection.

Every random decision is drawn from a PRNG seeded by ``(seed, request_id)``, so the same
request id always gets the same latency jitter and the same injected fault regardless
of concurrency or ordering. Faults can be time-windowed (``faults``) or injected at
runtime (``inject_fault``), which is how demos open and close circuit breakers.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import AsyncIterator

from inferencelab.backends.base import (
    BackendCapabilities,
    BackendChunk,
    BackendHealth,
    BackendMetrics,
    BackendRequest,
    BackendResponse,
    HealthStatus,
    collect_stream,
)
from inferencelab.backends.config import FaultMode, FaultWindow, MockBackendConfig
from inferencelab.clock import Clock, SystemClock
from inferencelab.domain.errors import (
    BackendHTTPError,
    BackendUnavailableError,
    MalformedResponseError,
    StreamInterruptedError,
)
from inferencelab.domain.request import RequestKind
from inferencelab.domain.response import FinishReason, TokenCountSource, TokenUsage
from inferencelab.tokens import WhitespaceTokenCounter
from inferencelab.vocabulary import VOCABULARY

__all__ = ["MockBackend"]

_HANG_S = 24 * 3600.0


class MockBackend:
    def __init__(self, config: MockBackendConfig, *, clock: Clock | None = None) -> None:
        self._config = config
        self._clock = clock or SystemClock()
        self._started_at = self._clock.now()
        self._runtime_faults: list[tuple[float, FaultWindow]] = []
        self._counter = WhitespaceTokenCounter()
        self._inflight = 0
        self._served = 0
        self._failed = 0

    # ------------------------------------------------------------------ protocol
    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> MockBackendConfig:
        return self._config

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(usage_in_stream=True, cancellation=True, engine_metrics=True)

    async def health(self) -> BackendHealth:
        if any(f.mode is FaultMode.DOWN for f in self._active_faults()):
            return BackendHealth(HealthStatus.UNHEALTHY, detail="injected: down")
        return BackendHealth(HealthStatus.HEALTHY, latency_s=0.0)

    async def infer(self, request: BackendRequest) -> BackendResponse:
        return await collect_stream(self.stream(request), request)

    async def metrics(self) -> BackendMetrics:
        return BackendMetrics(
            source="mock",
            running_requests=float(self._inflight),
            waiting_requests=0.0,
            raw={"served_total": float(self._served), "failed_total": float(self._failed)},
        )

    async def aclose(self) -> None:
        return None

    # ------------------------------------------------------------------ faults
    def inject_fault(self, fault: FaultWindow | None) -> None:
        """Add a runtime fault (times relative to *now*), or clear all with ``None``."""
        if fault is None:
            self._runtime_faults.clear()
        else:
            self._runtime_faults.append((self._clock.now(), fault))

    def _active_faults(self) -> list[FaultWindow]:
        now = self._clock.now()
        active = [f for f in self._config.faults if f.active(now - self._started_at)]
        active.extend(f for t0, f in self._runtime_faults if f.active(now - t0))
        return active

    def _rng(self, request_id: str) -> random.Random:
        digest = hashlib.blake2b(f"{self._config.seed}:{request_id}".encode(), digest_size=8)
        return random.Random(int.from_bytes(digest.digest(), "big"))

    def _choose_fault(self, rng: random.Random) -> tuple[FaultMode | None, float]:
        for window in self._active_faults():
            if rng.random() < window.probability:
                return window.mode, window.slow_factor
        cfg = self._config
        draw = rng.random()
        for mode, p in (
            (FaultMode.ERROR, cfg.failure_probability),
            (FaultMode.TIMEOUT, cfg.timeout_probability),
            (FaultMode.DISCONNECT, cfg.disconnect_probability),
            (FaultMode.MALFORMED, cfg.malformed_probability),
        ):
            if draw < p:
                return mode, 1.0
            draw -= p
        return None, 1.0

    def _jittered(self, base_s: float, rng: random.Random) -> float:
        j = self._config.latency.jitter
        return base_s * (1.0 + rng.uniform(-j, j)) if j else base_s

    def _input_tokens(self, request: BackendRequest) -> int:
        if request.kind is RequestKind.CHAT:
            return sum(self._counter.count(m.content) for m in request.messages)
        return self._counter.count(request.prompt or "")

    # ------------------------------------------------------------------ generation
    async def stream(self, request: BackendRequest) -> AsyncIterator[BackendChunk]:
        cfg = self._config
        rng = self._rng(request.request_id)
        fault, slow_factor = self._choose_fault(rng)
        if fault is FaultMode.DOWN:
            raise BackendUnavailableError("mock backend is down", backend=self.name)
        if cfg.max_concurrency is not None and self._inflight >= cfg.max_concurrency:
            raise BackendHTTPError(503, "mock backend at capacity", backend=self.name)

        n_out = request.sampling.max_tokens
        if cfg.output_tokens is not None:
            n_out = min(n_out, cfg.output_tokens)
        slow = slow_factor if fault is FaultMode.SLOW else 1.0
        ttft_s = self._jittered(cfg.latency.ttft_ms / 1000.0, rng) * slow
        break_at = rng.randint(1, max(1, n_out - 1)) if n_out > 1 else 0

        self._inflight += 1
        try:
            if fault is FaultMode.ERROR:
                await self._clock.sleep(min(ttft_s, 0.005))
                self._failed += 1
                raise BackendHTTPError(500, "injected mock failure", backend=self.name)
            if fault is FaultMode.TIMEOUT:
                await self._clock.sleep(_HANG_S)  # until the caller's timeout cancels us
            await self._clock.sleep(ttft_s)
            for i in range(n_out):
                if i > 0:
                    await self._clock.sleep(
                        self._jittered(cfg.latency.tpot_ms / 1000.0, rng) * slow
                    )
                if i == break_at and fault is FaultMode.DISCONNECT:
                    self._failed += 1
                    raise StreamInterruptedError(
                        "injected mid-stream disconnect", backend=self.name
                    )
                if i == break_at and fault is FaultMode.MALFORMED:
                    self._failed += 1
                    raise MalformedResponseError("injected malformed chunk", backend=self.name)
                word = VOCABULARY[rng.randrange(len(VOCABULARY))]
                yield BackendChunk(text=word if i == 0 else " " + word, token_count=1)
            self._served += 1
            finish = (
                FinishReason.LENGTH if n_out == request.sampling.max_tokens else FinishReason.STOP
            )
            yield BackendChunk(
                text="",
                token_count=0,
                finish_reason=finish,
                usage=TokenUsage(self._input_tokens(request), n_out, TokenCountSource.REPORTED),
            )
        finally:
            self._inflight -= 1
