"""Async serving shell around :class:`SimulatedEngine`.

A single background task runs the engine loop: admit arrived requests, execute one
step, sleep until the step's end (on the injected clock) and deliver the emitted tokens
to per-request queues. Engine time is kept *causal and drift-free*:

* a request only enters the engine once engine time has reached its arrival time;
* sleeping targets absolute engine time, so coarse OS timers (≈15.6 ms on Windows)
  delay token delivery slightly but never lower the simulated throughput.

Closing a stream early aborts the sequence and frees its KV blocks, mirroring how
real engines abort requests when the client disconnects.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

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
from inferencelab.backends.config import SimulatedBackendConfig
from inferencelab.backends.simulated.engine import (
    EngineRejectedError,
    SeqTimings,
    SimulatedEngine,
    StepResult,
)
from inferencelab.clock import Clock, SystemClock
from inferencelab.domain.errors import BackendHTTPError, BackendUnavailableError, InferenceError
from inferencelab.domain.request import ChatMessage, RequestKind
from inferencelab.domain.response import EngineTimings, FinishReason, TokenCountSource, TokenUsage
from inferencelab.vocabulary import VOCABULARY, word_to_id

__all__ = ["SimulatedInferenceBackend", "render_prompt_tokens"]


def render_prompt_tokens(request: BackendRequest) -> list[int]:
    """The simulator's 'tokenizer' + chat template: one token per word or marker."""
    if request.kind is RequestKind.COMPLETION:
        return [word_to_id(w) for w in (request.prompt or "").split()]
    return _render_chat(request.messages)


def _render_chat(messages: tuple[ChatMessage, ...]) -> list[int]:
    ids: list[int] = []
    for message in messages:
        ids.append(word_to_id(f"<|{message.role}|>"))
        ids.extend(word_to_id(w) for w in message.content.split())
        ids.append(word_to_id("<|end|>"))
    ids.append(word_to_id("<|assistant|>"))  # generation prompt
    return ids


@dataclass(frozen=True, slots=True)
class _Token:
    word: str
    first: bool


@dataclass(frozen=True, slots=True)
class _Done:
    timings: SeqTimings


@dataclass(frozen=True, slots=True)
class _Failed:
    error: InferenceError


_Event = _Token | _Done | _Failed


class SimulatedInferenceBackend:
    def __init__(self, config: SimulatedBackendConfig, *, clock: Clock | None = None) -> None:
        self._config = config
        self._clock = clock or SystemClock()
        self.engine = SimulatedEngine(config.engine)
        self._queues: dict[str, asyncio.Queue[_Event]] = {}
        self._pending: deque[tuple[float, str, list[int], int]] = deque()
        self._wakeup: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._engine_time = self._clock.now()
        self._ids = itertools.count()
        self._closed = False

    # ------------------------------------------------------------------ protocol
    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> SimulatedBackendConfig:
        return self._config

    @property
    def capabilities(self) -> BackendCapabilities:
        eng = self._config.engine
        return BackendCapabilities(
            usage_in_stream=True,
            prefix_caching=eng.prefix_caching,
            chunked_prefill=eng.chunked_prefill,
            cancellation=True,
            engine_metrics=True,
        )

    async def health(self) -> BackendHealth:
        if self._closed:
            return BackendHealth(HealthStatus.UNHEALTHY, detail="closed")
        return BackendHealth(HealthStatus.HEALTHY, latency_s=0.0)

    async def infer(self, request: BackendRequest) -> BackendResponse:
        return await collect_stream(self.stream(request), request)

    async def metrics(self) -> BackendMetrics:
        s = self.engine.stats
        return BackendMetrics(
            source="simulator",
            running_requests=float(self.engine.num_running),
            waiting_requests=float(self.engine.num_waiting + len(self._pending)),
            kv_cache_usage=self.engine.blocks.usage,
            prefix_cache_queries=float(s.prefix_cache_queries),
            prefix_cache_hits=float(s.prefix_cache_hits),
            preemptions_total=float(s.preemptions),
            prompt_tokens_total=float(s.prompt_tokens_computed),
            generation_tokens_total=float(s.generation_tokens),
            raw={
                "steps_total": float(s.steps),
                "busy_time_seconds_total": s.busy_time_s,
                "finished_total": float(s.finished),
                "aborted_total": float(s.aborted),
            },
        )

    async def aclose(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._fail_all(BackendUnavailableError("simulated backend closed", backend=self.name))

    # ------------------------------------------------------------------ generation
    async def stream(self, request: BackendRequest) -> AsyncIterator[BackendChunk]:
        if self._closed:
            raise BackendUnavailableError("simulated backend closed", backend=self.name)
        ids = render_prompt_tokens(request)
        max_tokens = request.sampling.max_tokens
        try:
            self.engine.validate(len(ids), max_tokens)
        except EngineRejectedError as exc:
            raise BackendHTTPError(400, str(exc), backend=self.name) from exc
        seq_id = f"{request.request_id}#{next(self._ids)}"
        queue: asyncio.Queue[_Event] = asyncio.Queue()
        self._queues[seq_id] = queue
        self._pending.append((self._clock.now(), seq_id, ids, max_tokens))
        self._ensure_loop()
        done = False
        try:
            while True:
                event = await queue.get()
                if isinstance(event, _Token):
                    yield BackendChunk(
                        text=event.word if event.first else " " + event.word, token_count=1
                    )
                elif isinstance(event, _Done):
                    done = True
                    t = event.timings
                    yield BackendChunk(
                        text="",
                        token_count=0,
                        finish_reason=FinishReason.LENGTH,
                        usage=TokenUsage(
                            t.prompt_tokens,
                            t.output_tokens,
                            TokenCountSource.REPORTED,
                            cached_input_tokens=t.cached_tokens,
                        ),
                        engine=EngineTimings(
                            queue_s=t.queue_s, prefill_s=t.prefill_s, decode_s=t.decode_s
                        ),
                    )
                    return
                else:
                    done = True
                    raise event.error
        finally:
            self._queues.pop(seq_id, None)
            if not done:
                self._cancel(seq_id)

    # ------------------------------------------------------------------ engine loop
    def _ensure_loop(self) -> None:
        if self._wakeup is None:
            self._wakeup = asyncio.Event()
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self._run(), name=f"simulated-engine:{self.name}"
            )
            self._task.add_done_callback(self._on_loop_exit)
        self._wakeup.set()

    def _on_loop_exit(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:  # pragma: no cover - defensive: never leave streams hanging
            self._fail_all(InferenceError(f"simulated engine crashed: {exc!r}", backend=self.name))

    def _fail_all(self, error: InferenceError) -> None:
        for queue in self._queues.values():
            queue.put_nowait(_Failed(error))

    def _cancel(self, seq_id: str) -> None:
        for i, item in enumerate(self._pending):
            if item[1] == seq_id:
                del self._pending[i]
                return
        self.engine.abort(seq_id)

    async def _run(self) -> None:
        wakeup = self._wakeup
        if wakeup is None:  # pragma: no cover - created by _ensure_loop
            raise RuntimeError("engine loop started without a wakeup event")
        engine = self.engine
        while not self._closed:
            if not engine.has_work() and not self._pending:
                wakeup.clear()
                await wakeup.wait()
                continue
            if not engine.has_work():
                # Idle engine: jump engine time to the earliest arrival (never backwards).
                self._engine_time = max(self._engine_time, self._pending[0][0])
            while self._pending and self._pending[0][0] <= self._engine_time:
                arrival, seq_id, ids, max_tokens = self._pending.popleft()
                engine.submit(seq_id, ids, max_tokens, arrival_time=arrival)
            result = engine.step(self._engine_time)
            if result is None:
                continue
            self._engine_time += result.duration
            await self._clock.sleep(self._engine_time - self._clock.now())
            self._dispatch(result)

    def _dispatch(self, result: StepResult) -> None:
        for emission in result.emitted:
            queue = self._queues.get(emission.seq_id)
            if queue is not None:
                word = VOCABULARY[emission.token_id % len(VOCABULARY)]
                queue.put_nowait(_Token(word, emission.first))
        for seq_id, timings in result.finished:
            queue = self._queues.get(seq_id)
            if queue is not None:
                queue.put_nowait(_Done(timings))
