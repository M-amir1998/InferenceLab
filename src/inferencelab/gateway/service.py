"""Data-plane orchestration (framework-independent).

Request lifecycle::

    tenant quota ─► admission (bounded priority queue) ─► tiered routing
        ─► attempt loop:  dispatch ─► first token (first-token timeout)
              │ failure before output ─► breaker/stats ─► backoff ─► next candidate
              ▼
        stream remaining chunks (idle timeout, overall deadline)
              ▼
        finalize: circuit verdict, stats, metrics, quota refund, slot release

Retries happen only while no output has reached the client (streaming) or while the
response is still buffered (non-streaming). Every exit path — success, failure,
timeout, client disconnect — releases the admission slot, the tenant lease and the
backend connection exactly once.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field

from opentelemetry import trace

from inferencelab.admission.controller import AdmissionController, AdmissionPermit
from inferencelab.backends.base import BackendChunk, BackendRequest
from inferencelab.clock import Clock
from inferencelab.domain.errors import (
    CircuitOpenError,
    ErrorKind,
    InferenceError,
    MalformedResponseError,
    NoHealthyBackendError,
    TimeoutPhase,
)
from inferencelab.domain.request import InferenceRequest
from inferencelab.domain.response import (
    EngineTimings,
    FinishReason,
    GatewayTimings,
    InferenceResponse,
    TokenCountSource,
    TokenUsage,
)
from inferencelab.domain.slo import RequestThresholds
from inferencelab.gateway.pool import BackendHandle, BackendPool
from inferencelab.limits.tenant import TenantLease, TenantLimiter
from inferencelab.observability.prometheus import GatewayMetrics
from inferencelab.observability.tracing import get_tracer
from inferencelab.reliability.circuit_breaker import CircuitPermit
from inferencelab.reliability.retry import RetryBudget, RetryPolicy
from inferencelab.reliability.timeouts import TimeoutConfig, next_with_timeout
from inferencelab.routing.base import Router, RoutingContext, RoutingDecision
from inferencelab.routing.factory import rank_tiered
from inferencelab.tokens import HeuristicTokenCounter

__all__ = ["GatewayStream", "InferenceService"]

log = logging.getLogger(__name__)
_REJECTION_KINDS = frozenset(
    {
        ErrorKind.RATE_LIMITED,
        ErrorKind.QUEUE_FULL,
        ErrorKind.LOAD_SHED,
        ErrorKind.NO_HEALTHY_BACKEND,
        ErrorKind.CIRCUIT_OPEN,
        ErrorKind.UNAUTHORIZED,
        ErrorKind.INVALID_REQUEST,
    }
)


def _status(err: InferenceError) -> str:
    if err.kind.value.startswith("timeout"):
        return "timeout"
    if err.kind is ErrorKind.CANCELLED:
        return "cancelled"
    if err.kind in _REJECTION_KINDS:
        return "rejected"
    return "error"


@dataclass(slots=True)
class _Ctx:
    request: InferenceRequest
    received_at: float
    deadline: float | None
    span: trace.Span
    queue_s: float = 0.0
    lease: TenantLease | None = None
    permit: AdmissionPermit | None = None
    released: bool = False


@dataclass(slots=True)
class _Acc:
    """Output accounting for one attempt (no text is retained)."""

    chunks: int = 0
    chars: int = 0
    counted: int = 0
    counts_known: bool = True
    usage: TokenUsage | None = None
    finish: FinishReason | None = None
    engine: EngineTimings | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None

    def observe(self, chunk: BackendChunk, now: float) -> None:
        self.chunks += 1
        if chunk.text:
            self.chars += len(chunk.text)
            if self.first_token_at is None:
                self.first_token_at = now
            self.last_token_at = now
            if chunk.token_count is None:
                self.counts_known = False
        if chunk.token_count is not None:
            self.counted += chunk.token_count
        self.usage = chunk.usage or self.usage
        self.finish = chunk.finish_reason or self.finish
        self.engine = chunk.engine or self.engine


@dataclass(slots=True)
class _Attempt:
    handle: BackendHandle
    permit: CircuitPermit
    stream: AsyncIterator[BackendChunk]
    attempts: int
    decision: RoutingDecision
    dispatched_at: float
    acc: _Acc = field(default_factory=_Acc)
    first: BackendChunk | None = None
    closed: bool = False


class InferenceService:
    def __init__(
        self,
        *,
        pool: BackendPool,
        router: Router,
        admission: AdmissionController,
        retry: RetryPolicy,
        retry_budget: RetryBudget,
        timeouts: TimeoutConfig,
        metrics: GatewayMetrics,
        clock: Clock,
        limiter: TenantLimiter | None = None,
        slo_thresholds: RequestThresholds | None = None,
    ) -> None:
        self.pool = pool
        self.router = router
        self.admission = admission
        self.retry = retry
        self.retry_budget = retry_budget
        self.timeouts = timeouts
        self.metrics = metrics
        self.clock = clock
        self.limiter = limiter
        self.slo_thresholds = slo_thresholds
        self._counter = HeuristicTokenCounter()
        self._tracer = get_tracer()

    # ================================================================== public API
    async def open_stream(self, request: InferenceRequest) -> GatewayStream:
        """Admit, route and wait for the first token. Raises before any output on failure."""
        ctx = await self._admit(request)
        try:
            attempt = await self._start(ctx, attempts_made=0, excluded=frozenset())
        except BaseException as exc:
            self._finish_request(ctx, None, exc)
            raise
        return GatewayStream(self, ctx, attempt)

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        """Non-streaming: the response is buffered, so mid-generation retries are safe."""
        ctx = await self._admit(request)
        attempts = 0
        excluded: set[str] = set()
        attempt: _Attempt | None = None
        try:
            while True:
                attempt = await self._start(
                    ctx, attempts_made=attempts, excluded=frozenset(excluded)
                )
                attempts = attempt.attempts
                try:
                    drain = self._drain(ctx, attempt)
                    async with contextlib.aclosing(drain):
                        parts = [chunk.text async for chunk in drain if chunk.text]
                except InferenceError as err:
                    if not self._may_retry_after_output(ctx, err, attempts):
                        raise
                    self.metrics.retries.labels(attempt.handle.name, err.kind.value).inc()
                    if len(self.pool.candidates(request.model)) > len(excluded) + 1:
                        excluded.add(attempt.handle.name)
                    continue
                response = self._response(ctx, attempt, "".join(parts))
                self._finish_request(ctx, attempt, None)
                return response
        except BaseException as exc:
            self._finish_request(ctx, attempt, exc)
            raise

    # ================================================================== admission
    async def _admit(self, request: InferenceRequest) -> _Ctx:
        now = self.clock.now()
        overall = self.timeouts.overall_s
        span = self._tracer.start_span(
            "gateway.request",
            attributes={
                "inferencelab.request_id": request.request_id,
                "inferencelab.model": request.model,
                "inferencelab.stream": request.stream,
                "inferencelab.priority": request.priority.name.lower(),
            },
        )
        ctx = _Ctx(request, now, None if overall is None else now + overall, span)
        estimated = self._counter.count(request.text_for_estimation()) + request.sampling.max_tokens
        try:
            with self._tracer.start_as_current_span(
                "gateway.admission", context=trace.set_span_in_context(span)
            ):
                if self.limiter is not None:
                    ctx.lease = self.limiter.acquire(request.hints.tenant_id, estimated)
                try:
                    ctx.permit = await self.admission.acquire(request.priority)
                except BaseException:
                    if ctx.lease is not None:
                        ctx.lease.release(None)
                        ctx.lease = None
                    raise
        except InferenceError as err:
            self.metrics.rejections.labels(err.kind.value).inc()
            self._finish_request(ctx, None, err)
            raise
        except BaseException as exc:
            self._finish_request(ctx, None, exc)
            raise
        ctx.queue_s = ctx.permit.queue_s
        self.metrics.queue_wait.observe(ctx.queue_s)
        return ctx

    # ================================================================== routing
    def _route(self, ctx: _Ctx, excluded: frozenset[str]) -> RoutingDecision:
        request = ctx.request
        handles = [h for h in self.pool.candidates(request.model) if h.name not in excluded]
        if not handles:
            raise NoHealthyBackendError(f"no backend serves model {request.model!r}")
        budget = None
        if self.slo_thresholds is not None and self.slo_thresholds.ttft_ms is not None:
            budget = self.slo_thresholds.ttft_ms / 1000.0 - (self.clock.now() - ctx.received_at)
        rctx = RoutingContext(
            request_id=request.request_id,
            model=request.model,
            priority=request.priority,
            hints=request.hints,
            estimated_input_tokens=self._counter.count(request.text_for_estimation()),
            max_tokens=request.sampling.max_tokens,
            ttft_budget_s=budget,
        )
        with self._tracer.start_as_current_span(
            "gateway.route", context=trace.set_span_in_context(ctx.span)
        ) as span:
            decision = rank_tiered(self.router, rctx, [h.view() for h in handles])
            span.set_attribute("inferencelab.policy", decision.policy)
            span.set_attribute("inferencelab.ranking", ",".join(decision.ranking))
        if not decision.ranking:
            raise NoHealthyBackendError(
                f"no healthy backend for model {request.model!r}", retry_after_s=1.0
            )
        if decision.slo_at_risk:
            self.metrics.slo_at_risk.inc()
        return decision

    def _pick(
        self, ranking: tuple[str, ...], tried: set[str]
    ) -> tuple[BackendHandle, CircuitPermit] | None:
        for name in ranking:
            if name in tried:
                continue
            handle = self.pool.get(name)
            if handle is None:
                continue
            permit = handle.breaker.acquire()
            if permit is not None:
                return handle, permit
            tried.add(name)
            self.metrics.errors.labels(name, ErrorKind.CIRCUIT_OPEN.value).inc()
        return None

    # ================================================================== attempts
    async def _start(self, ctx: _Ctx, *, attempts_made: int, excluded: frozenset[str]) -> _Attempt:
        request = ctx.request
        decision = self._route(ctx, excluded)
        tried: set[str] = set()
        last_error: InferenceError | None = None
        failures = 0
        while attempts_made < self.retry.config.max_attempts:
            picked = self._pick(decision.ranking, tried)
            if picked is None and tried:
                tried = set()  # every candidate tried once: allow another round
                picked = self._pick(decision.ranking, tried)
            if picked is None:
                if last_error is not None:
                    raise last_error
                retry_after = min(
                    (
                        h.breaker.retry_after_s
                        for h in self.pool.handles
                        if h.name in decision.ranking
                    ),
                    default=1.0,
                )
                raise CircuitOpenError("all candidate circuits are open", retry_after_s=retry_after)
            handle, permit = picked
            if attempts_made > 0:
                if not self.retry_budget.try_spend():
                    permit.ignore()
                    self.metrics.rejections.labels("retry_budget").inc()
                    if last_error is None:  # attempts_made > 0 implies a recorded failure
                        raise RuntimeError("retry attempted without a recorded failure")
                    raise last_error
                delay = self.retry.backoff_s(failures)
                if ctx.deadline is not None and self.clock.now() + delay >= ctx.deadline:
                    permit.ignore()
                    raise last_error or NoHealthyBackendError("deadline exhausted")
                await self.clock.sleep(delay)
                self.metrics.retries.labels(
                    handle.name, last_error.kind.value if last_error else "unknown"
                ).inc()
            else:
                self.retry_budget.record_request()
                self.metrics.router_decisions.labels(decision.policy, handle.name).inc()
            attempts_made += 1
            tried.add(handle.name)
            attempt = self._dispatch(ctx, handle, permit, attempts_made, decision)
            try:
                await self._await_first(ctx, attempt)
            except InferenceError as err:
                await self._close_attempt(attempt, err)
                last_error = err
                failures += 1
                if not self.retry.should_retry(
                    err, attempts_made=attempts_made, output_started=False, streaming=request.stream
                ):
                    raise
                continue
            except BaseException:
                await self._close_attempt(attempt, None)
                raise
            return attempt
        raise last_error or NoHealthyBackendError("no attempts possible")

    def _dispatch(
        self,
        ctx: _Ctx,
        handle: BackendHandle,
        permit: CircuitPermit,
        attempts: int,
        decision: RoutingDecision,
    ) -> _Attempt:
        breq = BackendRequest.from_inference(
            ctx.request, handle.config.served_model(ctx.request.model)
        )
        handle.stats.on_dispatch()
        self.metrics.backend_outstanding.labels(handle.name).set(handle.stats.outstanding)
        return _Attempt(
            handle=handle,
            permit=permit,
            stream=handle.backend.stream(breq),
            attempts=attempts,
            decision=decision,
            dispatched_at=self.clock.now(),
        )

    async def _await_first(self, ctx: _Ctx, attempt: _Attempt) -> None:
        with self._tracer.start_as_current_span(
            "backend.attempt",
            context=trace.set_span_in_context(ctx.span),
            attributes={
                "inferencelab.backend": attempt.handle.name,
                "inferencelab.attempt": attempt.attempts,
            },
        ):
            try:
                first = await next_with_timeout(
                    attempt.stream,
                    clock=self.clock,
                    timeout_s=self.timeouts.first_token_s,
                    phase=TimeoutPhase.FIRST_TOKEN,
                    deadline=ctx.deadline,
                    backend=attempt.handle.name,
                )
            except StopAsyncIteration:
                raise MalformedResponseError(
                    "backend ended the stream without output", backend=attempt.handle.name
                ) from None
            except InferenceError:
                raise
            except Exception as exc:  # adapter bug: classify instead of crashing the gateway
                log.exception("backend %s raised an unclassified error", attempt.handle.name)
                raise InferenceError(
                    f"unexpected backend error: {type(exc).__name__}",
                    kind=ErrorKind.BACKEND_ERROR,
                    backend=attempt.handle.name,
                ) from exc
        now = self.clock.now()
        attempt.first = first
        attempt.acc.observe(first, now)
        if first.text:
            attempt.handle.stats.on_first_token(now - attempt.dispatched_at)

    async def _drain(self, ctx: _Ctx, attempt: _Attempt) -> AsyncGenerator[BackendChunk, None]:
        """Yield the first chunk, then the rest with idle/deadline timeouts."""
        if attempt.first is None:  # pragma: no cover - _start always sets it
            raise RuntimeError("attempt has no first chunk")
        try:
            yield attempt.first
            if not attempt.first.is_final:
                while True:
                    try:
                        chunk = await next_with_timeout(
                            attempt.stream,
                            clock=self.clock,
                            timeout_s=self.timeouts.idle_s,
                            phase=TimeoutPhase.IDLE,
                            deadline=ctx.deadline,
                            backend=attempt.handle.name,
                        )
                    except StopAsyncIteration:
                        break
                    attempt.acc.observe(chunk, self.clock.now())
                    yield chunk
                    if chunk.is_final:
                        break
        except InferenceError as err:
            await self._close_attempt(attempt, err)
            raise
        except Exception as exc:
            wrapped = InferenceError(
                f"unexpected backend error: {type(exc).__name__}",
                kind=ErrorKind.BACKEND_ERROR,
                backend=attempt.handle.name,
            )
            await self._close_attempt(attempt, wrapped)
            raise wrapped from exc
        except BaseException:  # cancellation / GeneratorExit: neutral for the backend
            await self._close_attempt(attempt, None)
            raise
        await self._close_attempt(attempt, None, success=True)

    async def _close_attempt(
        self, attempt: _Attempt, error: InferenceError | None, *, success: bool = False
    ) -> None:
        if attempt.closed:
            return
        attempt.closed = True
        handle = attempt.handle
        try:
            aclose = getattr(attempt.stream, "aclose", None)
            if aclose is not None:
                await aclose()
        except Exception:  # pragma: no cover - closing must never mask the real outcome
            log.debug("error while closing backend stream", exc_info=True)
        finally:
            if success:
                attempt.permit.success()
                handle.stats.on_complete(ok=True, tpot_s=self._tpot(attempt))
            elif error is not None:
                if error.counts_against_backend:
                    attempt.permit.failure()
                else:
                    attempt.permit.ignore()
                handle.stats.on_complete(ok=False)
                self.metrics.errors.labels(handle.name, error.kind.value).inc()
            else:
                attempt.permit.ignore()
                handle.stats.on_complete(ok=None)
            self.metrics.backend_outstanding.labels(handle.name).set(handle.stats.outstanding)
            self.metrics.circuit_state.labels(handle.name).set(handle.breaker.state.gauge_value)

    def _may_retry_after_output(self, ctx: _Ctx, err: InferenceError, attempts: int) -> bool:
        if not self.retry.should_retry(
            err, attempts_made=attempts, output_started=True, streaming=False
        ):
            return False
        if ctx.deadline is not None and self.clock.now() >= ctx.deadline:
            return False
        return self.retry_budget.try_spend()

    # ================================================================== accounting
    def _usage(self, ctx: _Ctx, attempt: _Attempt) -> TokenUsage:
        acc = attempt.acc
        if acc.usage is not None:
            return acc.usage
        output = (
            acc.counted if acc.counts_known else max(1, round(acc.chars / 4)) if acc.chars else 0
        )
        return TokenUsage(
            self._counter.count(ctx.request.text_for_estimation()),
            output,
            TokenCountSource.ESTIMATED,
        )

    def _tpot(self, attempt: _Attempt) -> float | None:
        acc = attempt.acc
        if acc.first_token_at is None or acc.last_token_at is None:
            return None
        n = acc.usage.output_tokens if acc.usage is not None else acc.counted
        if n < 2:
            return None
        return (acc.last_token_at - acc.first_token_at) / (n - 1)

    def _response(self, ctx: _Ctx, attempt: _Attempt, text: str) -> InferenceResponse:
        acc = attempt.acc
        end = acc.last_token_at or self.clock.now()
        return InferenceResponse(
            request_id=ctx.request.request_id,
            model=ctx.request.model,
            backend=attempt.handle.name,
            deployment=attempt.handle.deployment,
            text=text,
            finish_reason=acc.finish,
            usage=self._usage(ctx, attempt),
            timings=GatewayTimings(
                queue_s=ctx.queue_s,
                ttft_s=None if acc.first_token_at is None else acc.first_token_at - ctx.received_at,
                e2e_s=end - ctx.received_at,
            ),
            attempts=attempt.attempts,
            engine=acc.engine,
        )

    def _finish_request(
        self, ctx: _Ctx, attempt: _Attempt | None, exc: BaseException | None
    ) -> None:
        """Request-level bookkeeping; idempotent."""
        if ctx.released:
            return
        ctx.released = True
        backend = attempt.handle.name if attempt is not None else "none"
        usage = self._usage(ctx, attempt) if attempt is not None and exc is None else None
        if ctx.permit is not None:
            ctx.permit.release()
        if ctx.lease is not None:
            ctx.lease.release(usage.total_tokens if usage is not None else None)
        if exc is None and attempt is not None and usage is not None:
            status = "ok"
            acc = attempt.acc
            if acc.first_token_at is not None:
                self.metrics.ttft.labels(backend).observe(acc.first_token_at - ctx.received_at)
            tpot = self._tpot(attempt)
            if tpot is not None:
                self.metrics.tpot.labels(backend).observe(tpot)
            end = acc.last_token_at or self.clock.now()
            self.metrics.e2e.labels(backend).observe(end - ctx.received_at)
            src = usage.source.value
            self.metrics.tokens.labels(backend, "input", src).inc(usage.input_tokens)
            self.metrics.tokens.labels(backend, "output", src).inc(usage.output_tokens)
        elif isinstance(exc, InferenceError):
            status = _status(exc)
        elif isinstance(exc, asyncio.CancelledError | GeneratorExit):
            status = "cancelled"
        else:
            status = "error"
        self.metrics.requests.labels(ctx.request.model, backend, status).inc()
        ctx.span.set_attribute("inferencelab.status", status)
        ctx.span.set_attribute("inferencelab.backend", backend)
        ctx.span.end()


class GatewayStream:
    """A started generation whose first token is available. Iterate, then it cleans up."""

    def __init__(self, service: InferenceService, ctx: _Ctx, attempt: _Attempt) -> None:
        self._service = service
        self._ctx = ctx
        self._attempt = attempt

    @property
    def request_id(self) -> str:
        return self._ctx.request.request_id

    @property
    def backend(self) -> str:
        return self._attempt.handle.name

    @property
    def deployment(self) -> str | None:
        return self._attempt.handle.deployment

    @property
    def attempts(self) -> int:
        return self._attempt.attempts

    @property
    def queue_s(self) -> float:
        return self._ctx.queue_s

    @property
    def policy(self) -> str:
        return self._attempt.decision.policy

    def usage(self) -> TokenUsage:
        return self._service._usage(self._ctx, self._attempt)

    async def chunks(self) -> AsyncGenerator[BackendChunk, None]:
        error: BaseException | None = None
        drain = self._service._drain(self._ctx, self._attempt)
        try:
            # aclosing: `async for` alone would leave the backend stream open on early exit
            async with contextlib.aclosing(drain):
                async for chunk in drain:
                    yield chunk
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._service._finish_request(self._ctx, self._attempt, error)

    async def aclose(self) -> None:
        """Abandon the stream (e.g. client disconnected before iteration finished)."""
        if not self._attempt.closed:
            await self._service._close_attempt(self._attempt, None)
        self._service._finish_request(
            self._ctx, self._attempt, None if self._ctx.released else asyncio.CancelledError()
        )
