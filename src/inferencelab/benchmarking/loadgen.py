"""Asynchronous load generator.

* **Open loop** (constant / Poisson / burst / step / trace): sessions start on schedule
  regardless of server speed. At most ``max_in_flight`` sessions run at once; if the
  bound is hit the session is *dropped and recorded* rather than delayed, because
  delaying would silently change the arrival process (coordinated omission). Every
  record keeps its intended send time, so client lag is measurable.
* **Closed loop**: exactly ``concurrency`` workers, each starting its next session when
  the previous one finishes.
* Task creation is always bounded (semaphore / fixed worker pool).
* Multi-turn sessions resend the conversation, including the text the server generated,
  so session affinity and prefix caching can be evaluated end to end.
* All time comes from the injected :class:`~inferencelab.clock.Clock`.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import hmac
import itertools
import secrets
from collections.abc import Sequence
from dataclasses import dataclass

from inferencelab.benchmarking.targets import BenchmarkTarget, EventKind, TargetRequest
from inferencelab.clock import Clock
from inferencelab.domain.errors import ErrorKind, InferenceError
from inferencelab.domain.request import ChatMessage
from inferencelab.domain.response import EngineTimings, TokenCountSource, TokenUsage
from inferencelab.domain.result import Phase, RequestRecord, RequestStatus
from inferencelab.domain.workload import WarmupSpec, WorkloadSpec
from inferencelab.tokens import HeuristicTokenCounter, estimate_chat_tokens
from inferencelab.workloads.generator import SessionPlan, TurnPlan, WorkloadPlan

__all__ = ["LoadGenerator", "LoadResult", "assign_phases"]

_REJECTED = {
    ErrorKind.RATE_LIMITED,
    ErrorKind.QUEUE_FULL,
    ErrorKind.LOAD_SHED,
    ErrorKind.NO_HEALTHY_BACKEND,
    ErrorKind.CIRCUIT_OPEN,
    ErrorKind.BACKEND_OVERLOADED,
}


@dataclass(frozen=True, slots=True)
class LoadResult:
    records: list[RequestRecord]
    duration_s: float
    started_at: float  # clock time of run start


def _status_for(err: InferenceError) -> RequestStatus:
    if err.kind.value.startswith("timeout"):
        return RequestStatus.TIMEOUT
    if err.kind in _REJECTED:
        return RequestStatus.REJECTED
    if err.kind is ErrorKind.CANCELLED:
        return RequestStatus.CANCELLED
    return RequestStatus.ERROR


def assign_phases(
    records: Sequence[RequestRecord], warmup: WarmupSpec, cooldown: WarmupSpec
) -> list[RequestRecord]:
    """Mark warmup/cooldown requests (excluded from metrics) by index and/or time."""
    ordered = sorted(records, key=lambda r: r.index)
    if not ordered:
        return []
    last_index = ordered[-1].index
    last_sched = max(r.scheduled_at for r in ordered)
    out = []
    for r in ordered:
        phase = Phase.MEASURE
        if r.index < warmup.requests or r.scheduled_at < warmup.seconds:
            phase = Phase.WARMUP
        elif r.index > last_index - cooldown.requests or (
            cooldown.seconds and r.scheduled_at > last_sched - cooldown.seconds
        ):
            phase = Phase.COOLDOWN
        out.append(dataclasses.replace(r, phase=phase))
    return out


class LoadGenerator:
    def __init__(
        self,
        target: BenchmarkTarget,
        plan: WorkloadPlan,
        *,
        clock: Clock,
        run_tag: str = "run",
        capture_text: bool = False,
    ) -> None:
        self.target = target
        self.plan = plan
        self.spec: WorkloadSpec = plan.spec
        self.clock = clock
        self.run_tag = run_tag
        self.capture_text = capture_text
        self._salt = secrets.token_bytes(16)  # per-run, never persisted
        self._records: list[RequestRecord] = []
        self._index = itertools.count()
        self._t0 = 0.0
        self._counter = HeuristicTokenCounter()

    # ------------------------------------------------------------------ driver
    async def run(self) -> LoadResult:
        self._t0 = self.clock.now()
        concurrency = self.plan.closed_loop_concurrency
        if concurrency is not None:
            await self._run_closed(concurrency)
        else:
            await self._run_open()
        duration = self.clock.now() - self._t0
        records = assign_phases(self._records, self.spec.warmup, self.spec.cooldown)
        return LoadResult(records=records, duration_s=duration, started_at=self._t0)

    def _elapsed(self) -> float:
        return self.clock.now() - self._t0

    def _expired(self) -> bool:
        return self.spec.duration_s is not None and self._elapsed() >= self.spec.duration_s

    async def _run_open(self) -> None:
        sem = asyncio.Semaphore(self.spec.max_in_flight)
        async with asyncio.TaskGroup() as tg:
            for session in self.plan.sessions():
                offset = session.start_offset_s or 0.0
                if self.spec.duration_s is not None and offset >= self.spec.duration_s:
                    break
                await self.clock.sleep(self._t0 + offset - self.clock.now())
                if sem.locked():
                    self._drop(session, offset)
                    continue
                await sem.acquire()
                tg.create_task(self._guarded(session, offset, sem))

    async def _guarded(self, session: SessionPlan, offset: float, sem: asyncio.Semaphore) -> None:
        try:
            await self._session(session, offset)
        finally:
            sem.release()

    async def _run_closed(self, concurrency: int) -> None:
        sessions = self.plan.sessions()

        async def worker() -> None:
            while not self._expired():
                session = next(sessions, None)
                if session is None:
                    return
                await self._session(session, self._elapsed())

        async with asyncio.TaskGroup() as tg:
            for _ in range(concurrency):
                tg.create_task(worker())

    def _drop(self, session: SessionPlan, offset: float) -> None:
        for turn in session.turns:
            index = next(self._index)
            self._records.append(
                RequestRecord(
                    index=index,
                    request_id=f"{self.run_tag}-{index:07d}",
                    phase=Phase.MEASURE,
                    status=RequestStatus.DROPPED,
                    scheduled_at=offset,
                    sent_at=None,
                    first_token_at=None,
                    completed_at=None,
                    requested_input_tokens=turn.input_tokens,
                    requested_output_tokens=turn.max_tokens,
                    session_id=session.session_id,
                    prefix_group=session.prefix_group,
                    tenant_id=session.tenant_id,
                    priority=session.priority,
                    error_kind="client_max_in_flight",
                )
            )

    # ------------------------------------------------------------------ sessions
    def _history(self, session: SessionPlan) -> list[ChatMessage]:
        history: list[ChatMessage] = []
        prefix = self.plan.prefix_text(session.prefix_group)
        if session.system_prompt:
            history.append(ChatMessage("system", session.system_prompt))
        if prefix is not None:
            history.append(ChatMessage("system", prefix))
        return history

    async def _session(self, session: SessionPlan, scheduled: float) -> None:
        history = self._history(session)
        for turn_no, turn in enumerate(session.turns):
            if turn_no > 0:
                if session.think_time_s:
                    await self.clock.sleep(session.think_time_s)
                scheduled = self._elapsed()
            messages = [*history, ChatMessage("user", turn.user_text)]
            record, output = await self._one(session, turn, turn_no, messages, scheduled)
            self._records.append(record)
            if record.status is not RequestStatus.OK:
                return  # the simulated user abandons the conversation
            history = [*messages, ChatMessage("assistant", output)]

    def _fingerprint(self, text: str) -> str:
        return hmac.new(self._salt, text.encode(), hashlib.sha256).hexdigest()[:16]

    async def _one(
        self,
        session: SessionPlan,
        turn: TurnPlan,
        turn_no: int,
        messages: list[ChatMessage],
        scheduled: float,
    ) -> tuple[RequestRecord, str]:
        spec = self.spec
        index = next(self._index)
        request_id = f"{self.run_tag}-{index:07d}"
        if spec.api == "chat":
            req = TargetRequest(
                request_id=request_id,
                model=spec.model,
                api="chat",
                max_tokens=turn.max_tokens,
                messages=tuple(messages),
                temperature=spec.temperature,
                top_p=spec.top_p,
                ignore_eos=spec.ignore_eos,
                extra_body=spec.extra_body,
                tenant_id=session.tenant_id,
                priority=session.priority,
                conversation_id=session.session_id,
                prefix_key=session.prefix_key,
            )
            prompt_text = "\n".join(m.content for m in messages)
        else:
            prompt_text = " ".join(m.content for m in messages)
            req = TargetRequest(
                request_id=request_id,
                model=spec.model,
                api="completions",
                max_tokens=turn.max_tokens,
                prompt=prompt_text,
                temperature=spec.temperature,
                top_p=spec.top_p,
                ignore_eos=spec.ignore_eos,
                extra_body=spec.extra_body,
                tenant_id=session.tenant_id,
                priority=session.priority,
                conversation_id=session.session_id,
                prefix_key=session.prefix_key,
            )
        chunk_times: list[float] = []
        parts: list[str] = []
        counted = 0
        counts_known = True
        usage: TokenUsage | None = None
        engine: EngineTimings | None = None
        finish: str | None = None
        meta_backend = meta_deployment = None
        attempts: int | None = None
        queue_s: float | None = None
        status = RequestStatus.OK
        error_kind: str | None = None
        http_status: int | None = None
        message: str | None = None
        first: float | None = None
        sent = self._elapsed()
        try:
            async with self.clock.timeout(spec.request_timeout_s):
                stream = self.target.stream(req)
                async with contextlib.aclosing(stream):
                    async for event in stream:
                        if event.kind is EventKind.META and event.meta is not None:
                            meta_backend = event.meta.backend
                            meta_deployment = event.meta.deployment
                            attempts = event.meta.attempts
                            queue_s = event.meta.queue_s
                        elif event.kind is EventKind.CHUNK and event.text:
                            now = self._elapsed()
                            chunk_times.append(now)
                            if first is None:
                                first = now
                            parts.append(event.text)
                            if event.tokens is None:
                                counts_known = False
                            else:
                                counted += event.tokens
                            if (
                                turn.cancel_after_tokens is not None
                                and len(chunk_times) >= turn.cancel_after_tokens
                            ):
                                status = RequestStatus.CANCELLED
                                break
                        elif event.kind is EventKind.FINAL:
                            usage = event.usage or usage
                            engine = event.engine or engine
                            finish = event.finish_reason.value if event.finish_reason else None
        except TimeoutError:
            status, error_kind = RequestStatus.TIMEOUT, "client_timeout"
            message = f"client timeout after {spec.request_timeout_s}s"
        except InferenceError as err:
            status, error_kind, http_status = _status_for(err), err.kind.value, err.http_status
            message = err.message[:200]
        completed = self._elapsed()
        output_text = "".join(parts)
        if usage is not None:
            in_tok, out_tok, source = (
                usage.input_tokens,
                usage.output_tokens,
                TokenCountSource.REPORTED,
            )
            cached = usage.cached_input_tokens
        else:
            in_tok = (
                estimate_chat_tokens(messages, self._counter)
                if spec.api == "chat"
                else self._counter.count(prompt_text)
            )
            out_tok = counted if counts_known else self._counter.count(output_text)
            source, cached = TokenCountSource.ESTIMATED, None
        if status is not RequestStatus.OK and status is not RequestStatus.CANCELLED:
            in_tok_out: int | None = None
            out_tok_out: int | None = None
        else:
            in_tok_out, out_tok_out = in_tok, out_tok
        record = RequestRecord(
            index=index,
            request_id=request_id,
            phase=Phase.MEASURE,
            status=status,
            scheduled_at=scheduled,
            sent_at=sent,
            first_token_at=first,
            completed_at=completed,
            chunk_times=tuple(chunk_times),
            input_tokens=in_tok_out,
            output_tokens=out_tok_out,
            token_source=source,
            cached_input_tokens=cached if status is RequestStatus.OK else None,
            requested_input_tokens=turn.input_tokens,
            requested_output_tokens=turn.max_tokens,
            backend=meta_backend,
            deployment=meta_deployment,
            attempts=attempts,
            gateway_queue_s=queue_s,
            engine_queue_s=engine.queue_s if engine else None,
            engine_prefill_s=engine.prefill_s if engine else None,
            engine_decode_s=engine.decode_s if engine else None,
            finish_reason=finish,
            error_kind=error_kind,
            http_status=http_status,
            error_message=message,
            session_id=session.session_id,
            turn=turn_no,
            prefix_group=session.prefix_group,
            tenant_id=session.tenant_id,
            priority=session.priority,
            prompt_fingerprint=self._fingerprint(prompt_text),
            prompt_text=prompt_text if self.capture_text else None,
            output_text=output_text if self.capture_text else None,
        )
        return record, output_text
