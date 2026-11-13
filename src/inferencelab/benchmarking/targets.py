"""Benchmark targets: what the load generator sends requests to.

* :class:`HTTPTarget`     any OpenAI-compatible endpoint — an InferenceLab gateway or an
                          engine server directly (the GPU benchmark path).
* :class:`BackendTarget`  an in-process backend (no HTTP), e.g. the CPU simulator.
* :class:`ServiceTarget`  the gateway data plane in-process (routing/admission without
                          HTTP), deterministic under a :class:`~inferencelab.clock.VirtualClock`.

Targets only produce events; the load generator owns all timestamps.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol

import httpx

from inferencelab.backends.base import BackendRequest, InferenceBackend
from inferencelab.backends.openai_compatible.backend import translate_httpx_error
from inferencelab.domain.errors import BackendHTTPError, ErrorKind, InferenceError
from inferencelab.domain.request import (
    ChatMessage,
    InferenceRequest,
    Priority,
    RequestKind,
    RoutingHints,
    SamplingParams,
)
from inferencelab.domain.response import EngineTimings, FinishReason, TokenUsage
from inferencelab.gateway.service import InferenceService
from inferencelab.openai_api.sse import iter_sse_data, parse_stream_payload

__all__ = [
    "BackendTarget",
    "BenchmarkTarget",
    "EventKind",
    "HTTPTarget",
    "ServiceTarget",
    "TargetEvent",
    "TargetMeta",
    "TargetRequest",
]

_KINDS = {k.value for k in ErrorKind}


class EventKind(StrEnum):
    META = "meta"
    CHUNK = "chunk"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class TargetMeta:
    backend: str | None = None
    deployment: str | None = None
    attempts: int | None = None
    queue_s: float | None = None


@dataclass(frozen=True, slots=True)
class TargetEvent:
    kind: EventKind
    text: str = ""
    tokens: int | None = None
    usage: TokenUsage | None = None
    finish_reason: FinishReason | None = None
    engine: EngineTimings | None = None
    meta: TargetMeta | None = None


@dataclass(frozen=True, slots=True)
class TargetRequest:
    request_id: str
    model: str
    api: Literal["chat", "completions"]
    max_tokens: int
    messages: tuple[ChatMessage, ...] = ()
    prompt: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = True
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    tenant_id: str | None = None
    priority: str = "normal"
    conversation_id: str | None = None
    prefix_key: str | None = None

    def extras(self) -> dict[str, Any]:
        extra = dict(self.extra_body)
        if self.ignore_eos:
            extra.setdefault("ignore_eos", True)
        return extra


class BenchmarkTarget(Protocol):
    description: str

    def stream(self, request: TargetRequest) -> AsyncGenerator[TargetEvent, None]: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------- HTTP
class HTTPTarget:
    def __init__(
        self,
        base_url: str,
        *,
        api_key_env: str | None = None,
        headers: Mapping[str, str] | None = None,
        max_connections: int = 1024,
        connect_timeout_s: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.description = f"http:{self.base_url}"
        base_headers = dict(headers or {})
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ValueError(f"environment variable {api_key_env} is not set")
            base_headers["authorization"] = f"Bearer {key}"
        self._owns = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            headers=base_headers,
            # read timeouts are enforced by the load generator's clock, not httpx
            timeout=httpx.Timeout(connect=connect_timeout_s, read=None, write=30.0, pool=None),
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_connections
            ),
        )

    def _body(self, req: TargetRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if req.api == "chat":
            body["messages"] = [{"role": m.role, "content": m.content} for m in req.messages]
        else:
            body["prompt"] = req.prompt
        body.update(req.extras())
        return body

    @staticmethod
    def _headers(req: TargetRequest) -> dict[str, str]:
        headers = {"x-request-id": req.request_id, "x-priority": req.priority}
        if req.tenant_id:
            headers["x-tenant-id"] = req.tenant_id
        if req.conversation_id:
            headers["x-conversation-id"] = req.conversation_id
        if req.prefix_key:
            headers["x-prefix-key"] = req.prefix_key
        return headers

    @staticmethod
    def _error(status: int, body: bytes) -> InferenceError:
        message = body[:300].decode("utf-8", errors="replace")
        with contextlib.suppress(ValueError, AttributeError, TypeError):
            err = json.loads(body).get("error", {})
            message = str(err.get("message", message))
            code = err.get("code")
            if isinstance(code, str) and code in _KINDS:
                return InferenceError(message, kind=ErrorKind(code), http_status=status)
        return BackendHTTPError(status, message)

    async def stream(self, request: TargetRequest) -> AsyncGenerator[TargetEvent, None]:
        path = "/v1/chat/completions" if request.api == "chat" else "/v1/completions"
        chat = request.api == "chat"
        try:
            async with self._client.stream(
                "POST", path, json=self._body(request), headers=self._headers(request)
            ) as response:
                if response.status_code >= 400:
                    raise self._error(response.status_code, await response.aread())
                h = response.headers
                queue_ms = h.get("x-inferencelab-queue-ms")
                attempts = h.get("x-inferencelab-attempts")
                yield TargetEvent(
                    EventKind.META,
                    meta=TargetMeta(
                        backend=h.get("x-inferencelab-backend") or None,
                        deployment=h.get("x-inferencelab-deployment") or None,
                        attempts=int(attempts) if attempts and attempts.isdigit() else None,
                        queue_s=float(queue_ms) / 1000.0 if queue_ms else None,
                    ),
                )
                usage: TokenUsage | None = None
                finish: FinishReason | None = None
                async for payload in iter_sse_data(response.aiter_lines()):
                    parsed = parse_stream_payload(payload, chat=chat)
                    if parsed.done:
                        break
                    usage = parsed.usage or usage
                    finish = parsed.finish_reason or finish
                    if parsed.text:
                        yield TargetEvent(EventKind.CHUNK, text=parsed.text)
                yield TargetEvent(EventKind.FINAL, usage=usage, finish_reason=finish)
        except httpx.HTTPError as exc:
            raise translate_httpx_error(exc, None) from exc

    async def aclose(self) -> None:
        if self._owns:
            await self._client.aclose()


# --------------------------------------------------------------------------- in-process
def _messages_or_prompt(
    req: TargetRequest,
) -> tuple[RequestKind, tuple[ChatMessage, ...], str | None]:
    if req.api == "chat":
        return RequestKind.CHAT, req.messages, None
    return RequestKind.COMPLETION, (), req.prompt


class BackendTarget:
    """Drive a backend directly (no gateway, no HTTP)."""

    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend
        self.description = f"backend:{backend.name}"

    async def stream(self, request: TargetRequest) -> AsyncGenerator[TargetEvent, None]:
        kind, messages, prompt = _messages_or_prompt(request)
        breq = BackendRequest(
            request_id=request.request_id,
            model=request.model,
            kind=kind,
            sampling=SamplingParams(
                max_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p
            ),
            messages=messages,
            prompt=prompt,
            extra_body=request.extras(),
            prefix_key=request.prefix_key,
        )
        yield TargetEvent(EventKind.META, meta=TargetMeta(backend=self.backend.name, attempts=1))
        stream = self.backend.stream(breq)
        try:
            async for chunk in stream:
                if chunk.text:
                    yield TargetEvent(EventKind.CHUNK, text=chunk.text, tokens=chunk.token_count)
                if chunk.is_final:
                    yield TargetEvent(
                        EventKind.FINAL,
                        usage=chunk.usage,
                        finish_reason=chunk.finish_reason,
                        engine=chunk.engine,
                    )
                    return
            yield TargetEvent(EventKind.FINAL)
        finally:
            await stream.aclose()  # type: ignore[attr-defined]

    async def aclose(self) -> None:
        return None


class ServiceTarget:
    """Drive the gateway's data plane in-process (routing, admission, retries)."""

    def __init__(self, service: InferenceService) -> None:
        self.service = service
        self.description = "gateway:in-process"

    async def stream(self, request: TargetRequest) -> AsyncGenerator[TargetEvent, None]:
        kind, messages, prompt = _messages_or_prompt(request)
        inference = InferenceRequest(
            request_id=request.request_id,
            model=request.model,
            kind=kind,
            sampling=SamplingParams(
                max_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p
            ),
            messages=messages,
            prompt=prompt,
            stream=True,
            include_usage=True,
            priority=Priority.parse(request.priority),
            hints=RoutingHints(
                tenant_id=request.tenant_id,
                conversation_id=request.conversation_id,
                prefix_key=request.prefix_key,
            ),
            extra_body=request.extras(),
        )
        gs = await self.service.open_stream(inference)
        try:
            yield TargetEvent(
                EventKind.META,
                meta=TargetMeta(
                    backend=gs.backend,
                    deployment=gs.deployment,
                    attempts=gs.attempts,
                    queue_s=gs.queue_s,
                ),
            )
            chunks = gs.chunks()
            try:
                async for chunk in chunks:
                    if chunk.text:
                        yield TargetEvent(
                            EventKind.CHUNK, text=chunk.text, tokens=chunk.token_count
                        )
                    if chunk.is_final:
                        yield TargetEvent(
                            EventKind.FINAL,
                            usage=chunk.usage,
                            finish_reason=chunk.finish_reason,
                            engine=chunk.engine,
                        )
                        return
            finally:
                await chunks.aclose()
            yield TargetEvent(EventKind.FINAL, usage=gs.usage())
        finally:
            await gs.aclose()

    async def aclose(self) -> None:
        return None
