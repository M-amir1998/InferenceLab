"""Serve a local backend (mock or simulated engine) over an OpenAI-compatible HTTP API.

Used by the Docker Compose demo, the E2E tests and ``inferencelab demo`` as a stand-in
for a real engine server. Fault injection is faithful at the protocol level: a
*malformed* fault emits an unparseable SSE payload and a *disconnect* fault drops the
connection mid-stream, so the gateway's HTTP client sees exactly what it would see
from a misbehaving engine. ``/admin/fault`` (opt-in) toggles faults at runtime.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pydantic
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from inferencelab import __version__
from inferencelab.backends.base import (
    BackendChunk,
    BackendMetrics,
    BackendRequest,
    InferenceBackend,
)
from inferencelab.backends.config import FaultWindow
from inferencelab.domain.errors import (
    BackendHTTPError,
    ErrorKind,
    InferenceError,
    MalformedResponseError,
    RequestValidationError,
    StreamInterruptedError,
)
from inferencelab.domain.request import ChatMessage, RequestKind, SamplingParams
from inferencelab.openai_api.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    chat_completion,
    completion_response,
    error_body,
    models_list,
)
from inferencelab.openai_api.streaming import ClosingStreamingResponse, openai_sse

__all__ = ["create_backend_app", "render_backend_metrics"]

_PREFIX = "inferencelab_engine_"


def render_backend_metrics(m: BackendMetrics, *, backend: str, kind: str) -> str:
    """Prometheus text exposition of engine metrics; absent values are omitted."""
    lines = [
        f"# HELP {_PREFIX}info Static information about the serving engine",
        f"# TYPE {_PREFIX}info gauge",
        f'{_PREFIX}info{{backend="{backend}",type="{kind}",version="{__version__}"}} 1',
    ]

    def emit(name: str, value: float | None, mtype: str, help_text: str) -> None:
        if value is None:
            return
        lines.extend(
            [
                f"# HELP {_PREFIX}{name} {help_text}",
                f"# TYPE {_PREFIX}{name} {mtype}",
                f"{_PREFIX}{name} {value}",
            ]
        )

    emit("running_requests", m.running_requests, "gauge", "Sequences being decoded or prefilled")
    emit("waiting_requests", m.waiting_requests, "gauge", "Sequences waiting for admission")
    emit("kv_cache_usage_ratio", m.kv_cache_usage, "gauge", "Fraction of KV blocks in use")
    emit("prefix_cache_queries_total", m.prefix_cache_queries, "counter", "Prompt tokens looked up")
    emit(
        "prefix_cache_hits_total", m.prefix_cache_hits, "counter", "Prompt tokens served from cache"
    )
    emit("preemptions_total", m.preemptions_total, "counter", "Sequences preempted for KV space")
    emit("prompt_tokens_total", m.prompt_tokens_total, "counter", "Prompt tokens computed")
    emit("generation_tokens_total", m.generation_tokens_total, "counter", "Tokens generated")
    for key, value in sorted(m.raw.items()):
        emit(key, value, "counter" if key.endswith("_total") else "gauge", "Engine statistic")
    return "\n".join(lines) + "\n"


def _status_for(err: InferenceError) -> int:
    if isinstance(err, BackendHTTPError):
        return err.backend_status
    if err.kind is ErrorKind.BACKEND_UNAVAILABLE:
        return 503
    return err.http_status


def _to_backend_request(
    body: ChatCompletionRequest | CompletionRequest, request_id: str
) -> BackendRequest:
    sampling_kwargs = {
        "temperature": 1.0 if body.temperature is None else body.temperature,
        "top_p": 1.0 if body.top_p is None else body.top_p,
        "stop": body.stop_sequences(),
        "seed": body.seed,
    }
    if isinstance(body, ChatCompletionRequest):
        max_tokens = body.requested_max_tokens() or 512
        return BackendRequest(
            request_id=request_id,
            model=body.model,
            kind=RequestKind.CHAT,
            sampling=SamplingParams(max_tokens=max_tokens, **sampling_kwargs),  # type: ignore[arg-type]
            messages=tuple(
                ChatMessage(role="system" if m.role == "developer" else m.role, content=m.text())
                for m in body.messages
            ),
            extra_body=body.passthrough(),
        )
    prompt = body.prompt[0] if isinstance(body.prompt, list) and body.prompt else body.prompt
    if not isinstance(prompt, str):
        raise RequestValidationError("prompt must be a string")
    return BackendRequest(
        request_id=request_id,
        model=body.model,
        kind=RequestKind.COMPLETION,
        sampling=SamplingParams(max_tokens=body.max_tokens, **sampling_kwargs),  # type: ignore[arg-type]
        prompt=prompt,
        extra_body=body.passthrough(),
    )


def create_backend_app(
    backend: InferenceBackend,
    *,
    models: Sequence[str],
    kind: str,
    admin: bool = False,
) -> FastAPI:
    served = list(models)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await backend.aclose()

    app = FastAPI(
        title=f"InferenceLab {kind} backend",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def err(e: InferenceError) -> JSONResponse:
        status = _status_for(e)
        return JSONResponse(
            error_body(e.message, kind=e.kind.value, status=status), status_code=status
        )

    async def generate(request: Request, *, chat: bool) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        try:
            raw = await request.body()
            try:
                body: ChatCompletionRequest | CompletionRequest = (
                    ChatCompletionRequest.model_validate_json(raw)
                    if chat
                    else CompletionRequest.model_validate_json(raw)
                )
            except pydantic.ValidationError as exc:
                raise RequestValidationError(
                    f"invalid request: {exc.error_count()} errors"
                ) from exc
            if body.model not in served:
                return JSONResponse(
                    error_body(
                        f"model {body.model!r} does not exist", kind="invalid_request", status=404
                    ),
                    status_code=404,
                )
            breq = _to_backend_request(body, request_id)
            if not body.stream:
                resp = await backend.infer(breq)
                maker = chat_completion if chat else completion_response
                return JSONResponse(
                    maker(
                        response_id=f"{'chatcmpl' if chat else 'cmpl'}-{request_id}",
                        model=body.model,
                        text=resp.text,
                        finish_reason=resp.finish_reason,
                        usage=resp.usage,
                    )
                )
            stream = backend.stream(breq)
            first = await anext(stream)  # failures before output -> proper status code
        except InferenceError as e:
            return err(e)
        except StopAsyncIteration:
            return err(MalformedResponseError("empty stream"))
        include_usage = bool(body.stream_options and body.stream_options.include_usage)

        async def chunks() -> AsyncGenerator[BackendChunk, None]:
            yield first
            async for chunk in stream:
                yield chunk

        async def sse() -> AsyncGenerator[bytes, None]:
            inner = chunks()
            gen = openai_sse(
                inner,
                chat=chat,
                response_id=f"{'chatcmpl' if chat else 'cmpl'}-{request_id}",
                model=body.model,
                include_usage=include_usage,
                errors="raise",
            )
            try:
                async for part in gen:
                    yield part
            except MalformedResponseError:
                yield b'data: {"choices": [{"delta": {"content": \n\n'  # deliberately broken JSON
            except StreamInterruptedError:
                raise  # abort the connection mid-body, like a crashed engine
            except InferenceError as e:
                yield (
                    b"data: "
                    + JSONResponse(
                        error_body(e.message, kind=e.kind.value, status=_status_for(e))
                    ).body
                    + b"\n\ndata: [DONE]\n\n"
                )
            finally:
                await gen.aclose()
                await inner.aclose()
                await stream.aclose()  # type: ignore[attr-defined]

        return ClosingStreamingResponse(sse(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await generate(request, chat=True)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await generate(request, chat=False)

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return models_list(served)

    @app.get("/health")
    async def health() -> Response:
        status = await backend.health()
        return JSONResponse(
            {"status": status.status.value, "detail": status.detail},
            status_code=200 if status.healthy else 503,
        )

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {"version": __version__, "engine": f"inferencelab-{kind}"}

    @app.get("/metrics")
    async def metrics() -> Response:
        text = render_backend_metrics(await backend.metrics(), backend=backend.name, kind=kind)
        return Response(text, media_type="text/plain; version=0.0.4")

    if admin:

        @app.post("/admin/fault")
        async def fault(request: Request) -> Response:
            inject = getattr(backend, "inject_fault", None)
            if inject is None:
                return JSONResponse(
                    {"error": "backend does not support fault injection"}, status_code=404
                )
            payload = await request.json()
            if isinstance(payload, dict) and payload.get("clear"):
                inject(None)
                return JSONResponse({"faults": "cleared"})
            try:
                window = FaultWindow.model_validate(payload)
            except pydantic.ValidationError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            inject(window)
            return JSONResponse({"fault": window.model_dump(mode="json")})

    return app
