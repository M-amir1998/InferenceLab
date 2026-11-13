"""FastAPI application: the OpenAI-compatible gateway.

Endpoints: ``GET /health`` (liveness), ``GET /ready`` (readiness), ``GET /metrics``,
``GET /v1/models``, ``POST /v1/chat/completions``, ``POST /v1/completions`` and
``GET /inferencelab/backends`` (operator view). FastAPI is only the HTTP shell; all
data-plane behaviour lives in :class:`~inferencelab.gateway.service.InferenceService`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

import pydantic
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from inferencelab import __version__
from inferencelab.domain.errors import (
    InferenceError,
    RequestCancelledError,
    RequestValidationError,
)
from inferencelab.domain.request import InferenceRequest
from inferencelab.gateway.auth import ApiKeyAuth
from inferencelab.gateway.config import GatewayConfig
from inferencelab.gateway.middleware import BodySizeLimitMiddleware
from inferencelab.gateway.pool import BackendHandle
from inferencelab.gateway.runtime import GatewayRuntime
from inferencelab.gateway.translate import chat_request, completion_request, request_id_from
from inferencelab.observability.prometheus import CONTENT_TYPE_LATEST
from inferencelab.observability.tracing import configure_tracing
from inferencelab.openai_api.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    chat_completion,
    completion_response,
    error_body,
    models_list,
)
from inferencelab.openai_api.streaming import ClosingStreamingResponse, openai_sse

__all__ = ["create_app"]

log = logging.getLogger(__name__)
T = TypeVar("T")


def _error(err: InferenceError, request_id: str) -> JSONResponse:
    headers = {"x-request-id": request_id}
    if err.retry_after_s is not None and err.retry_after_s != float("inf"):
        headers["retry-after"] = str(max(1, round(err.retry_after_s)))
    return JSONResponse(
        error_body(err.message, kind=err.kind.value, status=err.http_status),
        status_code=err.http_status,
        headers=headers,
    )


async def _until_disconnect(request: Request, work: Awaitable[T]) -> T:
    """Run ``work`` but cancel it if the client disconnects (frees backend capacity)."""
    task = asyncio.ensure_future(work)

    async def watch() -> None:
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    watcher = asyncio.ensure_future(watch())
    try:
        done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, InferenceError):
            await task
        raise RequestCancelledError("client disconnected")
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        if not task.done():
            task.cancel()


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else seconds * 1000


def _backend_status(h: BackendHandle) -> dict[str, Any]:
    return {
        "name": h.name,
        "type": h.config.type,
        "deployment": h.deployment,
        "models": h.config.models,
        "tier": h.config.routing.tier,
        "healthy": h.healthy,
        "circuit": h.breaker.state.value,
        "outstanding": h.stats.outstanding,
        "ewma_ttft_ms": _ms(h.stats.ewma_ttft_s),
        "ewma_tpot_ms": _ms(h.stats.ewma_tpot_s),
        "capabilities": h.backend.capabilities.as_dict(),
    }


def create_app(config: GatewayConfig, *, runtime: GatewayRuntime | None = None) -> FastAPI:
    rt = runtime or GatewayRuntime(config)
    auth = ApiKeyAuth.from_env(config.auth.api_keys_env) if config.auth.enabled else None
    configure_tracing(config.observability.tracing)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await rt.start()
        try:
            yield
        finally:
            await rt.stop()

    app = FastAPI(
        title="InferenceLab Gateway",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=config.limits.max_request_bytes)
    app.state.runtime = rt

    def tenant_for(request: Request) -> str | None:
        if auth is None:
            return None
        return auth.authenticate(
            request.headers.get("authorization"), request.headers.get("x-api-key")
        )

    async def handle(request: Request, *, chat: bool) -> Response:
        request_id = request_id_from(request.headers)
        try:
            tenant = tenant_for(request)
            raw = await request.body()
            try:
                if chat:
                    inference = chat_request(
                        ChatCompletionRequest.model_validate_json(raw),
                        headers=request.headers,
                        request_id=request_id,
                        limits=config.limits,
                        tenant=tenant,
                    )
                else:
                    inference = completion_request(
                        CompletionRequest.model_validate_json(raw),
                        headers=request.headers,
                        request_id=request_id,
                        limits=config.limits,
                        tenant=tenant,
                    )
            except pydantic.ValidationError as exc:
                first = exc.errors()[0] if exc.errors() else {"msg": "invalid request"}
                loc = ".".join(str(p) for p in first.get("loc", ()))
                raise RequestValidationError(f"invalid request: {loc} {first.get('msg')}") from exc
            if inference.stream:
                return await stream(inference, chat=chat)
            return await _until_disconnect(request, unary(inference, chat=chat))
        except InferenceError as err:
            return _error(err, request_id)

    async def unary(inference: InferenceRequest, *, chat: bool) -> Response:
        response = await rt.service.complete(inference)
        maker = chat_completion if chat else completion_response
        prefix = "chatcmpl" if chat else "cmpl"
        body = maker(
            response_id=f"{prefix}-{inference.request_id}",
            model=inference.model,
            text=response.text,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )
        return JSONResponse(
            body,
            headers={
                "x-request-id": inference.request_id,
                "x-inferencelab-backend": response.backend,
                "x-inferencelab-deployment": response.deployment or "",
                "x-inferencelab-attempts": str(response.attempts),
                "x-inferencelab-queue-ms": f"{response.timings.queue_s * 1000:.3f}",
                "x-inferencelab-token-source": response.usage.source.value,
            },
        )

    async def stream(inference: InferenceRequest, *, chat: bool) -> Response:
        gs = await rt.service.open_stream(inference)
        prefix = "chatcmpl" if chat else "cmpl"

        async def body() -> AsyncGenerator[bytes, None]:
            chunks = gs.chunks()
            sse = openai_sse(
                chunks,
                chat=chat,
                response_id=f"{prefix}-{inference.request_id}",
                model=inference.model,
                include_usage=inference.include_usage,
                fallback_usage=gs.usage,
            )
            try:
                async for part in sse:
                    yield part
            finally:
                await sse.aclose()
                await chunks.aclose()
                await gs.aclose()

        return ClosingStreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
                "x-request-id": inference.request_id,
                "x-inferencelab-backend": gs.backend,
                "x-inferencelab-deployment": gs.deployment or "",
                "x-inferencelab-attempts": str(gs.attempts),
                "x-inferencelab-queue-ms": f"{gs.queue_s * 1000:.3f}",
                "x-inferencelab-route-policy": gs.policy,
            },
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await handle(request, chat=True)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await handle(request, chat=False)

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        try:
            tenant_for(request)
        except InferenceError as err:
            return _error(err, request_id_from(request.headers))
        return JSONResponse(models_list(rt.pool.models()))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> Response:
        ok = rt.pool.any_ready()
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(rt.metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/inferencelab/backends")
    async def backends(request: Request) -> Response:
        try:
            tenant_for(request)
        except InferenceError as err:
            return _error(err, request_id_from(request.headers))
        out = [_backend_status(h) for h in rt.pool.handles]
        return JSONResponse(
            {
                "backends": out,
                "queue": {"in_flight": rt.admission.in_flight, "pending": rt.admission.pending},
            }
        )

    return app
