"""Generic client for any OpenAI-compatible serving endpoint.

Engine adapters (vLLM, SGLang, TensorRT-LLM) subclass this and only override what is
genuinely engine-specific: capabilities, metric names and version discovery.

Cancellation: leaving the stream early closes the HTTP response, which closes the
connection; OpenAI-compatible engines abort the request when the client disconnects.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx

from inferencelab.backends.base import (
    BackendCapabilities,
    BackendChunk,
    BackendHealth,
    BackendMetrics,
    BackendRequest,
    BackendResponse,
    HealthStatus,
)
from inferencelab.backends.config import HTTPBackendConfig
from inferencelab.backends.openai_compatible.prom import (
    INFERENCELAB_SERVER_METRICS,
    MetricMap,
    build_metrics,
    parse_prometheus_text,
)
from inferencelab.domain.errors import (
    BackendHTTPError,
    BackendTimeoutError,
    BackendUnavailableError,
    InferenceError,
    MalformedResponseError,
    StreamInterruptedError,
    TimeoutPhase,
)
from inferencelab.domain.request import RequestKind
from inferencelab.domain.response import FinishReason, TokenCountSource, TokenUsage
from inferencelab.openai_api.sse import iter_sse_data, parse_stream_payload, parse_usage
from inferencelab.tokens import HeuristicTokenCounter

__all__ = ["OpenAICompatibleBackend", "translate_httpx_error"]

_ERROR_EXCERPT = 300


def translate_httpx_error(exc: httpx.HTTPError, name: str | None) -> InferenceError:
    """Classify an httpx transport failure (shared by backends and the benchmark client)."""
    label = name or "upstream"
    if isinstance(exc, httpx.ConnectTimeout | httpx.PoolTimeout):
        return BackendTimeoutError(TimeoutPhase.CONNECT, f"{label}: connect timeout", backend=name)
    if isinstance(exc, httpx.ReadTimeout | httpx.WriteTimeout):
        return BackendTimeoutError(TimeoutPhase.IDLE, f"{label}: read timeout", backend=name)
    if isinstance(exc, httpx.ConnectError):
        return BackendUnavailableError(f"{label}: connection failed: {exc}", backend=name)
    if isinstance(exc, httpx.RemoteProtocolError | httpx.ReadError):
        return StreamInterruptedError(f"{label}: connection lost: {exc}", backend=name)
    return BackendUnavailableError(f"{label}: transport error: {exc}", backend=name)


class OpenAICompatibleBackend:
    #: Engine family label used in metrics and reports.
    kind: ClassVar[str] = "openai"
    #: Generic servers: understand InferenceLab's own serving app; others report None.
    metric_map: ClassVar[MetricMap] = INFERENCELAB_SERVER_METRICS
    #: Histogram base names for engine-side phase timings (``_sum``/``_count`` deltas).
    phase_metrics: ClassVar[dict[str, str]] = {}
    #: Endpoints probed (in order) to discover the engine version.
    version_paths: ClassVar[tuple[str, ...]] = ("/version",)

    def __init__(
        self, config: HTTPBackendConfig, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._config = config
        self._owns_client = client is None
        if client is None:
            timeout = httpx.Timeout(
                connect=config.connect_timeout_s,
                read=config.read_timeout_s,
                write=30.0,
                pool=config.connect_timeout_s,
            )
            limits = httpx.Limits(
                max_connections=config.max_connections,
                max_keepalive_connections=min(config.max_connections, 128),
            )
            client = httpx.AsyncClient(
                base_url=config.base_url,
                timeout=timeout,
                limits=limits,
                verify=config.verify_tls,
                headers=self._auth_headers(),
            )
        self._client = client

    # ------------------------------------------------------------------ properties
    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> HTTPBackendConfig:
        return self._config

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            usage_in_stream=self._config.request_usage_in_stream,
            cancellation=True,
            engine_metrics=self._config.metrics_path is not None,
        )

    def _auth_headers(self) -> dict[str, str]:
        env = self._config.api_key_env
        if not env:
            return {}
        key = os.environ.get(env)
        if not key:
            raise ValueError(f"backend {self.name!r}: environment variable {env} is not set")
        return {"Authorization": f"Bearer {key}"}

    # ------------------------------------------------------------------ payloads
    def _path(self, request: BackendRequest) -> str:
        return "/v1/chat/completions" if request.kind is RequestKind.CHAT else "/v1/completions"

    def _engine_extras(self, request: BackendRequest) -> dict[str, Any]:
        """Hook for engine-specific request fields (none by default)."""
        return {}

    def build_payload(self, request: BackendRequest, *, stream: bool) -> dict[str, Any]:
        s = request.sampling
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": s.max_tokens,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "stream": stream,
        }
        if s.stop:
            body["stop"] = list(s.stop)
        if s.seed is not None:
            body["seed"] = s.seed
        if request.kind is RequestKind.CHAT:
            body["messages"] = [{"role": m.role, "content": m.content} for m in request.messages]
        else:
            body["prompt"] = request.prompt
        if stream and self._config.request_usage_in_stream:
            body["stream_options"] = {"include_usage": True}
        body.update(self._engine_extras(request))
        body.update(request.extra_body)
        return body

    # ------------------------------------------------------------------ errors
    def _http_error(self, status: int, body: bytes) -> BackendHTTPError:
        message = body[:_ERROR_EXCERPT].decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                err = parsed.get("error", parsed)
                if isinstance(err, dict) and isinstance(err.get("message"), str):
                    message = err["message"][:_ERROR_EXCERPT]
                elif isinstance(parsed.get("detail"), str):
                    message = parsed["detail"][:_ERROR_EXCERPT]
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return BackendHTTPError(status, f"{self.name}: HTTP {status}: {message}", backend=self.name)

    def _translate(self, exc: httpx.HTTPError) -> InferenceError:
        return translate_httpx_error(exc, self.name)

    # ------------------------------------------------------------------ generation
    async def stream(self, request: BackendRequest) -> AsyncIterator[BackendChunk]:
        chat = request.kind is RequestKind.CHAT
        body = self.build_payload(request, stream=True)
        try:
            async with self._client.stream("POST", self._path(request), json=body) as response:
                if response.status_code >= 400:
                    raise self._http_error(response.status_code, await response.aread())
                ctype = response.headers.get("content-type", "")
                if "text/event-stream" not in ctype:
                    raise MalformedResponseError(
                        f"{self.name}: expected text/event-stream, got {ctype!r}", backend=self.name
                    )
                usage: TokenUsage | None = None
                finish: FinishReason | None = None
                done = False
                async for payload in iter_sse_data(response.aiter_lines()):
                    parsed = parse_stream_payload(payload, chat=chat, backend=self.name)
                    if parsed.done:
                        done = True
                        break
                    usage = parsed.usage or usage
                    finish = parsed.finish_reason or finish
                    if parsed.text:
                        yield BackendChunk(text=parsed.text)
                if not done and finish is None:
                    raise StreamInterruptedError(
                        f"{self.name}: stream ended without completion", backend=self.name
                    )
                yield BackendChunk(text="", finish_reason=finish or FinishReason.STOP, usage=usage)
        except httpx.HTTPError as exc:
            raise self._translate(exc) from exc

    async def infer(self, request: BackendRequest) -> BackendResponse:
        body = self.build_payload(request, stream=False)
        try:
            response = await self._client.post(self._path(request), json=body)
        except httpx.HTTPError as exc:
            raise self._translate(exc) from exc
        if response.status_code >= 400:
            raise self._http_error(response.status_code, response.content)
        try:
            data = response.json()
            choice = data["choices"][0]
            if request.kind is RequestKind.CHAT:
                text = choice["message"].get("content") or ""
            else:
                text = choice.get("text") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise MalformedResponseError(
                f"{self.name}: unexpected response shape", backend=self.name
            ) from exc
        usage = parse_usage(data.get("usage"))
        if usage is None:
            usage = TokenUsage(
                request.estimate_input_tokens(),
                HeuristicTokenCounter().count(text),
                TokenCountSource.ESTIMATED,
            )
        return BackendResponse(
            text=text,
            finish_reason=FinishReason.parse(choice.get("finish_reason")),
            usage=usage,
            chunks=1,
        )

    # ------------------------------------------------------------------ control
    async def health(self) -> BackendHealth:
        start = time.perf_counter()
        try:
            response = await self._client.get(self._config.health_path)
        except httpx.HTTPError as exc:
            return BackendHealth(HealthStatus.UNHEALTHY, detail=f"{type(exc).__name__}: {exc}")
        latency = time.perf_counter() - start
        if response.status_code == 200:
            return BackendHealth(HealthStatus.HEALTHY, latency_s=latency)
        return BackendHealth(
            HealthStatus.UNHEALTHY, latency_s=latency, detail=f"HTTP {response.status_code}"
        )

    async def metrics(self) -> BackendMetrics:
        path = self._config.metrics_path
        if path is None:
            return BackendMetrics(source="unavailable")
        try:
            response = await self._client.get(path)
        except httpx.HTTPError:
            return BackendMetrics(source="unavailable")
        if response.status_code != 200:
            return BackendMetrics(source="unavailable")
        try:
            samples = parse_prometheus_text(response.text)
        except ValueError:
            return BackendMetrics(source="unparseable")
        return build_metrics(samples, self.metric_map, source=f"prometheus:{path}")

    async def models(self) -> list[str]:
        try:
            response = await self._client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            return [str(m["id"]) for m in data.get("data", [])]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return []

    async def version(self) -> str | None:
        """Engine version string, if the server exposes one (recorded in run metadata)."""
        for path in self.version_paths:
            try:
                response = await self._client.get(path)
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            try:
                data = response.json()
            except ValueError:
                continue
            if isinstance(data, dict):
                for key in ("version", "sglang_version", "tensorrt_llm_version"):
                    if isinstance(data.get(key), str):
                        return str(data[key])
        return None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
