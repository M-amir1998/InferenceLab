from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from inferencelab.backends.base import BackendChunk
from inferencelab.backends.config import (
    HTTPBackendConfig,
    OpenAIBackendConfig,
    SGLangBackendConfig,
    TensorRTLLMBackendConfig,
    VLLMBackendConfig,
)
from inferencelab.backends.openai_compatible.backend import OpenAICompatibleBackend
from inferencelab.backends.openai_compatible.prom import engine_phase_means, parse_prometheus_text
from inferencelab.backends.sglang.backend import SGLangBackend
from inferencelab.backends.tensorrt_llm.backend import TensorRTLLMBackend
from inferencelab.backends.vllm.backend import VLLMBackend
from inferencelab.domain.errors import (
    BackendHTTPError,
    BackendTimeoutError,
    BackendUnavailableError,
    ErrorKind,
    MalformedResponseError,
    StreamInterruptedError,
)
from inferencelab.domain.response import FinishReason, TokenCountSource
from tests.conftest import backend_request

Handler = Callable[[httpx.Request], httpx.Response]


def _sse(*events: dict[str, object] | str) -> bytes:
    out = []
    for e in events:
        out.append(f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n")
    return "".join(out).encode()


def _backend(
    handler: Handler, cls: type[OpenAICompatibleBackend] = OpenAICompatibleBackend, **cfg: object
) -> OpenAICompatibleBackend:
    config_types: dict[type[OpenAICompatibleBackend], type[HTTPBackendConfig]] = {
        OpenAICompatibleBackend: OpenAIBackendConfig,
        VLLMBackend: VLLMBackendConfig,
        SGLangBackend: SGLangBackendConfig,
        TensorRTLLMBackend: TensorRTLLMBackendConfig,
    }
    config = config_types[cls].model_validate({"name": "b", "base_url": "http://engine", **cfg})
    client = httpx.AsyncClient(base_url="http://engine", transport=httpx.MockTransport(handler))
    return cls(config, client=client)


async def _stream(backend: OpenAICompatibleBackend) -> list[BackendChunk]:
    return [c async for c in backend.stream(backend_request())]


async def test_streaming_with_usage_and_request_shape() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        body = _sse(
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}},
            "[DONE]",
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    chunks = await _stream(_backend(handler))
    assert [c.text for c in chunks if c.text] == ["Hello", " world"]
    final = chunks[-1]
    assert final.finish_reason is FinishReason.LENGTH
    assert final.usage is not None and final.usage.source is TokenCountSource.REPORTED
    assert seen["stream"] is True and seen["stream_options"] == {"include_usage": True}
    assert seen["temperature"] == 0.0 and seen["max_tokens"] == 8 and "messages" in seen


async def test_http_errors_are_classified() -> None:
    for status, kind in (
        (500, ErrorKind.BACKEND_ERROR),
        (503, ErrorKind.BACKEND_OVERLOADED),
        (400, ErrorKind.BACKEND_REJECTED),
    ):

        def handler(req: httpx.Request, status: int = status) -> httpx.Response:
            return httpx.Response(status, json={"error": {"message": "nope"}})

        with pytest.raises(BackendHTTPError) as exc:
            await _stream(_backend(handler))
        assert exc.value.kind is kind and "nope" in exc.value.message


async def test_transport_failures_are_classified() -> None:
    def refuse(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    def slow(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=req)

    with pytest.raises(BackendUnavailableError):
        await _stream(_backend(refuse))
    with pytest.raises(BackendTimeoutError):
        await _stream(_backend(slow))


async def test_malformed_and_truncated_streams() -> None:
    def malformed(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"data: {broken\n\n", headers={"content-type": "text/event-stream"}
        )

    def truncated(req: httpx.Request) -> httpx.Response:
        body = _sse({"choices": [{"delta": {"content": "partial"}}]})
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    def wrong_type(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(MalformedResponseError):
        await _stream(_backend(malformed))
    with pytest.raises(StreamInterruptedError):
        await _stream(_backend(truncated))
    with pytest.raises(MalformedResponseError):
        await _stream(_backend(wrong_type))


async def test_non_streaming_infer_with_and_without_usage() -> None:
    def with_usage(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    def without_usage(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi there"}}]})

    resp = await _backend(with_usage).infer(backend_request())
    assert resp.text == "hi there" and resp.usage.source is TokenCountSource.REPORTED
    est = await _backend(without_usage).infer(backend_request())
    assert est.usage.source is TokenCountSource.ESTIMATED


async def test_health_models_and_version() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/health":
            return httpx.Response(200)
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen"}]})
        if req.url.path == "/version":
            return httpx.Response(200, json={"version": "0.33.0"})
        return httpx.Response(404)

    backend = _backend(handler, VLLMBackend)
    assert (await backend.health()).healthy
    assert await backend.models() == ["qwen"]
    assert await backend.version() == "0.33.0"
    assert backend.capabilities.prefix_caching and backend.capabilities.speculative_decoding


async def test_unhealthy_on_503_or_connection_error() -> None:
    def down(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    assert not (await _backend(down).health()).healthy


VLLM_METRICS = """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="q"} 3.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="q"} 1.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="q"} 0.25
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total{model_name="q"} 1000.0
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{model_name="q"} 250.0
# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_bucket{le="+Inf",model_name="q"} 4.0
vllm:request_queue_time_seconds_count{model_name="q"} 4.0
vllm:request_queue_time_seconds_sum{model_name="q"} 0.4
"""


async def test_vllm_metric_mapping_and_phase_means() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=VLLM_METRICS)

    m = await _backend(handler, VLLMBackend).metrics()
    assert m.running_requests == 3 and m.waiting_requests == 1 and m.kv_cache_usage == 0.25
    assert m.derived_prefix_hit_rate == pytest.approx(0.25)
    assert m.spec_decode_acceptance_rate is None  # not exposed: never fabricated
    before = parse_prometheus_text(VLLM_METRICS)
    after = dict(before)
    after["vllm:request_queue_time_seconds_count"] += 2
    after["vllm:request_queue_time_seconds_sum"] += 0.5
    means = engine_phase_means(VLLMBackend.phase_metrics, before, after)
    assert means == {"queue": pytest.approx(0.25)}


async def test_legacy_vllm_names_and_sglang_mapping() -> None:
    legacy = "# TYPE vllm:gpu_cache_usage_perc gauge\nvllm:gpu_cache_usage_perc 0.5\n"
    sgl = (
        "# TYPE sglang:num_running_reqs gauge\nsglang:num_running_reqs 7\n"
        "# TYPE sglang:cache_hit_rate gauge\nsglang:cache_hit_rate 0.4\n"
        "# TYPE sglang:spec_accept_rate gauge\nsglang:spec_accept_rate 0.6\n"
    )
    v = await _backend(lambda r: httpx.Response(200, text=legacy), VLLMBackend).metrics()
    assert v.kv_cache_usage == 0.5
    s = await _backend(lambda r: httpx.Response(200, text=sgl), SGLangBackend).metrics()
    assert s.running_requests == 7 and s.derived_prefix_hit_rate == 0.4
    assert s.spec_decode_acceptance_rate == 0.6


async def test_trtllm_uses_prometheus_path_and_unavailable_metrics() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        return httpx.Response(404)

    backend = _backend(handler, TensorRTLLMBackend)
    m = await backend.metrics()
    assert paths == ["/prometheus/metrics"] and m.source == "unavailable"
    no_metrics = _backend(handler, metrics_path=None)
    assert (await no_metrics.metrics()).source == "unavailable"


def test_api_key_comes_from_named_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_ENGINE_KEY", "k" * 20)
    backend = OpenAICompatibleBackend(
        OpenAIBackendConfig(name="b", base_url="http://x", api_key_env="TEST_ENGINE_KEY")
    )
    assert backend._client.headers["authorization"] == "Bearer " + "k" * 20
    monkeypatch.delenv("TEST_ENGINE_KEY")
    with pytest.raises(ValueError, match="not set"):
        OpenAICompatibleBackend(
            OpenAIBackendConfig(name="b", base_url="http://x", api_key_env="TEST_ENGINE_KEY")
        )


def test_base_url_validation() -> None:
    import pydantic

    for bad in ("ftp://x", "http://user:pass@host", "nohost"):
        with pytest.raises(pydantic.ValidationError):
            OpenAIBackendConfig(name="b", base_url=bad)
    assert OpenAIBackendConfig(name="b", base_url="http://h:1/").base_url == "http://h:1"
