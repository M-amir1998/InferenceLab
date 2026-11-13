"""Gateway FastAPI app in-process (ASGI transport): endpoints, errors, auth, SSE format."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from inferencelab.gateway.app import create_app
from inferencelab.gateway.config import GatewayConfig

pytestmark = pytest.mark.integration


def gateway(**overrides: Any) -> httpx.AsyncClient:
    data: dict[str, Any] = {
        "backends": [
            {
                "type": "mock",
                "name": "m1",
                "models": ["demo-model"],
                "latency": {"ttft_ms": 1, "tpot_ms": 1},
            },
            {
                "type": "mock",
                "name": "m2",
                "models": ["demo-model"],
                "latency": {"ttft_ms": 1, "tpot_ms": 1},
            },
        ],
        "health": {"enabled": False},
    }
    data.update(overrides)
    app = create_app(GatewayConfig.model_validate(data))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


CHAT = {
    "model": "demo-model",
    "messages": [{"role": "user", "content": "hello there"}],
    "max_tokens": 5,
}


async def test_health_ready_models_metrics_and_backends() -> None:
    async with gateway() as client:
        assert (await client.get("/health")).json() == {"status": "ok"}
        assert (await client.get("/ready")).status_code == 200
        models = (await client.get("/v1/models")).json()
        assert [m["id"] for m in models["data"]] == ["demo-model"]
        await client.post("/v1/chat/completions", json=CHAT)
        metrics = (await client.get("/metrics")).text
        assert (
            "inferencelab_requests_total" in metrics
            and "inferencelab_ttft_seconds_bucket" in metrics
        )
        backends = (await client.get("/inferencelab/backends")).json()
        assert {b["name"] for b in backends["backends"]} == {"m1", "m2"}
        assert backends["queue"] == {"in_flight": 0, "pending": 0}


async def test_non_streaming_chat_and_completion() -> None:
    async with gateway() as client:
        resp = await client.post("/v1/chat/completions", json=CHAT, headers={"x-request-id": "abc"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion" and body["id"] == "chatcmpl-abc"
        assert (
            body["usage"]["completion_tokens"] == 5
            and body["choices"][0]["finish_reason"] == "length"
        )
        assert resp.headers["x-request-id"] == "abc" and resp.headers["x-inferencelab-backend"] in {
            "m1",
            "m2",
        }
        assert resp.headers["x-inferencelab-token-source"] == "reported"
        comp = await client.post(
            "/v1/completions", json={"model": "demo-model", "prompt": "a b c", "max_tokens": 3}
        )
        assert comp.json()["object"] == "text_completion"


async def test_streaming_sse_format() -> None:
    async with gateway() as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={**CHAT, "stream": True, "stream_options": {"include_usage": True}},
        )
        assert resp.status_code == 200 and resp.headers["content-type"].startswith(
            "text/event-stream"
        )
        events = [line[6:] for line in resp.text.splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        content = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]
        )
        assert len(content.split()) == 5
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 5
        finishes = [
            c["choices"][0]["finish_reason"]
            for c in chunks
            if c["choices"] and c["choices"][0]["finish_reason"]
        ]
        assert finishes == ["length"]


async def test_validation_errors_use_openai_envelope() -> None:
    async with gateway() as client:
        bad = await client.post(
            "/v1/chat/completions", json={"model": "demo-model", "messages": []}
        )
        assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_request"
        garbage = await client.post(
            "/v1/chat/completions",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        assert garbage.status_code == 400
        unknown = await client.post("/v1/chat/completions", json={**CHAT, "model": "nope"})
        assert (
            unknown.status_code == 503 and unknown.json()["error"]["code"] == "no_healthy_backend"
        )
        capped = await client.post("/v1/chat/completions", json={**CHAT, "max_tokens": 100_000})
        assert capped.status_code == 400


async def test_backend_failure_maps_to_502() -> None:
    backends = [
        {"type": "mock", "name": "bad", "models": ["demo-model"], "failure_probability": 1.0}
    ]
    async with gateway(backends=backends, retry={"max_attempts": 1}) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT)
        assert resp.status_code == 502 and resp.json()["error"]["type"] == "server_error"


async def test_auth_required_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "k" * 24
    monkeypatch.setenv("TEST_GW_KEYS", f"team-a:{key}")
    async with gateway(auth={"enabled": True, "api_keys_env": "TEST_GW_KEYS"}) as client:
        assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 401
        assert (await client.get("/v1/models")).status_code == 401
        ok = await client.post(
            "/v1/chat/completions", json=CHAT, headers={"authorization": f"Bearer {key}"}
        )
        assert ok.status_code == 200
        assert (await client.get("/health")).status_code == 200  # liveness stays open


async def test_rate_limit_returns_429_with_retry_after() -> None:
    tenants = {"tenants": {"t": {"requests_per_second": 0.5, "burst": 1}}}
    async with gateway(tenants=tenants) as client:
        headers = {"x-tenant-id": "t"}
        assert (
            await client.post("/v1/chat/completions", json=CHAT, headers=headers)
        ).status_code == 200
        limited = await client.post("/v1/chat/completions", json=CHAT, headers=headers)
        assert limited.status_code == 429 and limited.headers["retry-after"] == "2"
        assert limited.json()["error"]["code"] == "rate_limited"


async def test_request_body_limit() -> None:
    async with gateway(limits={"max_request_bytes": 2048}) as client:
        big = {**CHAT, "messages": [{"role": "user", "content": "x" * 5000}]}
        assert (await client.post("/v1/chat/completions", json=big)).status_code == 413


async def test_ready_is_503_without_healthy_backends() -> None:
    from inferencelab.gateway.runtime import GatewayRuntime

    cfg = GatewayConfig.model_validate(
        {"backends": [{"type": "mock", "name": "a"}], "health": {"enabled": False}}
    )
    rt = GatewayRuntime(cfg)
    for handle in rt.pool.handles:
        handle.healthy = False
    app = create_app(cfg, runtime=rt)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    ) as client:
        assert (await client.get("/ready")).status_code == 503


async def _aiter(data: bytes) -> AsyncIterator[bytes]:
    yield data
