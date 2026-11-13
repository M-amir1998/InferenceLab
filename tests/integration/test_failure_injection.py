"""Failure injection over real loopback HTTP: gateway service -> HTTP adapter -> mock server.

Covers: backend 500, backend timeout, mid-stream disconnect, slow first token, slow decode,
malformed response, all backends unavailable and queue overflow.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from inferencelab.backends.config import MockBackendConfig
from inferencelab.backends.mock.backend import MockBackend
from inferencelab.backends.server import create_backend_app
from inferencelab.domain.errors import ErrorKind, InferenceError, QueueFullError
from inferencelab.gateway.config import GatewayConfig
from inferencelab.gateway.runtime import GatewayRuntime
from inferencelab.net import LoopbackServer
from tests.conftest import chat_request

pytestmark = pytest.mark.integration


@asynccontextmanager
async def stack(*mocks: dict[str, Any], **gateway: Any) -> AsyncIterator[GatewayRuntime]:
    servers = []
    try:
        for spec in mocks:
            cfg = MockBackendConfig.model_validate(
                {"models": ["demo-model"], "latency": {"ttft_ms": 5, "tpot_ms": 2}, **spec}
            )
            server = LoopbackServer(
                create_backend_app(MockBackend(cfg), models=["demo-model"], kind="mock")
            )
            await server.start()
            servers.append((cfg.name, server))
        data: dict[str, Any] = {
            "backends": [
                {"type": "openai", "name": name, "base_url": srv.url, "models": ["demo-model"]}
                for name, srv in servers
            ],
            "health": {"enabled": False},
            "routing": {"policy": "round_robin"},
            "retry": {"max_attempts": 3, "backoff_initial_s": 0.0},
            "timeouts": {"first_token_s": 1.0, "idle_s": 0.5, "overall_s": 10.0},
        }
        data.update(gateway)
        rt = GatewayRuntime(GatewayConfig.model_validate(data))
        try:
            yield rt
        finally:
            await rt.stop()
    finally:
        for _, srv in servers:
            await srv.stop()


async def stream_all(rt: GatewayRuntime, rid: str = "r", max_tokens: int = 6) -> tuple[str, str]:
    gs = await rt.service.open_stream(chat_request(rid, stream=True, max_tokens=max_tokens))
    text = "".join([c.text async for c in gs.chunks()])
    return gs.backend, text


async def test_backend_500_is_retried_on_healthy_backend() -> None:
    async with stack({"name": "err", "failure_probability": 1.0}, {"name": "ok"}) as rt:
        results = [await stream_all(rt, f"r{i}") for i in range(3)]
    assert all(b == "ok" and len(t.split()) == 6 for b, t in results)


async def test_backend_hang_hits_first_token_timeout_then_falls_back() -> None:
    async with stack({"name": "hang", "timeout_probability": 1.0}, {"name": "ok"}) as rt:
        results = [await stream_all(rt, f"r{i}") for i in range(2)]
    assert {b for b, _ in results} == {"ok"}


async def test_slow_first_token_without_alternative_times_out() -> None:
    async with stack(
        {"name": "slow", "latency": {"ttft_ms": 3000, "tpot_ms": 1}}, retry={"max_attempts": 1}
    ) as rt:
        with pytest.raises(InferenceError) as exc:
            await stream_all(rt)
    assert exc.value.kind is ErrorKind.TIMEOUT_FIRST_TOKEN


async def test_slow_decode_hits_idle_timeout_mid_stream() -> None:
    async with stack({"name": "slow", "latency": {"ttft_ms": 5, "tpot_ms": 2000}}) as rt:
        gs = await rt.service.open_stream(chat_request(stream=True, max_tokens=5))
        with pytest.raises(InferenceError) as exc:
            async for _ in gs.chunks():
                pass
    assert exc.value.kind is ErrorKind.TIMEOUT_IDLE


async def test_mid_stream_disconnect_is_not_retried() -> None:
    async with stack(
        {"name": "cut", "disconnect_probability": 1.0},
        {"name": "ok"},
        routing={"policy": "least_outstanding"},
    ) as rt:
        outcomes = []
        for i in range(4):
            try:
                outcomes.append(await stream_all(rt, f"r{i}", max_tokens=20))
            except InferenceError as exc:
                outcomes.append(("error", exc.kind.value))
    assert ("error", "stream_interrupted") in outcomes  # surfaced, never silently retried
    assert any(b == "ok" for b, _ in outcomes)


async def test_malformed_response_is_retried_before_output() -> None:
    async with stack(
        {"name": "bad", "malformed_probability": 1.0, "output_tokens": 1}, {"name": "ok"}
    ) as rt:
        backend, text = await stream_all(rt)
    assert backend == "ok" and text


async def test_all_backends_unavailable() -> None:
    dead = LoopbackServer(
        create_backend_app(
            MockBackend(MockBackendConfig(name="x")), models=["demo-model"], kind="mock"
        )
    )
    url = dead.url
    dead._sock.close()  # nothing listens on this port
    cfg = GatewayConfig.model_validate(
        {
            "backends": [
                {
                    "type": "openai",
                    "name": "gone",
                    "base_url": url,
                    "models": ["demo-model"],
                    "connect_timeout_s": 1,
                }
            ],
            "health": {"enabled": False},
            "retry": {"max_attempts": 2, "backoff_initial_s": 0.0},
        }
    )
    rt = GatewayRuntime(cfg)
    with pytest.raises(InferenceError) as exc:
        await rt.service.complete(chat_request())
    assert exc.value.kind in (ErrorKind.BACKEND_UNAVAILABLE, ErrorKind.TIMEOUT_CONNECT)
    await rt.stop()


async def test_queue_overflow_rejects_with_503() -> None:
    async with stack(
        {"name": "slow", "latency": {"ttft_ms": 300, "tpot_ms": 1}},
        queue={"max_concurrency": 1, "max_pending_requests": 1},
    ) as rt:
        tasks = [
            asyncio.ensure_future(rt.service.complete(chat_request(f"q{i}", max_tokens=2)))
            for i in range(4)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    rejected = [r for r in results if isinstance(r, QueueFullError)]
    assert len(rejected) == 2 and all(r.http_status == 503 for r in rejected)
