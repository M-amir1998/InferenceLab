"""Mock/simulated serving apps over real loopback HTTP, driven by the OpenAI HTTP adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from inferencelab.backends.config import (
    MockBackendConfig,
    OpenAIBackendConfig,
    SimulatedBackendConfig,
)
from inferencelab.backends.mock.backend import MockBackend
from inferencelab.backends.openai_compatible.backend import OpenAICompatibleBackend
from inferencelab.backends.server import create_backend_app
from inferencelab.backends.simulated.backend import SimulatedInferenceBackend
from inferencelab.domain.errors import (
    BackendHTTPError,
    MalformedResponseError,
    StreamInterruptedError,
)
from inferencelab.domain.request import ChatMessage
from inferencelab.domain.response import TokenCountSource
from inferencelab.net import LoopbackServer
from tests.conftest import backend_request

pytestmark = pytest.mark.e2e


@asynccontextmanager
async def served_mock(
    **cfg: object,
) -> AsyncIterator[tuple[LoopbackServer, OpenAICompatibleBackend]]:
    config = MockBackendConfig.model_validate(
        {"name": "mock", "models": ["demo-model"], "latency": {"ttft_ms": 5, "tpot_ms": 2}, **cfg}
    )
    async with LoopbackServer(
        create_backend_app(MockBackend(config), models=["demo-model"], kind="mock", admin=True)
    ) as server:
        adapter = OpenAICompatibleBackend(
            OpenAIBackendConfig(name="adapter", base_url=server.url, models=["demo-model"])
        )
        try:
            yield server, adapter
        finally:
            await adapter.aclose()


async def test_stream_and_infer_over_http() -> None:
    async with served_mock() as (_, adapter):
        chunks = [c async for c in adapter.stream(backend_request(max_tokens=7))]
        assert sum(1 for c in chunks if c.text) == 7
        usage = chunks[-1].usage
        assert (
            usage is not None
            and usage.source is TokenCountSource.REPORTED
            and usage.output_tokens == 7
        )
        resp = await adapter.infer(backend_request(max_tokens=3))
        assert resp.usage.output_tokens == 3 and len(resp.text.split()) == 3


async def test_control_endpoints() -> None:
    async with served_mock() as (server, adapter):
        assert (await adapter.health()).healthy
        assert await adapter.models() == ["demo-model"]
        assert await adapter.version() is not None
        assert (await adapter.metrics()).source.startswith("prometheus")
        async with httpx.AsyncClient(base_url=server.url) as client:
            missing = await client.post(
                "/v1/chat/completions",
                json={"model": "other", "messages": [{"role": "user", "content": "x"}]},
            )
            assert missing.status_code == 404


async def test_runtime_fault_injection_round_trip() -> None:
    async with (
        served_mock() as (server, adapter),
        httpx.AsyncClient(base_url=server.url) as client,
    ):
        assert (await client.post("/admin/fault", json={"mode": "error"})).status_code == 200
        with pytest.raises(BackendHTTPError) as exc:
            [c async for c in adapter.stream(backend_request())]
        assert exc.value.backend_status == 500
        await client.post("/admin/fault", json={"clear": True})
        assert [c async for c in adapter.stream(backend_request(max_tokens=2))]
        assert (await client.post("/admin/fault", json={"mode": "nope"})).status_code == 400


async def test_malformed_sse_is_detected_by_the_client() -> None:
    async with served_mock(malformed_probability=1.0, output_tokens=4) as (_, adapter):
        with pytest.raises(MalformedResponseError):
            [c async for c in adapter.stream(backend_request(max_tokens=4))]


async def test_mid_stream_disconnect_is_detected_by_the_client() -> None:
    async with served_mock(disconnect_probability=1.0) as (_, adapter):
        with pytest.raises(StreamInterruptedError):
            [c async for c in adapter.stream(backend_request(max_tokens=20))]


async def test_simulated_engine_metrics_are_exposed_and_mapped() -> None:
    backend = SimulatedInferenceBackend(SimulatedBackendConfig(name="sim", models=["demo-model"]))
    async with LoopbackServer(
        create_backend_app(backend, models=["demo-model"], kind="simulated")
    ) as server:
        adapter = OpenAICompatibleBackend(
            OpenAIBackendConfig(name="a", base_url=server.url, models=["demo-model"])
        )
        history = (ChatMessage("system", " ".join(["data"] * 64)), ChatMessage("user", "q"))
        for i in range(2):
            [
                c
                async for c in adapter.stream(
                    backend_request(f"r{i}", messages=history, max_tokens=2)
                )
            ]
        m = await adapter.metrics()
        assert m.prefix_cache_queries and m.prefix_cache_hits and m.prefix_cache_hits > 0
        assert m.running_requests == 0 and m.generation_tokens_total == 4
        await adapter.aclose()
