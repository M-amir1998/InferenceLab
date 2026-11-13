"""Full chain over loopback HTTP: client -> gateway -> router -> engines -> stream -> metrics."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from inferencelab.backends.config import SimulatedBackendConfig
from inferencelab.backends.server import create_backend_app
from inferencelab.backends.simulated.backend import SimulatedInferenceBackend
from inferencelab.benchmarking.config import BenchmarkConfig, HTTPTargetConfig
from inferencelab.benchmarking.runner import run_benchmark
from inferencelab.domain.workload import WorkloadSpec
from inferencelab.gateway.app import create_app
from inferencelab.gateway.config import GatewayConfig
from inferencelab.net import LoopbackServer

pytestmark = pytest.mark.e2e


@asynccontextmanager
async def deployment(
    policy: str = "least_outstanding",
) -> AsyncIterator[tuple[str, list[SimulatedInferenceBackend]]]:
    engines = [
        SimulatedInferenceBackend(SimulatedBackendConfig(name=f"sim-{i}", models=["demo-model"]))
        for i in range(2)
    ]
    servers = [
        LoopbackServer(create_backend_app(e, models=["demo-model"], kind="simulated"))
        for e in engines
    ]
    for s in servers:
        await s.start()
    cfg = GatewayConfig.model_validate(
        {
            "backends": [
                {
                    "type": "openai",
                    "name": e.name,
                    "base_url": s.url,
                    "models": ["demo-model"],
                    "deployment": e.name,
                }
                for e, s in zip(engines, servers, strict=True)
            ],
            # load_factor 100 disables bounded-load spill so stickiness is deterministic here;
            # the spill behaviour itself is unit-tested in tests/unit/test_routing.py.
            "routing": {
                "policy": policy,
                "affinity": {"enabled": True, "keys": ["conversation_id"], "load_factor": 100},
            },
            "health": {"interval_s": 0.5},
            "slo": {"ttft_p95_ms": 2000},
        }
    )
    gateway = LoopbackServer(create_app(cfg))
    await gateway.start()
    try:
        yield gateway.url, engines
    finally:
        await gateway.stop()
        for s in servers:
            await s.stop()


async def test_benchmark_through_gateway_over_http(tmp_path: Path) -> None:
    async with deployment() as (url, _):
        workload = WorkloadSpec.model_validate(
            {
                "name": "e2e-chat",
                "model": "demo-model",
                "num_requests": 24,
                "sessions": {"turns": {"kind": "fixed", "value": 3}},
                "input_tokens": {"kind": "uniform", "min": 16, "max": 64},
                "output_tokens": {"kind": "fixed", "value": 8},
                "arrival": {"kind": "closed_loop", "concurrency": 4},
            }
        )
        config = BenchmarkConfig(
            name="e2e",
            target=HTTPTargetConfig(base_url=url),
            workload=workload,
            output_dir=str(tmp_path),
        )
        result = await run_benchmark(config, output_root=tmp_path, plots=False)
        async with httpx.AsyncClient(base_url=url) as client:
            metrics = (await client.get("/metrics")).text
    s = result.summary
    # The gateway's backends are generic OpenAI endpoints: never labelled as GPU results.
    assert s.counts.ok == 24 and s.label == "UNVERIFIED BACKEND TYPE" and s.simulated is None
    assert set(s.per_backend) <= {"sim-0", "sim-1"}
    assert all(r.gateway_queue_s is not None and r.attempts == 1 for r in result.records)
    # session affinity: every turn of a conversation lands on the same engine
    by_session: dict[str | None, set[str | None]] = {}
    for r in result.records:
        by_session.setdefault(r.session_id, set()).add(r.backend)
    assert all(len(b) == 1 for b in by_session.values())
    assert s.tokens.cached_input_ratio is not None and s.tokens.cached_input_ratio > 0.3
    assert (
        'inferencelab_requests_total{backend="sim-0",model="demo-model",status="ok"}' in metrics
        or "sim-1" in metrics
    )


async def test_client_disconnect_aborts_engine_work() -> None:
    async with deployment() as (url, engines):
        async with httpx.AsyncClient(base_url=url, timeout=10.0) as client:
            body = {
                "model": "demo-model",
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 4000,
                "stream": True,
            }
            async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
                assert resp.status_code == 200
                lines = resp.aiter_lines()
                for _ in range(6):
                    await anext(lines)
            # the response context closed the connection mid-stream
        for _ in range(100):
            if all(e.engine.num_running == 0 for e in engines):
                break
            await asyncio.sleep(0.05)
        assert all(e.engine.num_running == 0 for e in engines)
        assert sum(e.engine.stats.aborted for e in engines) == 1


async def test_non_streaming_request_and_models_endpoint() -> None:
    async with deployment("round_robin") as (url, _), httpx.AsyncClient(base_url=url) as client:
        models = (await client.get("/v1/models")).json()
        assert [m["id"] for m in models["data"]] == ["demo-model"]
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "demo-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 6,
            },
            headers={"x-conversation-id": "conv-1"},
        )
        assert resp.status_code == 200 and resp.json()["usage"]["completion_tokens"] == 6
        ready = await client.get("/ready")
        assert ready.json() == {"ready": True}
