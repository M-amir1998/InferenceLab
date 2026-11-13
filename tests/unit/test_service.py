"""InferenceService (gateway data plane) under virtual time with mock backends."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from inferencelab.backends.config import FaultMode, FaultWindow
from inferencelab.backends.mock.backend import MockBackend
from inferencelab.clock import VirtualClock
from inferencelab.domain.errors import (
    CircuitOpenError,
    ErrorKind,
    InferenceError,
    NoHealthyBackendError,
    QueueFullError,
    RateLimitedError,
)
from inferencelab.domain.request import RoutingHints
from inferencelab.gateway.config import GatewayConfig
from inferencelab.gateway.runtime import GatewayRuntime
from inferencelab.reliability.circuit_breaker import CircuitState
from tests.conftest import chat_request


def runtime(clock: VirtualClock, backends: list[dict[str, Any]], **cfg: Any) -> GatewayRuntime:
    data: dict[str, Any] = {
        "backends": [{"type": "mock", "models": ["demo-model"], **b} for b in backends],
        "health": {"enabled": False},
        "routing": {"policy": "round_robin"},
        "retry": {"max_attempts": 3, "backoff_initial_s": 0.01, "jitter": 0.0},
    }
    data.update(cfg)
    return GatewayRuntime(GatewayConfig.model_validate(data), clock=clock)


def metric(rt: GatewayRuntime, name: str, **labels: str) -> float:
    value = rt.metrics.registry.get_sample_value(name, labels)
    return 0.0 if value is None else value


async def test_complete_success_and_accounting(vclock: VirtualClock) -> None:
    rt = runtime(vclock, [{"name": "a", "latency": {"ttft_ms": 50, "tpot_ms": 10}}])
    resp = await vclock.run(rt.service.complete(chat_request(max_tokens=5)))
    assert resp.backend == "a" and resp.attempts == 1
    assert resp.usage.output_tokens == 5 and len(resp.text.split()) == 5
    assert resp.timings.ttft_s == pytest.approx(0.05) and resp.timings.e2e_s == pytest.approx(0.09)
    assert (
        metric(rt, "inferencelab_requests_total", model="demo-model", backend="a", status="ok") == 1
    )
    assert rt.admission.in_flight == 0
    assert rt.pool.get("a").stats.outstanding == 0  # type: ignore[union-attr]


async def test_retry_on_other_backend_before_first_token(vclock: VirtualClock) -> None:
    rt = runtime(vclock, [{"name": "bad", "failure_probability": 1.0}, {"name": "good"}])

    async def main() -> list[str]:
        out = []
        for i in range(4):
            gs = await rt.service.open_stream(chat_request(f"r{i}", stream=True))
            async for _ in gs.chunks():
                pass
            out.append(f"{gs.backend}:{gs.attempts}")
        return out

    results = await vclock.run(main())
    assert all(r.startswith("good") for r in results)
    assert any(r.endswith(":2") for r in results)  # some requests needed a retry
    assert metric(rt, "inferencelab_retries_total", backend="good", reason="backend_error") >= 1


async def test_no_retry_after_streaming_started(vclock: VirtualClock) -> None:
    rt = runtime(vclock, [{"name": "cut", "disconnect_probability": 1.0}, {"name": "good"}])

    async def main() -> InferenceError:
        gs = await rt.service.open_stream(chat_request(stream=True, max_tokens=20))
        assert gs.backend == "cut"
        with pytest.raises(InferenceError) as exc:
            async for _ in gs.chunks():
                pass
        return exc.value

    err = await vclock.run(main())
    assert err.kind is ErrorKind.STREAM_INTERRUPTED
    assert (
        metric(rt, "inferencelab_requests_total", model="demo-model", backend="cut", status="error")
        == 1
    )


async def test_non_streaming_retries_mid_generation_failure(vclock: VirtualClock) -> None:
    rt = runtime(vclock, [{"name": "cut", "disconnect_probability": 1.0}, {"name": "good"}])
    resp = await vclock.run(rt.service.complete(chat_request(max_tokens=20)))
    assert resp.backend == "good" and resp.attempts == 2


async def test_first_token_timeout_falls_back(vclock: VirtualClock) -> None:
    rt = runtime(
        vclock,
        [{"name": "hang", "timeout_probability": 1.0}, {"name": "good"}],
        timeouts={"first_token_s": 1.0, "idle_s": 5.0, "overall_s": 30.0},
    )
    # Round robin tries "good" first for one request and "hang" first for the other.
    first = await vclock.run(rt.service.complete(chat_request("a")))
    second = await vclock.run(rt.service.complete(chat_request("b")))
    assert {first.backend, second.backend} == {"good"}
    assert {first.attempts, second.attempts} == {1, 2}
    assert metric(rt, "inferencelab_errors_total", backend="hang", kind="timeout_first_token") == 1


async def test_overall_deadline_bounds_total_time(vclock: VirtualClock) -> None:
    rt = runtime(
        vclock,
        [{"name": "hang", "timeout_probability": 1.0}],
        timeouts={"first_token_s": 10.0, "idle_s": 10.0, "overall_s": 2.0},
    )
    with pytest.raises(InferenceError) as exc:
        await vclock.run(rt.service.complete(chat_request()))
    assert exc.value.kind is ErrorKind.TIMEOUT_OVERALL
    assert vclock.now() == pytest.approx(2.0, abs=0.05)


async def test_idle_timeout_mid_stream(vclock: VirtualClock) -> None:
    rt = runtime(
        vclock,
        [{"name": "slow", "latency": {"ttft_ms": 10, "tpot_ms": 5000}}],
        timeouts={"first_token_s": 5.0, "idle_s": 1.0, "overall_s": 60.0},
    )

    async def main() -> ErrorKind:
        gs = await rt.service.open_stream(chat_request(stream=True))
        with pytest.raises(InferenceError) as exc:
            async for _ in gs.chunks():
                pass
        return exc.value.kind

    assert await vclock.run(main()) is ErrorKind.TIMEOUT_IDLE


async def test_circuit_opens_and_traffic_avoids_backend(vclock: VirtualClock) -> None:
    rt = runtime(
        vclock,
        [{"name": "bad", "failure_probability": 1.0}, {"name": "good"}],
        circuit_breaker={
            "failure_threshold": 2,
            "open_duration_s": 60,
            "failure_rate_threshold": None,
        },
    )

    async def main() -> None:
        for i in range(10):
            await rt.service.complete(chat_request(f"r{i}"))

    await vclock.run(main())
    bad = rt.pool.get("bad")
    assert bad is not None and bad.breaker.state is CircuitState.OPEN
    assert bad.stats.failures == 2  # no traffic after the circuit opened
    assert metric(rt, "inferencelab_circuit_transitions_total", backend="bad", to_state="open") == 1


async def test_all_circuits_open_rejects_fast(vclock: VirtualClock) -> None:
    rt = runtime(
        vclock,
        [{"name": "bad", "failure_probability": 1.0}],
        circuit_breaker={"failure_threshold": 1, "open_duration_s": 60},
        retry={"max_attempts": 1},
    )
    with pytest.raises(InferenceError):
        await vclock.run(rt.service.complete(chat_request("r0")))
    with pytest.raises(NoHealthyBackendError):  # circuit-open backends are not routable
        await vclock.run(rt.service.complete(chat_request("r1")))


async def test_tiered_fallback_only_when_primary_unroutable(vclock: VirtualClock) -> None:
    rt = runtime(
        vclock,
        [{"name": "primary", "routing": {"tier": 0}}, {"name": "fallback", "routing": {"tier": 1}}],
    )
    first = await vclock.run(rt.service.complete(chat_request("a")))
    assert first.backend == "primary"
    primary = rt.pool.get("primary")
    assert primary is not None
    primary.healthy = False
    second = await vclock.run(rt.service.complete(chat_request("b")))
    assert second.backend == "fallback"


async def test_unknown_model_is_rejected(vclock: VirtualClock) -> None:
    rt = runtime(vclock, [{"name": "a"}])
    with pytest.raises(NoHealthyBackendError, match="no backend serves"):
        await vclock.run(rt.service.complete(chat_request(model="other")))


async def test_queue_overflow_and_metrics(vclock: VirtualClock) -> None:
    rt = runtime(
        vclock,
        [{"name": "a", "latency": {"ttft_ms": 1000, "tpot_ms": 10}}],
        queue={"max_concurrency": 1, "max_pending_requests": 1},
    )

    async def main() -> list[str]:
        tasks = [
            asyncio.ensure_future(rt.service.complete(chat_request(f"r{i}"))) for i in range(3)
        ]
        out = []
        for t in tasks:
            try:
                await t
                out.append("ok")
            except QueueFullError:
                out.append("queue_full")
        return out

    assert sorted(await vclock.run(main())) == ["ok", "ok", "queue_full"]
    assert metric(rt, "inferencelab_rejections_total", reason="queue_full") == 1


async def test_tenant_rate_limit(vclock: VirtualClock) -> None:
    rt = runtime(
        vclock, [{"name": "a"}], tenants={"tenants": {"t": {"requests_per_second": 1, "burst": 1}}}
    )
    hints = RoutingHints(tenant_id="t")
    await vclock.run(rt.service.complete(chat_request("a", hints=hints)))
    with pytest.raises(RateLimitedError):
        await vclock.run(rt.service.complete(chat_request("b", hints=hints)))
    assert rt.admission.in_flight == 0


async def test_client_cancellation_releases_everything(vclock: VirtualClock) -> None:
    rt = runtime(vclock, [{"name": "a", "latency": {"ttft_ms": 10, "tpot_ms": 100}}])

    async def main() -> None:
        gs = await rt.service.open_stream(chat_request(stream=True, max_tokens=50))
        chunks = gs.chunks()
        await anext(chunks)
        await chunks.aclose()
        await gs.aclose()

    await vclock.run(main())
    handle = rt.pool.get("a")
    assert (
        handle is not None
        and handle.stats.outstanding == 0
        and handle.breaker.state is CircuitState.CLOSED
    )
    assert rt.admission.in_flight == 0
    assert (
        metric(
            rt, "inferencelab_requests_total", model="demo-model", backend="a", status="cancelled"
        )
        == 1
    )


async def test_health_checks_mark_backend_down(vclock: VirtualClock) -> None:
    rt = runtime(vclock, [{"name": "a"}], health={"enabled": True, "unhealthy_threshold": 1})
    backend = rt.pool.get("a")
    assert backend is not None and isinstance(backend.backend, MockBackend)
    backend.backend.inject_fault(FaultWindow(mode=FaultMode.DOWN))
    await vclock.run(rt.pool.check_all())
    assert not backend.healthy and not rt.pool.any_ready()
    assert metric(rt, "inferencelab_backend_healthy", backend="a") == 0


async def test_runtime_rejects_unknown_overrides(vclock: VirtualClock) -> None:
    with pytest.raises(ValueError, match="overrides"):
        GatewayRuntime(
            GatewayConfig.model_validate({"backends": [{"type": "mock", "name": "a"}]}),
            clock=vclock,
            backends={"zzz": MockBackend.__new__(MockBackend)},
        )


def test_circuit_open_error_is_503() -> None:
    assert CircuitOpenError("x").http_status == 503
