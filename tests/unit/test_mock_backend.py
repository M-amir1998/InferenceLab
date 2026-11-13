from __future__ import annotations

import pytest

from inferencelab.backends.base import BackendChunk
from inferencelab.backends.config import FaultMode, FaultWindow, LatencySpec, MockBackendConfig
from inferencelab.backends.mock.backend import MockBackend
from inferencelab.clock import VirtualClock
from inferencelab.domain.errors import (
    BackendHTTPError,
    BackendUnavailableError,
    ErrorKind,
    MalformedResponseError,
    StreamInterruptedError,
)
from inferencelab.domain.response import FinishReason
from tests.conftest import backend_request


async def _drain(
    backend: MockBackend, clock: VirtualClock, rid: str = "r1", max_tokens: int = 8
) -> tuple[list[BackendChunk], float]:
    async def go() -> list[BackendChunk]:
        return [c async for c in backend.stream(backend_request(rid, max_tokens=max_tokens))]

    t0 = clock.now()
    chunks = await clock.run(go())
    return chunks, clock.now() - t0


async def test_latency_and_token_counts_are_exact(vclock: VirtualClock) -> None:
    cfg = MockBackendConfig(name="m", latency=LatencySpec(ttft_ms=50, tpot_ms=10))
    chunks, elapsed = await _drain(MockBackend(cfg, clock=vclock), vclock, max_tokens=5)
    assert elapsed == pytest.approx(0.05 + 4 * 0.01)
    content = [c for c in chunks if c.text]
    assert len(content) == 5
    final = chunks[-1]
    assert final.finish_reason is FinishReason.LENGTH
    assert (
        final.usage is not None
        and final.usage.output_tokens == 5
        and final.usage.input_tokens == 10
    )


async def test_output_tokens_cap_and_stop_reason(vclock: VirtualClock) -> None:
    cfg = MockBackendConfig(name="m", output_tokens=3)
    chunks, _ = await _drain(MockBackend(cfg, clock=vclock), vclock, max_tokens=10)
    assert chunks[-1].finish_reason is FinishReason.STOP
    assert sum(1 for c in chunks if c.text) == 3


async def test_same_request_id_is_deterministic(vclock: VirtualClock) -> None:
    cfg = MockBackendConfig(
        name="m", latency=LatencySpec(ttft_ms=50, tpot_ms=10, jitter=0.5), seed=3
    )
    a, ta = await _drain(MockBackend(cfg, clock=vclock), vclock, rid="same")
    b, tb = await _drain(MockBackend(cfg, clock=vclock), vclock, rid="same")
    assert [c.text for c in a] == [c.text for c in b] and ta == pytest.approx(tb)


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("failure_probability", BackendHTTPError),
        ("disconnect_probability", StreamInterruptedError),
        ("malformed_probability", MalformedResponseError),
    ],
)
async def test_injected_faults(vclock: VirtualClock, field: str, error: type[Exception]) -> None:
    cfg = MockBackendConfig.model_validate({"name": "m", field: 1.0})
    with pytest.raises(error):
        await _drain(MockBackend(cfg, clock=vclock), vclock)


async def test_timeout_fault_hangs_until_cancelled(vclock: VirtualClock) -> None:
    backend = MockBackend(MockBackendConfig(name="m", timeout_probability=1.0), clock=vclock)

    async def go() -> None:
        async with vclock.timeout(2.0):
            async for _ in backend.stream(backend_request()):
                pass

    with pytest.raises(TimeoutError):
        await vclock.run(go())


async def test_fault_windows_and_runtime_injection(vclock: VirtualClock) -> None:
    window = FaultWindow(mode=FaultMode.DOWN, start_s=1.0, end_s=2.0)
    backend = MockBackend(MockBackendConfig(name="m", faults=[window]), clock=vclock)
    assert (await backend.health()).healthy
    vclock.advance(1.5)
    assert not (await backend.health()).healthy
    with pytest.raises(BackendUnavailableError):
        await _drain(backend, vclock)
    vclock.advance(1.0)
    assert (await backend.health()).healthy
    backend.inject_fault(FaultWindow(mode=FaultMode.ERROR, end_s=5.0))
    with pytest.raises(BackendHTTPError) as exc:
        await _drain(backend, vclock)
    assert exc.value.kind is ErrorKind.BACKEND_ERROR
    backend.inject_fault(None)
    chunks, _ = await _drain(backend, vclock)
    assert chunks[-1].usage is not None


async def test_capacity_limit_returns_503(vclock: VirtualClock) -> None:
    backend = MockBackend(MockBackendConfig(name="m", max_concurrency=1), clock=vclock)
    first = backend.stream(backend_request("a"))

    async def go() -> None:
        await anext(first)  # holds the only slot
        with pytest.raises(BackendHTTPError) as exc:
            await anext(backend.stream(backend_request("b")))
        assert exc.value.backend_status == 503
        await first.aclose()  # type: ignore[attr-defined]

    await vclock.run(go())
    assert (await backend.metrics()).running_requests == 0


async def test_slow_fault_multiplies_latency(vclock: VirtualClock) -> None:
    cfg = MockBackendConfig(
        name="m",
        latency=LatencySpec(ttft_ms=10, tpot_ms=10),
        faults=[FaultWindow(mode=FaultMode.SLOW, slow_factor=4.0)],
    )
    _, elapsed = await _drain(MockBackend(cfg, clock=vclock), vclock, max_tokens=2)
    assert elapsed == pytest.approx((0.01 + 0.01) * 4)
