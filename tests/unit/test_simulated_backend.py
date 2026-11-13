from __future__ import annotations

import asyncio

import pytest

from inferencelab.backends.base import BackendChunk
from inferencelab.backends.config import SimulatedBackendConfig
from inferencelab.backends.simulated.backend import SimulatedInferenceBackend, render_prompt_tokens
from inferencelab.backends.simulated.engine import SimulatedEngineConfig
from inferencelab.clock import VirtualClock
from inferencelab.domain.errors import BackendHTTPError, BackendUnavailableError
from inferencelab.domain.request import ChatMessage
from tests.conftest import backend_request


def _backend(clock: VirtualClock, **engine: object) -> SimulatedInferenceBackend:
    return SimulatedInferenceBackend(
        SimulatedBackendConfig(name="sim", engine=SimulatedEngineConfig.model_validate(engine)),
        clock=clock,
    )


async def _collect(
    backend: SimulatedInferenceBackend, rid: str, **kw: object
) -> list[BackendChunk]:
    return [c async for c in backend.stream(backend_request(rid, **kw))]  # type: ignore[arg-type]


async def test_stream_reports_usage_and_engine_timings(vclock: VirtualClock) -> None:
    backend = _backend(vclock)
    chunks = await vclock.run(_collect(backend, "a", words=40, max_tokens=6))
    content = [c for c in chunks if c.text]
    assert len(content) == 6 and not content[0].text.startswith(" ")
    final = chunks[-1]
    assert final.usage is not None and final.usage.output_tokens == 6
    assert final.usage.input_tokens == len(render_prompt_tokens(backend_request("a", words=40)))
    assert (
        final.engine is not None
        and final.engine.prefill_s is not None
        and final.engine.prefill_s > 0
    )
    await backend.aclose()


async def test_concurrent_requests_batch_together(vclock: VirtualClock) -> None:
    backend = _backend(vclock, step_overhead_s=0.01, decode_s_per_seq=0.0, prefill_s_per_token=0.0)

    async def main() -> float:
        await asyncio.gather(*(_collect(backend, f"r{i}", max_tokens=10) for i in range(8)))
        return vclock.now()

    elapsed = await vclock.run(main())
    # 8 sequences in one batch: ~10 steps of 10 ms, not 80 sequential steps.
    assert elapsed == pytest.approx(0.10, rel=0.15)
    await backend.aclose()


async def test_multi_turn_resend_hits_cache(vclock: VirtualClock) -> None:
    backend = _backend(vclock, block_size=4)
    history = (ChatMessage("system", " ".join(["data"] * 40)), ChatMessage("user", "query one"))

    async def main() -> int | None:
        first = [
            c async for c in backend.stream(backend_request("t1", messages=history, max_tokens=12))
        ]
        answer = "".join(c.text for c in first)
        turn2 = (*history, ChatMessage("assistant", answer), ChatMessage("user", "and then"))
        second = [
            c async for c in backend.stream(backend_request("t2", messages=turn2, max_tokens=4))
        ]
        usage = second[-1].usage
        return None if usage is None else usage.cached_input_tokens

    cached = await vclock.run(main())
    assert cached is not None and cached >= 48
    await backend.aclose()


async def test_closing_the_stream_aborts_the_sequence(vclock: VirtualClock) -> None:
    backend = _backend(vclock)

    async def main() -> None:
        stream = backend.stream(backend_request("a", max_tokens=50))
        await anext(stream)
        await stream.aclose()  # type: ignore[attr-defined]
        await vclock.sleep(0.5)

    await vclock.run(main())
    assert backend.engine.stats.aborted == 1
    assert backend.engine.blocks.used_blocks == 0
    metrics = await backend.metrics()
    assert metrics.running_requests == 0 and metrics.source == "simulator"
    await backend.aclose()


async def test_rejects_oversized_requests(vclock: VirtualClock) -> None:
    backend = _backend(vclock, max_model_len=64, num_kv_blocks=64)
    with pytest.raises(BackendHTTPError) as exc:
        await vclock.run(_collect(backend, "big", words=100, max_tokens=10))
    assert exc.value.backend_status == 400
    await backend.aclose()
    assert not (await backend.health()).healthy
    with pytest.raises(BackendUnavailableError):
        await vclock.run(_collect(backend, "after-close"))


def test_completion_prompt_tokenization() -> None:
    from inferencelab.backends.base import BackendRequest
    from inferencelab.domain.request import RequestKind, SamplingParams

    req = BackendRequest(
        request_id="x",
        model="m",
        kind=RequestKind.COMPLETION,
        sampling=SamplingParams(max_tokens=1),
        prompt="a b c",
    )
    assert len(render_prompt_tokens(req)) == 3
