from __future__ import annotations

import pydantic
import pytest

from inferencelab.backends.simulated.engine import (
    BlockManager,
    EngineRejectedError,
    SimulatedEngine,
    SimulatedEngineConfig,
    StepResult,
)


def _run(
    engine: SimulatedEngine, now: float = 0.0, max_steps: int = 10_000
) -> tuple[list[StepResult], float]:
    steps: list[StepResult] = []
    while engine.has_work() and len(steps) < max_steps:
        res = engine.step(now)
        assert res is not None
        steps.append(res)
        now += res.duration
        engine.blocks.check_invariants()
    return steps, now


def _ids(n: int, offset: int = 0) -> list[int]:
    return [offset + i for i in range(n)]


def test_config_validation() -> None:
    with pytest.raises(pydantic.ValidationError, match="max_num_seqs"):
        SimulatedEngineConfig(max_num_seqs=64, max_num_batched_tokens=32)
    with pytest.raises(pydantic.ValidationError, match="chunked prefill"):
        SimulatedEngineConfig(
            chunked_prefill=False, max_num_batched_tokens=1024, max_model_len=2048
        )
    with pytest.raises(pydantic.ValidationError, match="KV cache"):
        SimulatedEngineConfig(num_kv_blocks=4, block_size=16, max_model_len=4096)


def test_single_request_timing_matches_cost_model() -> None:
    cfg = SimulatedEngineConfig(
        step_overhead_s=0.01,
        prefill_s_per_token=0.001,
        decode_s_per_seq=0.0,
        decode_s_per_kv_token=0.0,
    )
    engine = SimulatedEngine(cfg)
    engine.submit("a", _ids(100), max_tokens=5, arrival_time=0.0)
    steps, end = _run(engine)
    assert len(steps) == 5  # 1 prefill step (emits token 1) + 4 decode steps
    assert steps[0].prefill_tokens == 100 and steps[0].emitted[0].first
    assert steps[0].duration == pytest.approx(0.01 + 0.1)
    timings = next(t for s in steps for _, t in s.finished)
    assert timings.output_tokens == 5 and timings.prompt_tokens == 100
    assert timings.prefill_s == pytest.approx(0.11) and timings.queue_s == 0.0
    assert end == pytest.approx(0.11 + 4 * 0.01)
    assert engine.blocks.used_blocks == 0


def test_prefix_cache_hits_for_repeated_prompt() -> None:
    engine = SimulatedEngine(SimulatedEngineConfig(block_size=16))
    prompt = _ids(160)
    engine.submit("a", prompt, max_tokens=2, arrival_time=0.0)
    _run(engine)
    engine.submit("b", prompt, max_tokens=2, arrival_time=1.0)
    steps, _ = _run(engine, now=1.0)
    timings = next(t for s in steps for _, t in s.finished)
    assert timings.cached_tokens == 159  # all blocks hit; the last token is recomputed
    assert steps[0].prefill_tokens == 1
    assert engine.stats.prefix_cache_hits == 159


def test_prefix_caching_disabled_means_no_hits() -> None:
    engine = SimulatedEngine(SimulatedEngineConfig(prefix_caching=False))
    for name, t in (("a", 0.0), ("b", 1.0)):
        engine.submit(name, _ids(64), max_tokens=1, arrival_time=t)
        _run(engine, now=t)
    assert engine.stats.prefix_cache_hits == 0


def test_generated_tokens_are_cached_for_multi_turn_reuse() -> None:
    engine = SimulatedEngine(SimulatedEngineConfig(block_size=4))
    engine.submit("turn1", _ids(8), max_tokens=9, arrival_time=0.0)
    seq = engine._by_id["turn1"]
    _steps, _ = _run(engine)
    output = seq.output_ids
    # turn 2 resends prompt + generated text + new user tokens
    engine.submit(
        "turn2", [*_ids(8), *output, *_ids(5, offset=500)], max_tokens=1, arrival_time=5.0
    )
    steps2, _ = _run(engine, now=5.0)
    cached = next(t for s in steps2 for _, t in s.finished).cached_tokens
    assert cached >= 12  # prompt blocks + full blocks of the generated answer


def test_chunked_prefill_bounds_tokens_per_step_and_keeps_decoding() -> None:
    cfg = SimulatedEngineConfig(max_num_batched_tokens=256, max_num_seqs=8, chunked_prefill=True)
    engine = SimulatedEngine(cfg)
    engine.submit("short", _ids(16), max_tokens=50, arrival_time=0.0)
    _steps, _ = _run(engine, max_steps=3)
    engine.submit("long", _ids(2000, offset=10_000), max_tokens=2, arrival_time=0.0)
    decode_stalls = 0
    while engine.has_work():
        res = engine.step(0.0)
        assert res is not None
        assert res.prefill_tokens + res.decode_seqs <= 256
        short_active = engine._by_id.get("short") is not None
        if short_active and res.decode_seqs == 0:
            decode_stalls += 1
    assert decode_stalls == 0  # decodes are co-scheduled with every prefill chunk


def test_prefill_first_scheduling_stalls_decodes() -> None:
    cfg = SimulatedEngineConfig(
        max_num_batched_tokens=8192, max_model_len=8192, chunked_prefill=False
    )
    engine = SimulatedEngine(cfg)
    engine.submit("short", _ids(16), max_tokens=20, arrival_time=0.0)
    for _ in range(3):
        engine.step(0.0)
    engine.submit("long", _ids(4000, offset=10_000), max_tokens=2, arrival_time=0.0)
    res = engine.step(0.0)
    assert res is not None and res.decode_seqs == 0 and res.prefill_tokens > 3000


def test_max_num_seqs_queues_requests() -> None:
    engine = SimulatedEngine(SimulatedEngineConfig(max_num_seqs=2, max_num_batched_tokens=64))
    for i in range(5):
        engine.submit(f"r{i}", _ids(8, offset=i * 100), max_tokens=3, arrival_time=0.0)
    first = engine.step(0.0)
    assert first is not None and first.num_running == 2 and first.num_waiting == 3
    steps, _ = _run(engine)
    finished = [sid for s in [first, *steps] for sid, _ in s.finished]
    assert sorted(finished) == [f"r{i}" for i in range(5)]


def test_kv_pressure_preempts_and_still_completes_every_request() -> None:
    cfg = SimulatedEngineConfig(
        block_size=4, num_kv_blocks=32, max_num_seqs=8, max_num_batched_tokens=64, max_model_len=128
    )
    engine = SimulatedEngine(cfg)
    for i in range(6):
        engine.submit(f"r{i}", _ids(12, offset=i * 1000), max_tokens=24, arrival_time=0.0)
    steps, _ = _run(engine)
    finished = {sid: t for s in steps for sid, t in s.finished}
    assert set(finished) == {f"r{i}" for i in range(6)}
    assert all(t.output_tokens == 24 for t in finished.values())
    assert engine.stats.preemptions > 0
    assert engine.blocks.used_blocks == 0


def test_abort_frees_blocks_and_queue() -> None:
    engine = SimulatedEngine(SimulatedEngineConfig(max_num_seqs=1, max_num_batched_tokens=64))
    engine.submit("a", _ids(40), max_tokens=10, arrival_time=0.0)
    engine.submit("b", _ids(40, offset=99), max_tokens=10, arrival_time=0.0)
    engine.step(0.0)
    assert engine.abort("a") and engine.abort("b")
    assert not engine.abort("missing")
    assert not engine.has_work() and engine.blocks.used_blocks == 0
    assert engine.stats.aborted == 2


def test_rejects_requests_that_can_never_fit() -> None:
    engine = SimulatedEngine(SimulatedEngineConfig(max_model_len=128))
    with pytest.raises(EngineRejectedError):
        engine.submit("a", _ids(100), max_tokens=100, arrival_time=0.0)
    with pytest.raises(EngineRejectedError):
        engine.submit("b", [], max_tokens=1, arrival_time=0.0)
    engine.submit("c", _ids(4), max_tokens=1, arrival_time=0.0)
    with pytest.raises(ValueError, match="duplicate"):
        engine.submit("c", _ids(4), max_tokens=1, arrival_time=0.0)


def test_block_manager_refcounts_shared_blocks() -> None:
    bm = BlockManager(num_blocks=8, block_size=4, enable_caching=True)
    hits, private = bm.allocate([11, 12], total_blocks=3)  # type: ignore[misc]
    assert hits == [] and private == 3
    bm.promote(11)
    bm.promote(12)
    bm.check_invariants()
    hits2, private2 = bm.allocate([11, 12], total_blocks=2)  # type: ignore[misc]
    assert hits2 == [11, 12] and private2 == 0
    assert bm.used_blocks == 3  # shared blocks counted once
    bm.free([11, 12], 1)
    bm.free([11, 12], 0)
    bm.check_invariants()
    assert bm.used_blocks == 0 and bm.cached_blocks == 2
    assert bm.allocate([], total_blocks=9) is None
