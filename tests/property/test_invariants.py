"""Property-based tests (Hypothesis) for core invariants."""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import date

from hypothesis import assume, given
from hypothesis import strategies as st

from inferencelab.backends.simulated.engine import BlockManager
from inferencelab.clock import VirtualClock
from inferencelab.cost.model import compute_cost
from inferencelab.domain.errors import RateLimitedError
from inferencelab.domain.hardware import PricingSpec
from inferencelab.domain.request import RoutingHints
from inferencelab.domain.result import Phase, RequestRecord, RequestStatus
from inferencelab.domain.slo import RequestThresholds, SLOSpec, is_request_compliant
from inferencelab.domain.workload import ConstantRateArrival, PoissonArrival, WorkloadSpec
from inferencelab.limits.tenant import TenantLimiter, TenantLimitsConfig, TenantQuota
from inferencelab.limits.token_bucket import TokenBucket
from inferencelab.metrics.stats import Distribution, percentile
from inferencelab.metrics.summary import summarize_records
from inferencelab.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from inferencelab.routing.affinity import PrefixAffinityRouter
from inferencelab.routing.base import BackendView, Router, RoutingContext
from inferencelab.routing.factory import rank_tiered
from inferencelab.routing.policies import (
    CostAwareRouter,
    LatencyAwareRouter,
    LeastOutstandingRouter,
    RandomRouter,
    RoundRobinRouter,
    SLOAwareRouter,
)
from inferencelab.workloads.arrivals import arrival_offsets

floats = st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------- rate limiting
@given(
    capacity=st.floats(min_value=0.5, max_value=100),
    rate=st.floats(min_value=0.0, max_value=50),
    ops=st.lists(
        st.tuples(
            st.sampled_from(["advance", "consume", "refund"]), st.floats(min_value=0, max_value=20)
        ),
        max_size=60,
    ),
)
def test_token_bucket_never_exceeds_capacity_or_goes_negative(
    capacity: float, rate: float, ops: list[tuple[str, float]]
) -> None:
    clock = VirtualClock()
    bucket = TokenBucket(capacity, rate, clock)
    for op, amount in ops:
        before = bucket.tokens
        if op == "advance":
            clock.advance(amount)
        elif op == "consume":
            ok = bucket.try_consume(amount)
            assert ok == (amount <= before + 1e-9)
        else:
            bucket.refund(amount)
        assert -1e-9 <= bucket.tokens <= capacity + 1e-9


@given(elapsed=st.floats(min_value=0, max_value=100), rate=st.floats(min_value=0.1, max_value=10))
def test_token_bucket_refill_is_linear_until_full(elapsed: float, rate: float) -> None:
    clock = VirtualClock()
    bucket = TokenBucket(1000.0, rate, clock, initial=0.0)
    clock.advance(elapsed)
    assert abs(bucket.tokens - min(1000.0, elapsed * rate)) < 1e-6


@given(st.lists(st.sampled_from(["a", "b"]), min_size=1, max_size=40))
def test_tenant_isolation(order: list[str]) -> None:
    clock = VirtualClock()
    cfg = TenantLimitsConfig(
        tenants={
            "a": TenantQuota(requests_per_second=1, burst=1),
            "b": TenantQuota(requests_per_second=1, burst=5),
        }
    )
    lim = TenantLimiter(cfg, clock)
    admitted = {"a": 0, "b": 0}
    for tenant in order:
        try:
            lim.acquire(tenant, 1).release()
        except RateLimitedError:
            continue
        admitted[tenant] += 1
    assert admitted["a"] == min(1, order.count("a"))
    assert admitted["b"] == min(5, order.count("b"))  # a's exhaustion never affects b


# ---------------------------------------------------------------- metrics
@given(
    st.lists(floats, min_size=1, max_size=200),
    st.floats(min_value=0, max_value=100),
    st.floats(min_value=0, max_value=100),
)
def test_percentile_bounds_and_monotonicity(values: list[float], q1: float, q2: float) -> None:
    lo, hi = sorted((q1, q2))
    p_lo, p_hi = percentile(values, lo), percentile(values, hi)
    assert min(values) - 1e-6 <= p_lo <= p_hi + 1e-6 <= max(values) + 2e-6


@given(st.lists(st.floats(min_value=0, max_value=1e3, allow_nan=False), min_size=0, max_size=600))
def test_distribution_percentiles_are_ordered_and_sized(values: list[float]) -> None:
    d = Distribution.of(values)
    present = [v for v in (d.p(50), d.p(90), d.p(95), d.p(99)) if v is not None]
    assert present == sorted(present)
    assert (d.p(99) is not None) == (len(values) >= 500)
    if values:
        assert (
            d.min is not None
            and d.max is not None
            and d.min <= (d.mean or 0) + 1e-9 <= d.max + 2e-9
        )


@given(
    st.lists(
        st.tuples(
            st.floats(0, 10),
            st.floats(0.001, 2),
            st.integers(1, 50),
            st.sampled_from(list(RequestStatus)),
        ),
        min_size=1,
        max_size=80,
    ),
    st.floats(1, 3000),
)
def test_goodput_never_exceeds_throughput(
    rows: list[tuple[float, float, int, RequestStatus]], ttft_ms: float
) -> None:
    records = []
    for i, (sent, ttft, tokens, status) in enumerate(rows):
        first = sent + ttft
        records.append(
            RequestRecord(
                index=i,
                request_id=str(i),
                phase=Phase.MEASURE,
                status=status,
                scheduled_at=sent,
                sent_at=None if status is RequestStatus.DROPPED else sent,
                first_token_at=first if status is RequestStatus.OK else None,
                completed_at=first + 0.01 * tokens,
                chunk_times=tuple(first + 0.01 * k for k in range(tokens))
                if status is RequestStatus.OK
                else (),
                output_tokens=tokens if status is RequestStatus.OK else None,
            )
        )
    s = summarize_records(
        records, run_id="p", simulated=True, slo=SLOSpec.model_validate({"ttft_p50_ms": ttft_ms})
    )
    c = s.counts
    assert c.ok + c.error + c.timeout + c.cancelled + c.rejected + c.dropped == c.measured
    if s.throughput.goodput_rps is not None:
        assert s.throughput.goodput_rps <= s.throughput.request_throughput + 1e-9
    assert s.slo is None or s.slo.attainment is None or 0.0 <= s.slo.attainment <= 1.0


# ---------------------------------------------------------------- SLO
opt = st.one_of(st.none(), st.floats(min_value=1, max_value=1e4))


@given(opt, opt, opt, st.floats(0, 10), st.floats(0, 1), st.floats(0, 20), st.floats(1.0, 3.0))
def test_slo_classification_is_monotonic_in_thresholds(
    ttft_ms: float | None,
    tpot_ms: float | None,
    e2e_ms: float | None,
    ttft: float,
    tpot: float,
    e2e: float,
    scale: float,
) -> None:
    strict = RequestThresholds(ttft_ms=ttft_ms, tpot_ms=tpot_ms, e2e_ms=e2e_ms)
    loose = RequestThresholds(
        ttft_ms=None if ttft_ms is None else ttft_ms * scale,
        tpot_ms=None if tpot_ms is None else tpot_ms * scale,
        e2e_ms=None if e2e_ms is None else e2e_ms * scale,
    )
    if is_request_compliant(strict, ttft_s=ttft, tpot_s=tpot, e2e_s=e2e):
        assert is_request_compliant(loose, ttft_s=ttft, tpot_s=tpot, e2e_s=e2e)


@given(
    st.sampled_from(["ttft", "tpot", "itl", "e2e"]),
    st.sampled_from([50, 90, 95, 99]),
    st.floats(1, 1e5),
)
def test_slo_compact_keys_roundtrip(metric: str, pct: int, threshold: float) -> None:
    spec = SLOSpec.model_validate({f"{metric}_p{pct}_ms": threshold})
    assert len(spec.objectives) == 1
    again = SLOSpec.model_validate(spec.model_dump())
    assert again == spec


# ---------------------------------------------------------------- cost
@given(
    st.floats(0, 100),
    st.floats(0, 1e6),
    st.integers(1, 16),
    st.integers(0, 10**9),
    st.integers(1, 10**6),
)
def test_cost_is_linear_and_consistent(
    price: float, duration: float, gpus: int, out_tokens: int, ok: int
) -> None:
    pricing = PricingSpec(
        hourly_cost_usd=price,
        unit="per_gpu_hour",
        price_source="manual",
        recorded_at=date(2026, 1, 1),
    )
    report = compute_cost(
        pricing,
        gpu_count=gpus,
        duration_s=duration,
        input_tokens=0,
        output_tokens=out_tokens,
        successful_requests=ok,
        slo_compliant_requests=None,
    )
    assert report.total_cost_usd >= 0
    assert abs(report.total_cost_usd - price * gpus * duration / 3600) <= 1e-6 * max(
        1.0, report.total_cost_usd
    )
    if out_tokens:
        assert report.cost_per_1m_output_tokens_usd is not None
        assert abs(
            report.cost_per_1m_output_tokens_usd * out_tokens / 1e6 - report.total_cost_usd
        ) <= 1e-6 * max(1.0, report.total_cost_usd)
    double = compute_cost(
        pricing,
        gpu_count=gpus,
        duration_s=2 * duration,
        input_tokens=0,
        output_tokens=out_tokens,
        successful_requests=ok,
        slo_compliant_requests=None,
    )
    assert abs(double.total_cost_usd - 2 * report.total_cost_usd) <= 1e-6 * max(
        1.0, double.total_cost_usd
    )


# ---------------------------------------------------------------- routing
view_strategy = st.builds(
    BackendView,
    name=st.text(alphabet="abcdefgh", min_size=1, max_size=4),
    tier=st.integers(0, 2),
    healthy=st.booleans(),
    circuit_open=st.booleans(),
    outstanding=st.integers(0, 50),
    max_outstanding=st.one_of(st.none(), st.integers(1, 50)),
    ewma_ttft_s=st.one_of(st.none(), st.floats(0.001, 5)),
    peak_latency_s=st.one_of(st.none(), st.floats(0.001, 5)),
    cost_per_1m_tokens_usd=st.one_of(st.none(), st.floats(0, 10)),
)
ROUTERS: list[Callable[[], Router]] = [
    RoundRobinRouter,
    LeastOutstandingRouter,
    LatencyAwareRouter,
    CostAwareRouter,
    lambda: RandomRouter(1),
    lambda: SLOAwareRouter(0.5),
]


@given(
    st.lists(view_strategy, min_size=1, max_size=8, unique_by=lambda v: v.name),
    st.integers(0, len(ROUTERS) - 1),
    st.one_of(st.none(), st.text(min_size=1, max_size=8)),
)
def test_routers_return_a_permutation_of_candidates(
    cands: list[BackendView], idx: int, key: str | None
) -> None:
    router = ROUTERS[idx]()
    ctx = RoutingContext(
        request_id="r", model="m", hints=RoutingHints(conversation_id=key), ttft_budget_s=0.5
    )
    decision = router.rank(ctx, cands)
    assert sorted(decision.ranking) == sorted(v.name for v in cands)
    wrapped = PrefixAffinityRouter(router, keys=("conversation_id",)).rank(ctx, cands)
    assert sorted(wrapped.ranking) == sorted(v.name for v in cands)
    tiered = rank_tiered(router, ctx, cands)
    assert set(tiered.ranking) == {v.name for v in cands if v.routable}
    tiers = [next(v.tier for v in cands if v.name == n) for n in tiered.ranking]
    assert tiers == sorted(tiers)  # preferred tier first


@given(
    st.lists(view_strategy, min_size=2, max_size=6, unique_by=lambda v: v.name),
    st.text(min_size=1, max_size=10),
)
def test_affinity_is_deterministic(cands: list[BackendView], key: str) -> None:
    ctx = RoutingContext(request_id="r", model="m", hints=RoutingHints(prefix_key=key))
    a = PrefixAffinityRouter(LeastOutstandingRouter()).rank(ctx, cands).primary
    b = PrefixAffinityRouter(LeastOutstandingRouter()).rank(ctx, cands).primary
    assert a == b


# ---------------------------------------------------------------- arrivals / config round trips
@given(st.floats(0.1, 100), st.floats(0.5, 30), st.integers(0, 1000))
def test_arrival_offsets_sorted_and_bounded(rate: float, duration: float, seed: int) -> None:
    for pattern in (ConstantRateArrival(rate=rate), PoissonArrival(rate=rate)):
        offsets = list(arrival_offsets(pattern, random.Random(seed), duration_s=duration))
        assert offsets == sorted(offsets) and all(0 <= t < duration for t in offsets)


@given(st.integers(1, 10_000), st.integers(1, 4096), st.integers(1, 4096), st.integers(1, 64))
def test_workload_spec_roundtrip(n: int, lo: int, width: int, conc: int) -> None:
    spec = WorkloadSpec.model_validate(
        {
            "name": "w",
            "num_requests": n,
            "input_tokens": {"kind": "uniform", "min": lo, "max": lo + width},
            "arrival": {"kind": "closed_loop", "concurrency": conc},
        }
    )
    assert WorkloadSpec.model_validate(spec.model_dump()) == spec


# ---------------------------------------------------------------- engine / reliability state machines
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["alloc", "free", "promote"]), st.integers(0, 6), st.integers(1, 6)
        ),
        max_size=80,
    )
)
def test_block_manager_invariants(ops: list[tuple[str, int, int]]) -> None:
    bm = BlockManager(num_blocks=16, block_size=4, enable_caching=True)
    held: list[tuple[list[int], int]] = []
    for op, seed, size in ops:
        if op == "alloc":
            hashes = [hash((seed, i)) for i in range(size)]
            res = bm.allocate(hashes[: size // 2], total_blocks=size)
            if res is not None:
                held.append(res)
        elif op == "promote" and held:
            hits, private = held[-1]
            if private:
                h = hash(("p", seed, len(hits)))
                bm.promote(h)
                held[-1] = ([*hits, h], private - 1)
        elif op == "free" and held:
            hits, private = held.pop(seed % len(held))
            bm.free(hits, private)
        bm.check_invariants()
    for hits, private in held:
        bm.free(hits, private)
    bm.check_invariants()
    assert bm.used_blocks == 0


@given(
    st.lists(
        st.tuples(st.sampled_from(["ok", "fail", "ignore", "tick", "hold"]), st.floats(0, 20)),
        max_size=80,
    )
)
def test_circuit_breaker_state_machine_invariants(events: list[tuple[str, float]]) -> None:
    clock = VirtualClock()
    cb = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=3, open_duration_s=5, half_open_max_calls=2), clock
    )
    held = []
    for kind, dt in events:
        if kind == "tick":
            clock.advance(dt)
            continue
        state = cb.state
        permit = cb.acquire()
        if state is CircuitState.OPEN:
            assert permit is None
        if permit is None:
            continue
        if kind == "hold":
            held.append(permit)
            assert cb._probes_in_flight <= cb.config.half_open_max_calls
            continue
        {"ok": permit.success, "fail": permit.failure, "ignore": permit.ignore}[kind]()
        assert cb.state in set(CircuitState)
    for permit in held:
        permit.ignore()
    assume(True)
