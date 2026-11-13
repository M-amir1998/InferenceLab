from __future__ import annotations

from collections import Counter

import pytest

from inferencelab.clock import VirtualClock
from inferencelab.domain.request import RoutingHints
from inferencelab.routing.affinity import PrefixAffinityRouter, rendezvous_order
from inferencelab.routing.base import BackendView, RoutingContext
from inferencelab.routing.factory import AffinityConfig, RoutingConfig, build_router, rank_tiered
from inferencelab.routing.policies import (
    CostAwareRouter,
    LatencyAwareRouter,
    LeastOutstandingRouter,
    RandomRouter,
    RoundRobinRouter,
    SLOAwareRouter,
    predicted_ttft_s,
)
from inferencelab.routing.stats import BackendStats, PeakEWMA

CTX = RoutingContext(request_id="r", model="m")


def views(*specs: tuple[str, dict[str, object]]) -> list[BackendView]:
    return [BackendView(name=n, **kw) for n, kw in specs]  # type: ignore[arg-type]


def test_round_robin_rotates_evenly() -> None:
    router = RoundRobinRouter()
    cands = views(("a", {}), ("b", {}), ("c", {}))
    firsts = [router.rank(CTX, cands).primary for _ in range(6)]
    assert firsts == ["a", "b", "c", "a", "b", "c"]
    assert sorted(router.rank(CTX, cands).ranking) == ["a", "b", "c"]


def test_random_is_seeded_and_complete() -> None:
    cands = views(("a", {}), ("b", {}), ("c", {}))
    r1 = [RandomRouter(7).rank(CTX, cands).ranking for _ in range(3)]
    r2 = [RandomRouter(7).rank(CTX, cands).ranking for _ in range(3)]
    assert r1 == r2 and all(sorted(r) == ["a", "b", "c"] for r in r1)
    counts = Counter(RandomRouter(1).rank(CTX, cands).primary for _ in range(1))
    assert sum(counts.values()) == 1


def test_least_outstanding_prefers_idle_and_rotates_ties() -> None:
    router = LeastOutstandingRouter()
    cands = views(("a", {"outstanding": 5}), ("b", {"outstanding": 1}), ("c", {"outstanding": 1}))
    primaries = {router.rank(CTX, cands).primary for _ in range(4)}
    assert primaries == {"b", "c"}
    assert router.rank(CTX, cands).ranking[-1] == "a"


def test_latency_aware_explores_then_prefers_fast() -> None:
    router = LatencyAwareRouter()
    cold = views(("measured", {"peak_latency_s": 0.1}), ("new", {}))
    assert router.rank(CTX, cold).primary == "new"
    warm = views(
        ("slow", {"peak_latency_s": 0.5}), ("fast", {"peak_latency_s": 0.1, "outstanding": 3})
    )
    assert router.rank(CTX, warm).primary == "fast"  # 0.1*4 < 0.5*1
    busy = views(
        ("slow", {"peak_latency_s": 0.5}), ("fast", {"peak_latency_s": 0.1, "outstanding": 9})
    )
    assert router.rank(CTX, busy).primary == "slow"  # 0.1*10 > 0.5


def test_cost_aware_cheapest_with_capacity_then_spill() -> None:
    router = CostAwareRouter()
    cands = views(
        ("cheap", {"cost_per_1m_tokens_usd": 0.1, "max_outstanding": 2, "outstanding": 1}),
        ("pricey", {"cost_per_1m_tokens_usd": 1.0, "max_outstanding": 2}),
        ("unknown", {}),
    )
    assert router.rank(CTX, cands).ranking == ("cheap", "pricey", "unknown")
    full = views(
        ("cheap", {"cost_per_1m_tokens_usd": 0.1, "max_outstanding": 2, "outstanding": 2}),
        ("pricey", {"cost_per_1m_tokens_usd": 1.0, "max_outstanding": 2}),
    )
    assert router.rank(CTX, full).primary == "pricey"
    all_full = views(("a", {"max_outstanding": 1, "outstanding": 1}))
    assert router.rank(CTX, all_full).reason == "all_full"


def test_slo_aware_cheapest_meeting_budget_else_fastest() -> None:
    router = SLOAwareRouter()
    ctx = RoutingContext(request_id="r", model="m", ttft_budget_s=0.3)
    cands = views(
        (
            "fast-expensive",
            {"ewma_ttft_s": 0.1, "cost_per_1m_tokens_usd": 2.0, "max_outstanding": 10},
        ),
        ("ok-cheap", {"ewma_ttft_s": 0.2, "cost_per_1m_tokens_usd": 0.5, "max_outstanding": 10}),
        ("too-slow", {"ewma_ttft_s": 0.9, "cost_per_1m_tokens_usd": 0.1, "max_outstanding": 10}),
    )
    decision = router.rank(ctx, cands)
    assert (
        decision.ranking == ("ok-cheap", "fast-expensive", "too-slow") and not decision.slo_at_risk
    )
    hopeless = views(("a", {"ewma_ttft_s": 1.0}), ("b", {"ewma_ttft_s": 0.8}))
    risky = router.rank(ctx, hopeless)
    assert risky.slo_at_risk and risky.primary == "b"
    assert router.rank(CTX, cands).reason == "no_budget"


def test_predicted_ttft_grows_with_utilisation() -> None:
    assert predicted_ttft_s(BackendView(name="a")) is None
    idle = predicted_ttft_s(BackendView(name="a", ewma_ttft_s=0.1, max_outstanding=10))
    busy = predicted_ttft_s(
        BackendView(name="a", ewma_ttft_s=0.1, max_outstanding=10, outstanding=10)
    )
    assert idle == pytest.approx(0.1) and busy == pytest.approx(0.2)


def test_rendezvous_hashing_is_stable_under_membership_change() -> None:
    names = ["a", "b", "c", "d"]
    keys = [f"conv-{i}" for i in range(400)]
    before = {k: rendezvous_order(k, names)[0] for k in keys}
    after = {k: rendezvous_order(k, [n for n in names if n != "d"])[0] for k in keys}
    moved = [k for k in keys if before[k] != after[k]]
    assert all(before[k] == "d" for k in moved)  # only keys that lived on "d" move
    assert 50 < len(moved) < 150


def test_prefix_affinity_sticks_and_respects_bounded_load() -> None:
    router = PrefixAffinityRouter(
        LeastOutstandingRouter(), keys=("conversation_id",), load_factor=1.25
    )
    ctx = RoutingContext(request_id="r", model="m", hints=RoutingHints(conversation_id="c1"))
    cands = views(("a", {}), ("b", {}), ("c", {}))
    first = router.rank(ctx, cands)
    assert first.reason == "affinity"
    assert all(router.rank(ctx, cands).primary == first.primary for _ in range(5))
    hot = [
        BackendView(name=v.name, outstanding=12 if v.name == first.primary else 0) for v in cands
    ]
    spilled = router.rank(ctx, hot)
    assert spilled.primary != first.primary and spilled.reason == "affinity"
    assert router.rank(CTX, cands).reason == "no_key"


def test_tiered_ranking_prefers_lower_tier_and_skips_unroutable() -> None:
    router = RoundRobinRouter()
    cands = views(
        ("fallback", {"tier": 1}),
        ("primary-down", {"tier": 0, "healthy": False}),
        ("primary-open", {"tier": 0, "circuit_open": True}),
        ("primary", {"tier": 0}),
    )
    decision = rank_tiered(router, CTX, cands)
    assert decision.ranking == ("primary", "fallback")
    only_fallback = rank_tiered(router, CTX, cands[:3])
    assert only_fallback.ranking == ("fallback",)
    assert rank_tiered(router, CTX, cands[1:3]).ranking == ()


def test_build_router_from_config() -> None:
    assert build_router(RoutingConfig(policy="round_robin")).name == "round_robin"
    assert (
        build_router(RoutingConfig(policy="slo_aware", default_ttft_budget_ms=500)).name
        == "slo_aware"
    )
    wrapped = build_router(
        RoutingConfig(policy="cost_aware", affinity=AffinityConfig(enabled=True))
    )
    assert wrapped.name == "prefix_affinity+cost_aware"
    with pytest.raises(ValueError):
        PrefixAffinityRouter(RoundRobinRouter(), load_factor=0.5)


def test_peak_ewma_jumps_up_and_decays_down() -> None:
    clock = VirtualClock()
    peak = PeakEWMA(tau_s=10.0, clock=clock)
    peak.observe(0.1)
    peak.observe(1.0)
    assert peak.value == 1.0  # immediate jump
    clock.advance(10.0)
    peak.observe(0.1)
    assert 0.1 < (peak.value or 0) < 1.0
    with pytest.raises(ValueError):
        PeakEWMA(0, clock)


def test_backend_stats_accounting() -> None:
    stats = BackendStats(VirtualClock(), alpha=0.5)
    stats.on_dispatch()
    stats.on_first_token(0.2)
    stats.on_complete(ok=True, tpot_s=0.02)
    stats.on_dispatch()
    stats.on_complete(ok=None)
    stats.on_dispatch()
    stats.on_complete(ok=False)
    assert (stats.outstanding, stats.successes, stats.failures) == (0, 1, 1)
    assert stats.ewma_ttft_s == 0.2 and stats.ewma_tpot_s == 0.02
    with pytest.raises(ValueError):
        BackendStats(VirtualClock(), alpha=0)
