"""Load-balancing policies. Each is small enough to explain in one sentence.

* RoundRobin         — rotate through candidates.
* Random             — uniform choice (seeded, reproducible).
* LeastOutstanding   — fewest in-flight requests; ties rotate.
* LatencyAware       — lowest peak-EWMA TTFT × (outstanding + 1) ("peak EWMA", as in
                       Finagle/Linkerd); backends without samples are tried first.
* CostAware          — cheapest backend that still has spare capacity, then spill over.
* SLOAware           — cheapest backend whose *predicted* TTFT meets the request's TTFT
                       budget; if none can, the fastest one (flagged ``slo_at_risk``).
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence

from inferencelab.routing.base import BackendView, RoutingContext, RoutingDecision

__all__ = [
    "CostAwareRouter",
    "LatencyAwareRouter",
    "LeastOutstandingRouter",
    "RandomRouter",
    "RoundRobinRouter",
    "SLOAwareRouter",
    "predicted_ttft_s",
]


def _stable(candidates: Sequence[BackendView]) -> list[BackendView]:
    return sorted(candidates, key=lambda v: v.name)


class _Rotation:
    """Rotating tie-breaker so equal candidates share load instead of the first winning."""

    def __init__(self) -> None:
        self._counter = itertools.count()

    def offsets(self, n: int) -> list[int]:
        start = next(self._counter) % n if n else 0
        return [(i - start) % n for i in range(n)]


class RoundRobinRouter:
    name = "round_robin"

    def __init__(self) -> None:
        self._rot = _Rotation()

    def rank(self, ctx: RoutingContext, candidates: Sequence[BackendView]) -> RoutingDecision:
        ordered = _stable(candidates)
        offsets = self._rot.offsets(len(ordered))
        ranking = [
            v.name for _, v in sorted(zip(offsets, ordered, strict=True), key=lambda p: p[0])
        ]
        return RoutingDecision(tuple(ranking), self.name)


class RandomRouter:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def rank(self, ctx: RoutingContext, candidates: Sequence[BackendView]) -> RoutingDecision:
        names = [v.name for v in _stable(candidates)]
        self._rng.shuffle(names)
        return RoutingDecision(tuple(names), self.name)


class LeastOutstandingRouter:
    name = "least_outstanding"

    def __init__(self) -> None:
        self._rot = _Rotation()

    def rank(self, ctx: RoutingContext, candidates: Sequence[BackendView]) -> RoutingDecision:
        ordered = _stable(candidates)
        tie = self._rot.offsets(len(ordered))
        keyed = sorted(
            zip(ordered, tie, strict=True), key=lambda p: (p[0].outstanding, p[0].utilization, p[1])
        )
        return RoutingDecision(tuple(v.name for v, _ in keyed), self.name)


class LatencyAwareRouter:
    name = "latency_aware"

    def __init__(self) -> None:
        self._rot = _Rotation()

    def rank(self, ctx: RoutingContext, candidates: Sequence[BackendView]) -> RoutingDecision:
        ordered = _stable(candidates)
        tie = self._rot.offsets(len(ordered))

        def cost(view: BackendView) -> float:
            if view.peak_latency_s is None:
                return -1.0  # explore unmeasured backends first
            return view.peak_latency_s * (view.outstanding + 1)

        keyed = sorted(zip(ordered, tie, strict=True), key=lambda p: (cost(p[0]), p[1]))
        return RoutingDecision(tuple(v.name for v, _ in keyed), self.name)


class CostAwareRouter:
    name = "cost_aware"

    def __init__(self) -> None:
        self._rot = _Rotation()

    def rank(self, ctx: RoutingContext, candidates: Sequence[BackendView]) -> RoutingDecision:
        ordered = _stable(candidates)
        tie = self._rot.offsets(len(ordered))

        def key(pair: tuple[BackendView, int]) -> tuple[int, float, float, int]:
            view, t = pair
            price = view.cost_per_1m_tokens_usd
            return (
                0 if view.has_capacity else 1,
                math.inf if price is None else price,
                view.utilization,
                t,
            )

        keyed = sorted(zip(ordered, tie, strict=True), key=key)
        spill = bool(keyed) and not keyed[0][0].has_capacity
        return RoutingDecision(
            tuple(v.name for v, _ in keyed), self.name, reason="all_full" if spill else ""
        )


def predicted_ttft_s(view: BackendView) -> float | None:
    """TTFT estimate under current load: observed EWMA inflated by utilisation."""
    if view.ewma_ttft_s is None:
        return None
    capacity = view.max_outstanding or max(1, view.outstanding + 1)
    return view.ewma_ttft_s * (1.0 + view.outstanding / capacity)


class SLOAwareRouter:
    name = "slo_aware"

    def __init__(self, default_ttft_budget_s: float | None = None) -> None:
        self.default_ttft_budget_s = default_ttft_budget_s
        self._fallback = LatencyAwareRouter()

    def rank(self, ctx: RoutingContext, candidates: Sequence[BackendView]) -> RoutingDecision:
        budget = ctx.ttft_budget_s if ctx.ttft_budget_s is not None else self.default_ttft_budget_s
        if budget is None:
            decision = self._fallback.rank(ctx, candidates)
            return RoutingDecision(decision.ranking, self.name, reason="no_budget")
        meets: list[tuple[float, float, str]] = []
        misses: list[tuple[float, str]] = []
        for view in _stable(candidates):
            pred = predicted_ttft_s(view)
            price = view.cost_per_1m_tokens_usd
            if pred is None or pred <= budget:  # no data yet: optimistic, to learn
                meets.append((math.inf if price is None else price, pred or 0.0, view.name))
            else:
                misses.append((pred, view.name))
        meets.sort()
        misses.sort()
        ranking = tuple(n for *_, n in meets) + tuple(n for _, n in misses)
        at_risk = not meets and bool(misses)
        return RoutingDecision(
            ranking, self.name, reason="none_meet_slo" if at_risk else "", slo_at_risk=at_risk
        )
