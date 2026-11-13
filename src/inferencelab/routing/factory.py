"""Router configuration and construction, plus tiered fallback ranking."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby
from typing import Literal

from pydantic import Field

from inferencelab.routing.affinity import AffinityKey, PrefixAffinityRouter
from inferencelab.routing.base import BackendView, Router, RoutingContext, RoutingDecision
from inferencelab.routing.policies import (
    CostAwareRouter,
    LatencyAwareRouter,
    LeastOutstandingRouter,
    RandomRouter,
    RoundRobinRouter,
    SLOAwareRouter,
)
from inferencelab.schema import StrictModel

__all__ = ["AffinityConfig", "PolicyName", "RoutingConfig", "build_router", "rank_tiered"]

PolicyName = Literal[
    "round_robin", "random", "least_outstanding", "latency_aware", "cost_aware", "slo_aware"
]


def _default_affinity_keys() -> list[AffinityKey]:
    return ["prefix_key", "conversation_id"]


class AffinityConfig(StrictModel):
    enabled: bool = False
    keys: list[AffinityKey] = Field(default_factory=_default_affinity_keys)
    load_factor: float = Field(default=1.25, ge=1.0)


class RoutingConfig(StrictModel):
    policy: PolicyName = "least_outstanding"
    affinity: AffinityConfig = Field(default_factory=AffinityConfig)
    seed: int = 0
    ewma_alpha: float = Field(default=0.3, gt=0, le=1)
    peak_ewma_decay_s: float = Field(default=10.0, gt=0)
    default_ttft_budget_ms: float | None = Field(default=None, gt=0)


def build_router(config: RoutingConfig) -> Router:
    policy: Router
    if config.policy == "round_robin":
        policy = RoundRobinRouter()
    elif config.policy == "random":
        policy = RandomRouter(config.seed)
    elif config.policy == "least_outstanding":
        policy = LeastOutstandingRouter()
    elif config.policy == "latency_aware":
        policy = LatencyAwareRouter()
    elif config.policy == "cost_aware":
        policy = CostAwareRouter()
    else:
        budget = (
            None if config.default_ttft_budget_ms is None else config.default_ttft_budget_ms / 1000
        )
        policy = SLOAwareRouter(budget)
    if config.affinity.enabled:
        return PrefixAffinityRouter(
            policy, keys=config.affinity.keys, load_factor=config.affinity.load_factor
        )
    return policy


def rank_tiered(
    router: Router, ctx: RoutingContext, views: Sequence[BackendView]
) -> RoutingDecision:
    """Rank routable backends tier by tier: preferred → secondary → fallback.

    Within a tier the policy decides; a lower tier is only reached after every routable
    backend of the tiers above it (unhealthy or circuit-open backends are excluded).
    """
    routable = sorted((v for v in views if v.routable), key=lambda v: v.tier)
    ranking: list[str] = []
    reasons: list[str] = []
    at_risk = False
    policy = router.name
    for tier, group in groupby(routable, key=lambda v: v.tier):
        decision = router.rank(ctx, list(group))
        ranking.extend(decision.ranking)
        policy = decision.policy
        if decision.reason:
            reasons.append(f"t{tier}:{decision.reason}")
        at_risk = at_risk or (tier == routable[0].tier and decision.slo_at_risk)
    return RoutingDecision(tuple(ranking), policy, ",".join(reasons), at_risk)
