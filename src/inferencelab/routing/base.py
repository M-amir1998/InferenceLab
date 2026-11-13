"""Routing contracts.

A router *ranks* candidate backends; it never performs I/O. The data plane walks the
ranking for the first attempt and for retries, so fallback order falls out of the same
decision. Routers see immutable :class:`BackendView` snapshots — never prompt text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from inferencelab.backends.base import BackendCapabilities
from inferencelab.domain.request import Priority, RoutingHints

__all__ = ["BackendView", "Router", "RoutingContext", "RoutingDecision"]


@dataclass(frozen=True, slots=True)
class BackendView:
    name: str
    tier: int = 0
    healthy: bool = True
    circuit_open: bool = False
    outstanding: int = 0
    max_outstanding: int | None = None
    ewma_ttft_s: float | None = None
    ewma_tpot_s: float | None = None
    peak_latency_s: float | None = None
    cost_per_1m_tokens_usd: float | None = None
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)

    @property
    def routable(self) -> bool:
        return self.healthy and not self.circuit_open

    @property
    def has_capacity(self) -> bool:
        return self.max_outstanding is None or self.outstanding < self.max_outstanding

    @property
    def utilization(self) -> float:
        if self.max_outstanding is None:
            return 0.0
        return self.outstanding / self.max_outstanding


@dataclass(frozen=True, slots=True)
class RoutingContext:
    request_id: str
    model: str
    priority: Priority = Priority.NORMAL
    hints: RoutingHints = field(default_factory=RoutingHints)
    estimated_input_tokens: int = 0
    max_tokens: int = 0
    ttft_budget_s: float | None = None


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    ranking: tuple[str, ...]
    policy: str
    reason: str = ""
    slo_at_risk: bool = False

    @property
    def primary(self) -> str | None:
        return self.ranking[0] if self.ranking else None


@runtime_checkable
class Router(Protocol):
    name: str

    def rank(self, ctx: RoutingContext, candidates: Sequence[BackendView]) -> RoutingDecision:
        """Return *every* candidate exactly once, most preferred first."""
        ...
