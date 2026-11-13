"""Prefix/session affinity via consistent hashing with bounded loads.

Requests carrying the same affinity key (``prefix_key``, ``conversation_id`` or
``tenant_id``) prefer the same backend, so the engine's own prefix/KV cache is reused.
InferenceLab never sees or stores the cache; it only routes by identifiers.

* **Rendezvous (HRW) hashing** gives every key a stable preference order over backends;
  adding or removing a backend only remaps the keys that pointed at it.
* **Bounded loads** (Mirrokni et al., 2018): a backend is skipped for affinity if it
  already carries more than ``load_factor`` × the mean load, so a hot key cannot pile
  every request onto one replica. Skipped requests fall through to the inner policy.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Literal

from inferencelab.routing.base import BackendView, Router, RoutingContext, RoutingDecision

__all__ = ["AffinityKey", "PrefixAffinityRouter", "rendezvous_order"]

AffinityKey = Literal["prefix_key", "conversation_id", "tenant_id"]


def _score(key: str, backend: str) -> int:
    digest = hashlib.blake2b(f"{key}\x00{backend}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def rendezvous_order(key: str, names: Sequence[str]) -> list[str]:
    """Highest-random-weight order of ``names`` for ``key`` (deterministic)."""
    return sorted(names, key=lambda n: (-_score(key, n), n))


class PrefixAffinityRouter:
    def __init__(
        self,
        inner: Router,
        *,
        keys: Sequence[AffinityKey] = ("prefix_key", "conversation_id"),
        load_factor: float = 1.25,
    ) -> None:
        if load_factor < 1.0:
            raise ValueError("load_factor must be >= 1")
        self.inner = inner
        self.keys = tuple(keys)
        self.load_factor = load_factor
        self.name = f"prefix_affinity+{inner.name}"

    def affinity_key(self, ctx: RoutingContext) -> str | None:
        for field in self.keys:
            value = getattr(ctx.hints, field)
            if value:
                return f"{field}:{value}"
        return None

    def rank(self, ctx: RoutingContext, candidates: Sequence[BackendView]) -> RoutingDecision:
        key = self.affinity_key(ctx)
        if key is None or not candidates:
            inner = self.inner.rank(ctx, candidates)
            return RoutingDecision(inner.ranking, self.name, "no_key", inner.slo_at_risk)
        by_name = {v.name: v for v in candidates}
        mean = (sum(v.outstanding for v in candidates) + 1) / len(candidates)
        bound = math.ceil(self.load_factor * mean)
        preferred = None
        for name in rendezvous_order(key, list(by_name)):
            view = by_name[name]
            if view.outstanding + 1 <= bound and view.has_capacity:
                preferred = name
                break
        rest = [v for v in candidates if v.name != preferred]
        inner = self.inner.rank(ctx, rest)
        if preferred is None:
            return RoutingDecision(
                inner.ranking, self.name, "affinity_overloaded", inner.slo_at_risk
            )
        return RoutingDecision((preferred, *inner.ranking), self.name, "affinity")
