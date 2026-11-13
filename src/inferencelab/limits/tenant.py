"""Per-tenant quotas: requests/second, concurrent requests and tokens/minute.

Token quotas reserve ``estimated input + max_tokens`` at admission (the worst case) and
refund the unused part when the request finishes, so a tenant can never exceed its
budget even before the engine reports usage.

State is process-local. A multi-replica gateway needs a shared store (e.g. Redis) to
enforce a global quota; that is intentionally out of scope for v0.1 (ADR-005).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

from pydantic import Field

from inferencelab.clock import Clock
from inferencelab.domain.errors import RateLimitedError
from inferencelab.limits.token_bucket import TokenBucket
from inferencelab.schema import StrictModel

__all__ = ["TenantLease", "TenantLimiter", "TenantLimitsConfig", "TenantQuota"]


class TenantQuota(StrictModel):
    requests_per_second: float | None = Field(default=None, gt=0)
    burst: int | None = Field(default=None, ge=1, description="Request bucket size")
    max_concurrent: int | None = Field(default=None, ge=1)
    tokens_per_minute: float | None = Field(default=None, gt=0)
    token_burst: int | None = Field(default=None, ge=1)


class TenantLimitsConfig(StrictModel):
    default: TenantQuota | None = Field(default=None, description="Quota for unlisted tenants")
    tenants: dict[str, TenantQuota] = Field(default_factory=dict)
    anonymous_tenant: str = "anonymous"
    max_tracked_tenants: int = Field(default=10_000, ge=1)


class _TenantState:
    __slots__ = ("in_flight", "quota", "requests", "tokens")

    def __init__(self, quota: TenantQuota, clock: Clock) -> None:
        self.quota = quota
        self.in_flight = 0
        self.requests: TokenBucket | None = None
        self.tokens: TokenBucket | None = None
        if quota.requests_per_second is not None:
            burst = quota.burst or max(1, math.ceil(quota.requests_per_second))
            self.requests = TokenBucket(burst, quota.requests_per_second, clock)
        if quota.tokens_per_minute is not None:
            capacity = quota.token_burst or quota.tokens_per_minute
            self.tokens = TokenBucket(capacity, quota.tokens_per_minute / 60.0, clock)


@dataclass(slots=True)
class TenantLease:
    tenant_id: str
    reserved_tokens: int
    _state: _TenantState | None
    _released: bool = False

    def release(self, actual_tokens: int | None = None) -> None:
        if self._released:
            return
        self._released = True
        if self._state is None:
            return
        self._state.in_flight -= 1
        if actual_tokens is not None and self._state.tokens is not None:
            self._state.tokens.refund(max(0, self.reserved_tokens - actual_tokens))


class TenantLimiter:
    def __init__(self, config: TenantLimitsConfig, clock: Clock) -> None:
        self.config = config
        self._clock = clock
        self._states: OrderedDict[str, _TenantState] = OrderedDict()

    def _state_for(self, tenant_id: str) -> _TenantState | None:
        state = self._states.get(tenant_id)
        if state is not None:
            self._states.move_to_end(tenant_id)
            return state
        quota = self.config.tenants.get(tenant_id, self.config.default)
        if quota is None:
            return None
        state = _TenantState(quota, self._clock)
        self._states[tenant_id] = state
        self._evict_idle()
        return state

    def _evict_idle(self) -> None:
        """Bound memory when clients present many distinct tenant ids."""
        while len(self._states) > self.config.max_tracked_tenants:
            for key, state in self._states.items():
                if state.in_flight == 0:
                    del self._states[key]
                    break
            else:
                return

    def acquire(self, tenant_id: str | None, estimated_tokens: int) -> TenantLease:
        tid = tenant_id or self.config.anonymous_tenant
        state = self._state_for(tid)
        if state is None:
            return TenantLease(tid, 0, None)
        quota = state.quota
        if quota.max_concurrent is not None and state.in_flight >= quota.max_concurrent:
            raise RateLimitedError(
                f"tenant {tid!r} exceeded {quota.max_concurrent} concurrent requests"
            )
        if state.requests is not None and not state.requests.try_consume(1.0):
            raise RateLimitedError(
                f"tenant {tid!r} exceeded {quota.requests_per_second:g} requests/s",
                retry_after_s=state.requests.time_until(1.0),
            )
        if state.tokens is not None:
            if estimated_tokens > state.tokens.capacity:
                if state.requests is not None:
                    state.requests.refund(1.0)
                raise RateLimitedError(
                    f"request needs {estimated_tokens} tokens, above tenant {tid!r} burst"
                )
            if not state.tokens.try_consume(float(estimated_tokens)):
                if state.requests is not None:
                    state.requests.refund(1.0)
                raise RateLimitedError(
                    f"tenant {tid!r} exceeded {quota.tokens_per_minute:g} tokens/min",
                    retry_after_s=state.tokens.time_until(float(estimated_tokens)),
                )
        state.in_flight += 1
        return TenantLease(tid, estimated_tokens if state.tokens is not None else 0, state)

    def in_flight(self, tenant_id: str) -> int:
        state = self._states.get(tenant_id)
        return 0 if state is None else state.in_flight
