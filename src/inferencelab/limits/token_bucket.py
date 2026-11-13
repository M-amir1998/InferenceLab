"""Token bucket rate limiter with an injected clock.

Invariant: ``0 <= tokens <= capacity`` at all times. Refill is computed lazily from
elapsed clock time, so no background task is needed and tests can advance time
explicitly instead of sleeping.
"""

from __future__ import annotations

import math

from inferencelab.clock import Clock

__all__ = ["TokenBucket"]

#: Tolerance for float accumulation (e.g. ten deposits of 0.1 must buy one token).
_EPS = 1e-9


class TokenBucket:
    __slots__ = ("_clock", "_last", "_tokens", "capacity", "refill_per_s")

    def __init__(
        self, capacity: float, refill_per_s: float, clock: Clock, *, initial: float | None = None
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_per_s < 0:
            raise ValueError("refill rate must be non-negative")
        self.capacity = float(capacity)
        self.refill_per_s = float(refill_per_s)
        self._clock = clock
        self._tokens = self.capacity if initial is None else min(self.capacity, max(0.0, initial))
        self._last = clock.now()

    def _refill(self) -> None:
        now = self._clock.now()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_s)
        self._last = max(self._last, now)

    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens

    def try_consume(self, amount: float = 1.0) -> bool:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        self._refill()
        if amount <= self._tokens + _EPS:
            self._tokens = max(0.0, self._tokens - amount)
            return True
        return False

    def refund(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        self._refill()
        self._tokens = min(self.capacity, self._tokens + amount)

    def time_until(self, amount: float = 1.0) -> float:
        """Seconds until ``amount`` tokens are available (``inf`` if never)."""
        self._refill()
        if amount <= self._tokens + _EPS:
            return 0.0
        if amount > self.capacity or self.refill_per_s == 0:
            return math.inf
        return (amount - self._tokens) / self.refill_per_s
