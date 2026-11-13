"""Per-backend runtime statistics used by routing policies."""

from __future__ import annotations

import math

from inferencelab.clock import Clock

__all__ = ["BackendStats", "PeakEWMA"]


class PeakEWMA:
    """Peak-sensitive EWMA: jumps up immediately on a slow observation, decays slowly.

    The decay is time-based (``tau_s``), so an idle backend's estimate does not stay
    stuck at a stale value as it would with a per-sample EWMA.
    """

    def __init__(self, tau_s: float, clock: Clock) -> None:
        if tau_s <= 0:
            raise ValueError("tau_s must be positive")
        self._tau = tau_s
        self._clock = clock
        self._value: float | None = None
        self._last = clock.now()

    @property
    def value(self) -> float | None:
        return self._value

    def observe(self, sample: float) -> None:
        now = self._clock.now()
        if self._value is None or sample > self._value:
            self._value = sample
        else:
            w = math.exp(-max(0.0, now - self._last) / self._tau)
            self._value = self._value * w + sample * (1.0 - w)
        self._last = now


class BackendStats:
    def __init__(self, clock: Clock, *, alpha: float = 0.3, peak_tau_s: float = 10.0) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self.outstanding = 0
        self.ewma_ttft_s: float | None = None
        self.ewma_tpot_s: float | None = None
        self.peak = PeakEWMA(peak_tau_s, clock)
        self.successes = 0
        self.failures = 0

    def _ewma(self, current: float | None, sample: float) -> float:
        return sample if current is None else (1 - self._alpha) * current + self._alpha * sample

    def on_dispatch(self) -> None:
        self.outstanding += 1

    def on_first_token(self, ttft_s: float) -> None:
        self.ewma_ttft_s = self._ewma(self.ewma_ttft_s, ttft_s)
        self.peak.observe(ttft_s)

    def on_complete(self, *, ok: bool | None, tpot_s: float | None = None) -> None:
        """``ok=None`` is neutral (e.g. client cancellation): only the slot is released."""
        self.outstanding = max(0, self.outstanding - 1)
        if ok is True:
            self.successes += 1
            if tpot_s is not None:
                self.ewma_tpot_s = self._ewma(self.ewma_tpot_s, tpot_s)
        elif ok is False:
            self.failures += 1
