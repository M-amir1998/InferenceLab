"""Per-backend circuit breaker: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

* **CLOSED**: calls flow. The circuit opens after ``failure_threshold`` consecutive
  failures, or when the failure rate over the last ``window_size`` calls reaches
  ``failure_rate_threshold`` (once ``min_calls`` results exist).
* **OPEN**: calls are rejected immediately (fail fast, give the backend room to recover).
  After ``open_duration_s`` the next state query moves it to HALF_OPEN.
* **HALF_OPEN**: at most ``half_open_max_calls`` concurrent probes; ``success_threshold``
  successful probes close the circuit, any probe failure re-opens it.

Every admitted call receives a :class:`CircuitPermit` stamped with the breaker's
*generation*. Results from a previous generation (a call admitted while CLOSED that
finishes after the circuit opened) cannot flip the current state — a classic source
of flapping in naive breakers.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from enum import StrEnum

from pydantic import Field

from inferencelab.clock import Clock
from inferencelab.schema import StrictModel

__all__ = ["CircuitBreaker", "CircuitBreakerConfig", "CircuitPermit", "CircuitState"]


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    @property
    def gauge_value(self) -> int:
        return {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}[self]


class CircuitBreakerConfig(StrictModel):
    enabled: bool = True
    failure_threshold: int = Field(default=5, ge=1)
    failure_rate_threshold: float | None = Field(default=0.5, gt=0, le=1)
    window_size: int = Field(default=20, ge=1)
    min_calls: int = Field(default=10, ge=1)
    open_duration_s: float = Field(default=10.0, gt=0)
    half_open_max_calls: int = Field(default=1, ge=1)
    success_threshold: int = Field(default=1, ge=1)


class CircuitPermit:
    __slots__ = ("_breaker", "_done", "generation", "probe")

    def __init__(self, breaker: CircuitBreaker, generation: int, probe: bool) -> None:
        self._breaker = breaker
        self.generation = generation
        self.probe = probe
        self._done = False

    def success(self) -> None:
        self._finish("success")

    def failure(self) -> None:
        self._finish("failure")

    def ignore(self) -> None:
        """Release without a verdict (cancellation, client error)."""
        self._finish("ignore")

    def _finish(self, outcome: str) -> None:
        if self._done:
            return
        self._done = True
        self._breaker._complete(self, outcome)


Transition = Callable[[CircuitState, CircuitState], None]


class CircuitBreaker:
    def __init__(
        self,
        config: CircuitBreakerConfig,
        clock: Clock,
        *,
        name: str = "",
        on_transition: Transition | None = None,
    ) -> None:
        self.config = config
        self.name = name
        self._clock = clock
        self._on_transition = on_transition
        self._state = CircuitState.CLOSED
        self._generation = 0
        self._opened_at = 0.0
        self._consecutive_failures = 0
        self._window: deque[bool] = deque(maxlen=config.window_size)
        self._probes_in_flight = 0
        self._probe_successes = 0

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> CircuitState:
        if (
            self._state is CircuitState.OPEN
            and self._clock.now() - self._opened_at >= self.config.open_duration_s
        ):
            self._transition(CircuitState.HALF_OPEN)
        return self._state

    @property
    def retry_after_s(self) -> float:
        if self.state is not CircuitState.OPEN:
            return 0.0
        return max(0.0, self.config.open_duration_s - (self._clock.now() - self._opened_at))

    def is_open(self) -> bool:
        """True if a call would be rejected right now (no side effects on permits)."""
        state = self.state
        if state is CircuitState.OPEN:
            return True
        return (
            state is CircuitState.HALF_OPEN
            and self._probes_in_flight >= self.config.half_open_max_calls
        )

    def acquire(self) -> CircuitPermit | None:
        """Admit a call, or return ``None`` if the circuit rejects it."""
        if not self.config.enabled:
            return CircuitPermit(self, self._generation, probe=False)
        state = self.state
        if state is CircuitState.CLOSED:
            return CircuitPermit(self, self._generation, probe=False)
        if (
            state is CircuitState.HALF_OPEN
            and self._probes_in_flight < self.config.half_open_max_calls
        ):
            self._probes_in_flight += 1
            return CircuitPermit(self, self._generation, probe=True)
        return None

    # ------------------------------------------------------------------ results
    def _complete(self, permit: CircuitPermit, outcome: str) -> None:
        if permit.probe and permit.generation == self._generation:
            self._probes_in_flight = max(0, self._probes_in_flight - 1)
        if not self.config.enabled or outcome == "ignore" or permit.generation != self._generation:
            return  # stale result from an earlier state: never flips the current state
        state = self._state
        if state is CircuitState.HALF_OPEN:
            if outcome == "failure":
                self._open()
            else:
                self._probe_successes += 1
                if self._probe_successes >= self.config.success_threshold:
                    self._transition(CircuitState.CLOSED)
        elif state is CircuitState.CLOSED:
            self._window.append(outcome == "success")
            if outcome == "success":
                self._consecutive_failures = 0
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.config.failure_threshold or self._rate_tripped():
                self._open()

    def _rate_tripped(self) -> bool:
        threshold = self.config.failure_rate_threshold
        if threshold is None or len(self._window) < self.config.min_calls:
            return False
        failures = sum(1 for ok in self._window if not ok)
        return failures / len(self._window) >= threshold

    def _open(self) -> None:
        self._opened_at = self._clock.now()
        self._transition(CircuitState.OPEN)

    def _transition(self, new: CircuitState) -> None:
        old = self._state
        if old is new:
            return
        self._state = new
        self._generation += 1
        self._probes_in_flight = 0
        self._probe_successes = 0
        if new is CircuitState.CLOSED:
            self._consecutive_failures = 0
            self._window.clear()
        if self._on_transition is not None:
            self._on_transition(old, new)
