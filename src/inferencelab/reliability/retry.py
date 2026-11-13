"""Bounded retries with exponential backoff and a retry budget.

Rules (see docs/architecture/data-plane.md):

1. Retries are bounded by ``max_attempts`` and by the request's overall deadline.
2. **Before any output reaches the client**, retryable failures (connect errors,
   5xx/overload, first-token timeouts, malformed responses) may be retried, preferably
   on a different backend.
3. **After streaming output reached the client, never retry**: a second generation
   would duplicate or contradict user-visible text. The stream ends with an error event.
4. Non-streaming responses are buffered, so a mid-generation failure is invisible to the
   client and may be retried when ``retry_non_streaming_after_output`` is set.
5. A retry budget caps retries at a fraction of request volume so that a struggling
   fleet is not hit by a retry storm (the Finagle/Envoy approach).
"""

from __future__ import annotations

import random

from pydantic import Field

from inferencelab.clock import Clock
from inferencelab.domain.errors import ErrorKind, InferenceError
from inferencelab.limits.token_bucket import TokenBucket
from inferencelab.schema import StrictModel

__all__ = ["RetryBudget", "RetryConfig", "RetryPolicy"]

_MID_STREAM_RETRYABLE = frozenset(
    {
        ErrorKind.STREAM_INTERRUPTED,
        ErrorKind.MALFORMED_RESPONSE,
        ErrorKind.TIMEOUT_IDLE,
        ErrorKind.BACKEND_ERROR,
    }
)


class RetryConfig(StrictModel):
    max_attempts: int = Field(default=3, ge=1, description="Total attempts including the first")
    backoff_initial_s: float = Field(default=0.05, ge=0)
    backoff_max_s: float = Field(default=1.0, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1)
    jitter: float = Field(
        default=0.5, ge=0, le=1, description="Random fraction removed from backoff"
    )
    budget_ratio: float | None = Field(
        default=0.2, gt=0, le=1, description="Max retries per request (long-run average)"
    )
    budget_min_retries_per_s: float = Field(default=1.0, ge=0)
    retry_non_streaming_after_output: bool = True


class RetryPolicy:
    def __init__(self, config: RetryConfig, *, seed: int = 0) -> None:
        self.config = config
        self._rng = random.Random(seed)

    def backoff_s(self, failures: int) -> float:
        """Delay before the attempt following the ``failures``-th failure (>= 1)."""
        c = self.config
        if failures < 1 or c.backoff_initial_s == 0:
            return 0.0
        base = min(c.backoff_max_s, c.backoff_initial_s * c.backoff_multiplier ** (failures - 1))
        return base * (1.0 - c.jitter * self._rng.random())

    def should_retry(
        self, error: InferenceError, *, attempts_made: int, output_started: bool, streaming: bool
    ) -> bool:
        if attempts_made >= self.config.max_attempts:
            return False
        if not output_started:
            return error.retryable
        if streaming:
            return False  # rule 3: never duplicate user-visible output
        return self.config.retry_non_streaming_after_output and error.kind in _MID_STREAM_RETRYABLE


class RetryBudget:
    """Token-bucket retry budget: each request deposits ``ratio`` tokens, a retry costs one.

    A small time-based allowance (``min_retries_per_s``) keeps low-traffic services able
    to retry at all.
    """

    def __init__(self, ratio: float | None, min_retries_per_s: float, clock: Clock) -> None:
        self._ratio = ratio
        capacity = max(10.0, min_retries_per_s * 10.0)
        self._bucket = TokenBucket(capacity, min_retries_per_s, clock, initial=capacity)

    def record_request(self) -> None:
        if self._ratio is not None:
            self._bucket.refund(self._ratio)

    def try_spend(self) -> bool:
        if self._ratio is None:
            return True
        return self._bucket.try_consume(1.0)
