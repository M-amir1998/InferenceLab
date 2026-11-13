"""Request timeouts.

Four independent bounds (seconds, ``None`` disables):

* ``connect``      TCP/TLS connect to a backend (enforced by the HTTP client; see
                   ``connect_timeout_s`` on HTTP backend configs)
* ``first_token``  from dispatch to the backend until the first content token
* ``idle``         maximum gap between consecutive stream chunks
* ``overall``      total request budget from gateway admission, including queueing
                   and retries (a deadline, propagated to every attempt)

Timeouts are implemented with the injected clock so they are deterministic in tests.
A timeout cancels the in-flight backend call, which closes its connection and thereby
signals the engine to abort the request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TypeVar

from pydantic import Field

from inferencelab.clock import Clock
from inferencelab.domain.errors import BackendTimeoutError, TimeoutPhase
from inferencelab.schema import StrictModel

__all__ = ["TimeoutConfig", "next_with_timeout", "remaining"]

T = TypeVar("T")


class TimeoutConfig(StrictModel):
    first_token_s: float | None = Field(default=60.0, gt=0)
    idle_s: float | None = Field(default=30.0, gt=0)
    overall_s: float | None = Field(default=600.0, gt=0)


def remaining(deadline: float | None, clock: Clock) -> float | None:
    return None if deadline is None else deadline - clock.now()


def _effective(
    timeout_s: float | None, deadline: float | None, clock: Clock
) -> tuple[float | None, bool]:
    """Smaller of the phase timeout and the deadline; flag says whether the deadline won."""
    left = remaining(deadline, clock)
    if left is None:
        return timeout_s, False
    if timeout_s is None or left <= timeout_s:
        return max(0.0, left), True
    return timeout_s, False


async def next_with_timeout(
    stream: AsyncIterator[T],
    *,
    clock: Clock,
    timeout_s: float | None,
    phase: TimeoutPhase,
    deadline: float | None = None,
    backend: str | None = None,
) -> T:
    """``anext(stream)`` bounded by a phase timeout and an absolute deadline.

    Raises :class:`BackendTimeoutError` (``phase`` or ``OVERALL``) and lets
    ``StopAsyncIteration`` propagate at the end of the stream.
    """
    budget, deadline_bound = _effective(timeout_s, deadline, clock)
    if budget is not None and budget <= 0:
        raise BackendTimeoutError(
            TimeoutPhase.OVERALL, "request deadline exceeded", backend=backend
        )
    try:
        async with clock.timeout(budget):
            return await anext(stream)
    except TimeoutError as exc:
        hit = TimeoutPhase.OVERALL if deadline_bound else phase
        raise BackendTimeoutError(
            hit, f"{hit.value} timeout after {budget:.3f}s", backend=backend
        ) from exc
