"""Arrival processes as deterministic schedules (offsets in seconds from run start).

Open-loop patterns produce a schedule that is independent of how fast the server
answers — the property that makes them suitable for latency-under-load studies.
Closed-loop concurrency has no schedule: N workers send back to back.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from inferencelab.domain.workload import (
    ArrivalPattern,
    BurstArrival,
    ClosedLoopArrival,
    ConstantRateArrival,
    PoissonArrival,
    StepArrival,
    TraceArrival,
)

__all__ = ["arrival_offsets"]


def arrival_offsets(
    pattern: ArrivalPattern,
    rng: random.Random,
    *,
    duration_s: float | None = None,
) -> Iterator[float]:
    """Yield non-decreasing session start offsets.

    The iterator is infinite for unbounded patterns; callers stop on request count or
    ``duration_s``. Step patterns end after their last stage.
    """
    if isinstance(pattern, ClosedLoopArrival | TraceArrival):
        raise ValueError(f"{pattern.kind} arrivals have no precomputed schedule")
    if isinstance(pattern, ConstantRateArrival):
        i = 0
        while True:
            t = i / pattern.rate
            if duration_s is not None and t >= duration_s:
                return
            yield t
            i += 1
    elif isinstance(pattern, PoissonArrival):
        t = 0.0
        while True:
            if duration_s is not None and t >= duration_s:
                return
            yield t
            t += rng.expovariate(pattern.rate)
    elif isinstance(pattern, BurstArrival):
        k = 0
        while True:
            t = k * pattern.interval_s
            if duration_s is not None and t >= duration_s:
                return
            for _ in range(pattern.burst_size):
                yield t
            k += 1
    elif isinstance(pattern, StepArrival):
        stage_start = 0.0
        for stage in pattern.stages:
            stage_end = stage_start + stage.duration_s
            t = stage_start
            while t < stage_end:
                if duration_s is not None and t >= duration_s:
                    return
                yield t
                t += rng.expovariate(stage.rate) if pattern.poisson else 1.0 / stage.rate
            stage_start = stage_end
