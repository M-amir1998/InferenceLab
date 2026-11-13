"""Time series derived from request records (no extra sampling needed)."""

from __future__ import annotations

from collections.abc import Sequence

from inferencelab.domain.result import RequestRecord

__all__ = ["concurrency_timeline"]


def _step(events: list[tuple[float, int]]) -> tuple[list[float], list[float]]:
    events.sort()
    t: list[float] = []
    v: list[float] = []
    level = 0
    for time, change in events:
        level += change
        if t and t[-1] == time:
            v[-1] = float(level)
        else:
            t.append(time)
            v.append(float(level))
    return t, v


def concurrency_timeline(
    records: Sequence[RequestRecord],
) -> tuple[list[float], dict[str, list[float]]]:
    """Client in-flight requests and gateway queue depth over time (step functions).

    Queue intervals come from the gateway-reported queue time of each request, so the
    series is only shown when the target reported it.
    """
    inflight: list[tuple[float, int]] = []
    queue: list[tuple[float, int]] = []
    for r in records:
        if r.sent_at is None:
            continue
        end = r.completed_at if r.completed_at is not None else r.sent_at
        inflight += [(r.sent_at, +1), (end, -1)]
        if r.gateway_queue_s:
            queue += [(r.sent_at, +1), (r.sent_at + r.gateway_queue_s, -1)]
    if not inflight:
        return [], {}
    t_in, v_in = _step(inflight)
    series = {"in flight": v_in}
    if queue:
        t_q, v_q = _step(queue)
        # resample the queue series on the in-flight grid
        resampled: list[float] = []
        j = -1
        for time in t_in:
            while j + 1 < len(t_q) and t_q[j + 1] <= time:
                j += 1
            resampled.append(v_q[j] if j >= 0 else 0.0)
        series["gateway queue"] = resampled
    return t_in, series
