"""Run summaries: latency distributions, throughput, goodput, errors, tokens and SLOs.

Definitions (docs/methodology/latency.md, goodput.md):

* Only requests in the **measurement window** (phase ``measure``) count.
* **Duration** is the makespan of measured requests: first send → last token
  (as ``vllm bench serve`` does). Throughput divides *successful* work by it.
* **Goodput** = successful, SLO-compliant requests per second. Throughput and goodput
  are always reported separately.
* **SLO attainment** = compliant / eligible, where eligible excludes requests the
  workload cancelled on purpose and requests the client never sent.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from inferencelab.domain.response import TokenCountSource
from inferencelab.domain.result import Phase, RequestRecord, RequestStatus
from inferencelab.domain.slo import RequestThresholds, SLOSpec, is_request_compliant
from inferencelab.metrics.stats import MIN_TAIL_SAMPLES, Distribution, min_samples_for
from inferencelab.schema import MutableModel

__all__ = [
    "DistributionSummary",
    "RunSummary",
    "SLOObjectiveResult",
    "SLOReport",
    "summarize_records",
]

SIMULATED_LABEL = "SIMULATED BACKEND RESULTS"
GPU_LABEL = "GPU INFERENCE RESULTS"
UNKNOWN_LABEL = "UNVERIFIED BACKEND TYPE"

_BACKEND_ERRORS = {
    "backend_error",
    "backend_unavailable",
    "backend_overloaded",
    "stream_interrupted",
    "internal",
}
_CIRCUIT = {"circuit_open", "no_healthy_backend"}
_QUEUE = {"queue_full", "load_shed"}


class DistributionSummary(MutableModel):
    count: int = 0
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None

    @classmethod
    def from_values(
        cls, values: Sequence[float], *, scale: float = 1000.0, min_tail: int = MIN_TAIL_SAMPLES
    ) -> DistributionSummary:
        d = Distribution.of(values, scale=scale, min_tail_samples=min_tail)
        return cls(
            count=d.count,
            mean=d.mean,
            std=d.std,
            min=d.min,
            max=d.max,
            p50=d.p(50),
            p90=d.p(90),
            p95=d.p(95),
            p99=d.p(99),
        )


class Counts(MutableModel):
    total: int = 0
    measured: int = 0
    ok: int = 0
    error: int = 0
    timeout: int = 0
    cancelled: int = 0
    rejected: int = 0
    dropped: int = 0


class Throughput(MutableModel):
    duration_s: float = 0.0
    request_throughput: float = 0.0
    input_token_throughput: float = 0.0
    output_token_throughput: float = 0.0
    total_token_throughput: float = 0.0
    goodput_rps: float | None = None
    goodput_output_token_throughput: float | None = None


class ErrorRates(MutableModel):
    attempted: int = 0
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    backend_error_rate: float = 0.0
    invalid_response_rate: float = 0.0
    cancelled_rate: float = 0.0
    circuit_rejection_rate: float = 0.0
    queue_rejection_rate: float = 0.0
    rate_limit_rate: float = 0.0
    by_kind: dict[str, int] = Field(default_factory=dict)
    by_http_status: dict[str, int] = Field(default_factory=dict)


class TokenTotals(MutableModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reported_requests: int = 0
    estimated_requests: int = 0
    mean_input_tokens: float | None = None
    mean_output_tokens: float | None = None
    cached_input_tokens: int | None = None
    cached_input_ratio: float | None = None

    @property
    def any_estimated(self) -> bool:
        return self.estimated_requests > 0


class BackendBreakdown(MutableModel):
    requests: int
    ok: int
    share: float
    ttft_ms: DistributionSummary
    e2e_ms: DistributionSummary
    output_tokens: int


class SLOObjectiveResult(MutableModel):
    label: str
    metric: str
    percentile: float
    threshold_ms: float
    measured_ms: float | None
    samples: int
    required_samples: int
    status: Literal["pass", "fail", "insufficient_data"]


class SLOReport(MutableModel):
    name: str
    objectives: list[SLOObjectiveResult] = Field(default_factory=list)
    availability_target: float | None = None
    availability_measured: float | None = None
    availability_status: Literal["pass", "fail", "insufficient_data"] | None = None
    thresholds: dict[str, float | None] = Field(default_factory=dict)
    attainment: float | None = None
    status: Literal["pass", "fail", "insufficient_data"] = "insufficient_data"


class RunSummary(MutableModel):
    schema_version: int = 1
    run_id: str
    label: str
    simulated: bool | None
    counts: Counts
    throughput: Throughput
    latency: dict[str, DistributionSummary]
    errors: ErrorRates
    tokens: TokenTotals
    per_backend: dict[str, BackendBreakdown] = Field(default_factory=dict)
    slo: SLOReport | None = None
    engine: dict[str, float] = Field(default_factory=dict)
    cost: dict[str, float | str | None] | None = None
    warnings: list[str] = Field(default_factory=list)


def _rate(n: int, d: int) -> float:
    return n / d if d else 0.0


def evaluate_slo(
    spec: SLOSpec, measured: Sequence[RequestRecord], *, min_tail: int = MIN_TAIL_SAMPLES
) -> tuple[SLOReport, set[int]]:
    """Evaluate objectives and classify requests. Returns the report and compliant indices."""
    ok = [r for r in measured if r.ok]
    values: dict[str, list[float]] = {
        "ttft": [v for r in ok if (v := r.ttft_s) is not None],
        "tpot": [v for r in ok if (v := r.tpot_s) is not None],
        "e2e": [v for r in ok if (v := r.e2e_s) is not None],
        "itl": [v for r in ok for v in r.itl_s],
    }
    results: list[SLOObjectiveResult] = []
    for obj in spec.objectives:
        sample = values[obj.metric]
        need = min_samples_for(obj.percentile, min_tail)
        dist = Distribution.of(
            sample, scale=1000.0, quantiles=(obj.percentile,), min_tail_samples=min_tail
        )
        measured_ms = dist.p(obj.percentile)
        status: Literal["pass", "fail", "insufficient_data"]
        if measured_ms is None:
            status = "insufficient_data"
        else:
            status = "pass" if measured_ms <= obj.threshold_ms else "fail"
        results.append(
            SLOObjectiveResult(
                label=obj.label,
                metric=obj.metric,
                percentile=obj.percentile,
                threshold_ms=obj.threshold_ms,
                measured_ms=measured_ms,
                samples=len(sample),
                required_samples=need,
                status=status,
            )
        )
    eligible = [
        r for r in measured if r.status not in (RequestStatus.CANCELLED, RequestStatus.DROPPED)
    ]
    thresholds: RequestThresholds = spec.effective_thresholds()
    compliant = {
        r.index
        for r in ok
        if r.status is RequestStatus.OK
        and is_request_compliant(thresholds, ttft_s=r.ttft_s, tpot_s=r.tpot_s, e2e_s=r.e2e_s)
    }
    report = SLOReport(
        name=spec.name,
        objectives=results,
        thresholds=thresholds.model_dump(),
        attainment=_rate(len(compliant), len(eligible)) if eligible else None,
    )
    if spec.availability is not None:
        report.availability_target = spec.availability
        if eligible:
            report.availability_measured = _rate(len(ok), len(eligible))
        # To observe a failure rate of (1 - target) at all you need ~1/(1 - target) requests.
        need = (
            round(1.0 / (1.0 - spec.availability)) if spec.availability < 1 else len(eligible) + 1
        )
        if len(eligible) < need or report.availability_measured is None:
            report.availability_status = "insufficient_data"
        else:
            report.availability_status = (
                "pass" if report.availability_measured >= spec.availability else "fail"
            )
    statuses = [o.status for o in results]
    if report.availability_status is not None:
        statuses.append(report.availability_status)
    if "fail" in statuses:
        report.status = "fail"
    elif not statuses or "insufficient_data" in statuses:
        report.status = "insufficient_data"
    else:
        report.status = "pass"
    return report, compliant


def summarize_records(
    records: Sequence[RequestRecord],
    *,
    run_id: str,
    simulated: bool | None,
    slo: SLOSpec | None = None,
    min_tail: int = MIN_TAIL_SAMPLES,
) -> RunSummary:
    measured = [r for r in records if r.phase is Phase.MEASURE]
    status_counts = Counter(r.status for r in measured)
    counts = Counts(
        total=len(records),
        measured=len(measured),
        ok=status_counts[RequestStatus.OK],
        error=status_counts[RequestStatus.ERROR],
        timeout=status_counts[RequestStatus.TIMEOUT],
        cancelled=status_counts[RequestStatus.CANCELLED],
        rejected=status_counts[RequestStatus.REJECTED],
        dropped=status_counts[RequestStatus.DROPPED],
    )
    ok = [r for r in measured if r.ok]
    sent = [r for r in measured if r.sent_at is not None]
    starts = [r.sent_at for r in sent if r.sent_at is not None]
    ends = [e for r in sent if (e := r.last_token_at) is not None]
    duration = max(ends) - min(starts) if starts and ends else 0.0

    in_tok = sum(r.input_tokens or 0 for r in ok)
    out_tok = sum(r.output_tokens or 0 for r in ok)
    throughput = Throughput(
        duration_s=duration,
        request_throughput=len(ok) / duration if duration > 0 else 0.0,
        input_token_throughput=in_tok / duration if duration > 0 else 0.0,
        output_token_throughput=out_tok / duration if duration > 0 else 0.0,
        total_token_throughput=(in_tok + out_tok) / duration if duration > 0 else 0.0,
    )

    def dist(vals: Sequence[float]) -> DistributionSummary:
        return DistributionSummary.from_values(vals, min_tail=min_tail)

    latency = {
        "ttft_ms": dist([v for r in ok if (v := r.ttft_s) is not None]),
        "tpot_ms": dist([v for r in ok if (v := r.tpot_s) is not None]),
        "itl_ms": dist([v for r in ok for v in r.itl_s]),
        "e2e_ms": dist([v for r in ok if (v := r.e2e_s) is not None]),
        "e2e_from_schedule_ms": dist([v for r in ok if (v := r.e2e_from_schedule_s) is not None]),
        "schedule_lag_ms": dist([v for r in sent if (v := r.schedule_lag_s) is not None]),
        "gateway_queue_ms": dist([r.gateway_queue_s for r in ok if r.gateway_queue_s is not None]),
        "engine_queue_ms": dist([r.engine_queue_s for r in ok if r.engine_queue_s is not None]),
        "engine_prefill_ms": dist(
            [r.engine_prefill_s for r in ok if r.engine_prefill_s is not None]
        ),
        "engine_decode_ms": dist([r.engine_decode_s for r in ok if r.engine_decode_s is not None]),
    }

    attempted = [r for r in measured if r.status is not RequestStatus.DROPPED]
    kinds = Counter(r.error_kind for r in attempted if r.error_kind)
    http = Counter(str(r.http_status) for r in attempted if r.http_status and r.http_status >= 400)
    n_att = len(attempted)
    errors = ErrorRates(
        attempted=n_att,
        error_rate=_rate(
            sum(1 for r in attempted if r.status in (RequestStatus.ERROR, RequestStatus.TIMEOUT)),
            n_att,
        ),
        timeout_rate=_rate(counts.timeout, n_att),
        backend_error_rate=_rate(sum(kinds[k] for k in _BACKEND_ERRORS), n_att),
        invalid_response_rate=_rate(kinds["malformed_response"], n_att),
        cancelled_rate=_rate(counts.cancelled, n_att),
        circuit_rejection_rate=_rate(sum(kinds[k] for k in _CIRCUIT), n_att),
        queue_rejection_rate=_rate(sum(kinds[k] for k in _QUEUE), n_att),
        rate_limit_rate=_rate(kinds["rate_limited"], n_att),
        by_kind=dict(sorted(kinds.items())),
        by_http_status=dict(sorted(http.items())),
    )

    reported = sum(1 for r in ok if r.token_source is TokenCountSource.REPORTED)
    cached_vals = [r for r in ok if r.cached_input_tokens is not None]
    cached = sum(r.cached_input_tokens or 0 for r in cached_vals)
    cached_input = sum(r.input_tokens or 0 for r in cached_vals)
    tokens = TokenTotals(
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=in_tok + out_tok,
        reported_requests=reported,
        estimated_requests=len(ok) - reported,
        mean_input_tokens=in_tok / len(ok) if ok else None,
        mean_output_tokens=out_tok / len(ok) if ok else None,
        cached_input_tokens=cached if cached_vals else None,
        cached_input_ratio=cached / cached_input if cached_vals and cached_input else None,
    )

    per_backend: dict[str, BackendBreakdown] = {}
    for name in sorted({r.backend for r in attempted if r.backend}):
        rs = [r for r in attempted if r.backend == name]
        rok = [r for r in rs if r.ok]
        per_backend[name] = BackendBreakdown(
            requests=len(rs),
            ok=len(rok),
            share=_rate(len(rs), n_att),
            ttft_ms=dist([v for r in rok if (v := r.ttft_s) is not None]),
            e2e_ms=dist([v for r in rok if (v := r.e2e_s) is not None]),
            output_tokens=sum(r.output_tokens or 0 for r in rok),
        )

    label = SIMULATED_LABEL if simulated else GPU_LABEL if simulated is False else UNKNOWN_LABEL
    summary = RunSummary(
        run_id=run_id,
        label=label,
        simulated=simulated,
        counts=counts,
        throughput=throughput,
        latency=latency,
        errors=errors,
        tokens=tokens,
        per_backend=per_backend,
    )
    if slo is not None:
        report, compliant = evaluate_slo(slo, measured, min_tail=min_tail)
        summary.slo = report
        good = [r for r in ok if r.index in compliant]
        if duration > 0:
            throughput.goodput_rps = len(good) / duration
            throughput.goodput_output_token_throughput = (
                sum(r.output_tokens or 0 for r in good) / duration
            )
    summary.warnings = _warnings(summary, measured)
    return summary


def _warnings(summary: RunSummary, measured: Sequence[RequestRecord]) -> list[str]:
    out: list[str] = []
    if summary.counts.measured == 0:
        out.append("no requests in the measurement window")
    if summary.counts.dropped:
        out.append(
            f"{summary.counts.dropped} requests were dropped client-side (max_in_flight reached); "
            "the load generator could not sustain the configured arrival rate"
        )
    if summary.tokens.any_estimated:
        out.append(
            f"token counts for {summary.tokens.estimated_requests} requests are ESTIMATED "
            "(the backend did not report usage)"
        )
    lag = summary.latency["schedule_lag_ms"]
    e2e = summary.latency["e2e_ms"]
    # Use the highest percentile the sample size supports (max for tiny runs).
    tail = next(
        (
            (label, v)
            for label, v in (("p99", lag.p99), ("p95", lag.p95), ("p90", lag.p90), ("max", lag.max))
            if v is not None
        ),
        None,
    )
    if tail is not None and e2e.mean and tail[1] > max(10.0, 0.05 * e2e.mean):
        out.append(
            f"{tail[0]} schedule lag {tail[1]:.1f} ms: the client fell behind its schedule; "
            "compare e2e_from_schedule_ms with e2e_ms (coordinated omission)"
        )
    ttft = summary.latency["ttft_ms"]
    withheld = [
        f"p{q:g} (needs {min_samples_for(q)})"
        for q in (50, 90, 95, 99)
        if ttft.count and ttft.model_dump()[f"p{q:g}"] is None
    ]
    if withheld:
        out.append(
            f"{ttft.count} TTFT samples: percentiles withheld as statistically meaningless: "
            + ", ".join(withheld)
        )
    return out
