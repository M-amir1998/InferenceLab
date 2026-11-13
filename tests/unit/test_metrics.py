from __future__ import annotations

import numpy as np
import pytest

from inferencelab.domain.response import TokenCountSource
from inferencelab.domain.result import Phase, RequestRecord, RequestStatus
from inferencelab.domain.slo import SLOSpec
from inferencelab.metrics.stats import Distribution, min_samples_for, percentile
from inferencelab.metrics.summary import (
    GPU_LABEL,
    SIMULATED_LABEL,
    UNKNOWN_LABEL,
    evaluate_slo,
    summarize_records,
)


def ok(
    i: int,
    *,
    sent: float,
    ttft: float,
    tokens: int,
    tpot: float,
    cached: int | None = None,
    phase: Phase = Phase.MEASURE,
    backend: str = "a",
) -> RequestRecord:
    first = sent + ttft
    chunks = tuple(first + k * tpot for k in range(tokens))
    return RequestRecord(
        index=i,
        request_id=f"r{i}",
        phase=phase,
        status=RequestStatus.OK,
        scheduled_at=sent,
        sent_at=sent,
        first_token_at=first,
        completed_at=chunks[-1] + 0.001,
        chunk_times=chunks,
        input_tokens=100,
        output_tokens=tokens,
        token_source=TokenCountSource.REPORTED,
        cached_input_tokens=cached,
        backend=backend,
    )


def failed(i: int, status: RequestStatus, kind: str, http: int | None = None) -> RequestRecord:
    return RequestRecord(
        index=i,
        request_id=f"f{i}",
        phase=Phase.MEASURE,
        status=status,
        scheduled_at=0.0,
        sent_at=0.0 if status is not RequestStatus.DROPPED else None,
        first_token_at=None,
        completed_at=0.1,
        error_kind=kind,
        http_status=http,
    )


# ---------------------------------------------------------------- percentiles
def test_percentile_matches_numpy_linear() -> None:
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    for q in (0, 25, 50, 90, 100):
        assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)))
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_min_samples_rule() -> None:
    assert [min_samples_for(q) for q in (50, 90, 95, 99)] == [10, 50, 100, 500]
    with pytest.raises(ValueError):
        min_samples_for(100)


def test_distribution_withholds_tail_percentiles_for_small_samples() -> None:
    d = Distribution.of([float(i) for i in range(1, 101)], scale=1000.0)
    assert d.count == 100 and d.mean == pytest.approx(50500.0)
    assert d.p(95) is not None and d.p(99) is None
    tiny = Distribution.of([1.0, 2.0])
    assert tiny.p(50) is None and tiny.max == 2.0
    assert Distribution.of([]).count == 0
    assert Distribution.of([1.0, float("nan"), float("inf")]).count == 1
    assert set(d.to_dict()) >= {"count", "mean", "p50", "p99"}


# ---------------------------------------------------------------- summaries
def test_summary_throughput_latency_and_tokens() -> None:
    records = [ok(i, sent=i * 0.5, ttft=0.1, tokens=11, tpot=0.01, cached=50) for i in range(20)]
    s = summarize_records(records, run_id="x", simulated=True)
    assert s.label == SIMULATED_LABEL
    assert s.counts.ok == 20 and s.counts.measured == 20
    duration = (19 * 0.5 + 0.1 + 10 * 0.01) - 0.0
    assert s.throughput.duration_s == pytest.approx(duration)
    assert s.throughput.request_throughput == pytest.approx(20 / duration)
    assert s.throughput.output_token_throughput == pytest.approx(220 / duration)
    assert s.latency["ttft_ms"].mean == pytest.approx(100.0)
    assert s.latency["tpot_ms"].mean == pytest.approx(10.0)
    assert s.latency["itl_ms"].count == 200 and s.latency["itl_ms"].mean == pytest.approx(10.0)
    assert s.latency["e2e_ms"].mean == pytest.approx(200.0)
    assert s.tokens.input_tokens == 2000 and s.tokens.cached_input_ratio == pytest.approx(0.5)
    assert not s.tokens.any_estimated
    assert s.per_backend["a"].share == 1.0


def test_warmup_records_are_excluded() -> None:
    records = [
        ok(0, sent=0, ttft=5.0, tokens=2, tpot=0.1, phase=Phase.WARMUP),
        ok(1, sent=1, ttft=0.1, tokens=2, tpot=0.1),
    ]
    s = summarize_records(records, run_id="x", simulated=None)
    assert s.counts.total == 2 and s.counts.measured == 1
    assert s.latency["ttft_ms"].max == pytest.approx(100.0)
    assert s.label == UNKNOWN_LABEL
    assert summarize_records(records, run_id="x", simulated=False).label == GPU_LABEL


def test_error_taxonomy_rates() -> None:
    records = [
        ok(0, sent=0, ttft=0.1, tokens=2, tpot=0.1),
        failed(1, RequestStatus.TIMEOUT, "timeout_first_token", 504),
        failed(2, RequestStatus.ERROR, "backend_error", 502),
        failed(3, RequestStatus.ERROR, "malformed_response", 502),
        failed(4, RequestStatus.REJECTED, "queue_full", 503),
        failed(5, RequestStatus.REJECTED, "rate_limited", 429),
        failed(6, RequestStatus.REJECTED, "circuit_open", 503),
        failed(7, RequestStatus.CANCELLED, "cancelled"),
        failed(8, RequestStatus.DROPPED, "client_max_in_flight"),
    ]
    s = summarize_records(records, run_id="x", simulated=True)
    e = s.errors
    assert e.attempted == 8  # dropped requests were never sent
    assert e.timeout_rate == pytest.approx(1 / 8)
    assert e.backend_error_rate == pytest.approx(1 / 8)
    assert e.invalid_response_rate == pytest.approx(1 / 8)
    assert e.queue_rejection_rate == pytest.approx(1 / 8)
    assert e.rate_limit_rate == pytest.approx(1 / 8)
    assert e.circuit_rejection_rate == pytest.approx(1 / 8)
    assert e.by_http_status == {"429": 1, "502": 2, "503": 2, "504": 1}
    assert any("dropped" in w for w in s.warnings)


def test_goodput_is_separate_from_throughput() -> None:
    fast = [ok(i, sent=i * 0.1, ttft=0.05, tokens=5, tpot=0.01) for i in range(10)]
    slow = [ok(10 + i, sent=i * 0.1, ttft=0.8, tokens=5, tpot=0.01) for i in range(10)]
    spec = SLOSpec.model_validate({"ttft_p50_ms": 500})
    s = summarize_records(fast + slow, run_id="x", simulated=True, slo=spec)
    assert s.throughput.goodput_rps == pytest.approx(s.throughput.request_throughput / 2)
    assert s.slo is not None and s.slo.attainment == pytest.approx(0.5)


def test_slo_evaluation_pass_fail_and_insufficient() -> None:
    records = [ok(i, sent=i * 0.1, ttft=0.1 + 0.001 * i, tokens=3, tpot=0.02) for i in range(120)]
    spec = SLOSpec.model_validate(
        {"ttft_p95_ms": 500, "tpot_p95_ms": 10, "e2e_p99_ms": 1000, "availability": 0.999}
    )
    report, compliant = evaluate_slo(spec, records)
    status = {o.metric: o.status for o in report.objectives}
    assert status == {"ttft": "pass", "tpot": "fail", "e2e": "insufficient_data"}
    assert report.availability_status == "insufficient_data"  # 0.999 needs ~1000 requests
    assert report.status == "fail"
    assert compliant == set()  # TPOT 20 ms > tail threshold 10 ms


def test_no_slo_means_no_goodput() -> None:
    s = summarize_records([ok(0, sent=0, ttft=0.1, tokens=2, tpot=0.1)], run_id="x", simulated=True)
    assert s.slo is None and s.throughput.goodput_rps is None


def test_schedule_lag_warning_detects_coordinated_omission() -> None:
    lagging = [
        RequestRecord(
            index=i,
            request_id=f"r{i}",
            phase=Phase.MEASURE,
            status=RequestStatus.OK,
            scheduled_at=i * 0.01,
            sent_at=i * 0.01 + 0.5,
            first_token_at=i * 0.01 + 0.6,
            completed_at=i * 0.01 + 0.7,
            chunk_times=(i * 0.01 + 0.6, i * 0.01 + 0.7),
            output_tokens=2,
        )
        for i in range(60)
    ]
    s = summarize_records(lagging, run_id="x", simulated=True)
    assert any("coordinated omission" in w for w in s.warnings)
