from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferencelab.benchmarking.artifacts import (
    load_run,
    read_records,
    resolve_run_dir,
    write_json,
    write_records,
)
from inferencelab.domain.response import TokenCountSource
from inferencelab.domain.result import Phase, RequestRecord, RequestStatus
from inferencelab.domain.slo import SLOSpec
from inferencelab.metrics.summary import summarize_records
from inferencelab.quality.evaluators import (
    BUILTIN_CASES,
    QualityCase,
    check_output,
    compare_backends,
    json_valid,
    prefix_agreement,
)
from inferencelab.reporting import plots
from inferencelab.reporting.report import render_run_report
from inferencelab.reporting.tables import (
    RunRow,
    comparison_table,
    fmt,
    latency_table,
    prefix_cache_table,
)
from inferencelab.reporting.timeseries import concurrency_timeline


def records(n: int = 30) -> list[RequestRecord]:
    out = []
    for i in range(n):
        sent = i * 0.1
        out.append(
            RequestRecord(
                index=i,
                request_id=f"r{i}",
                phase=Phase.MEASURE,
                status=RequestStatus.OK,
                scheduled_at=sent,
                sent_at=sent,
                first_token_at=sent + 0.05,
                completed_at=sent + 0.3,
                chunk_times=(sent + 0.05, sent + 0.15, sent + 0.25),
                input_tokens=50,
                output_tokens=3,
                token_source=TokenCountSource.REPORTED,
                gateway_queue_s=0.01 if i % 2 else 0.0,
                backend="a" if i % 3 else "b",
                cached_input_tokens=10,
            )
        )
    return out


def test_fmt() -> None:
    assert (
        fmt(None) == "n/a"
        and fmt(3) == "3"
        and fmt(1234.5) == "1,234"
        and fmt(1.234, 2, " ms") == "1.23 ms"
    )


def test_tables_only_show_measured_values() -> None:
    s = summarize_records(
        records(), run_id="x", simulated=True, slo=SLOSpec.model_validate({"ttft_p50_ms": 100})
    )
    table = comparison_table([RunRow(backend="vllm", config="bf16", summary=s)])
    assert table.splitlines()[0].startswith("| Backend | Config | TTFT p95")
    assert "| vllm | bf16 |" in table and "n/a" in table  # GPU memory and cost not measured
    cache = prefix_cache_table([RunRow(backend="vllm", config="c", summary=s, prefix_caching=True)])
    assert "| on |" in cache and "20.0%" in cache
    assert "| TTFT |" in latency_table(s)


def test_run_report_sections_and_banner(tmp_path: Path) -> None:
    s = summarize_records(
        records(), run_id="x", simulated=True, slo=SLOSpec.model_validate({"ttft_p95_ms": 100})
    )
    text = render_run_report(
        s,
        {"name": "demo", "created_at": "now", "inferencelab_version": "t"},
        plots=[tmp_path / "plots" / "a.png"],
        run_dir=tmp_path,
    )
    assert text.splitlines()[2].startswith("> **SIMULATED BACKEND RESULTS**")
    for section in (
        "## Deployment",
        "## Hardware",
        "## Workload",
        "## SLO",
        "## Latency",
        "## Throughput",
        "## Errors",
        "## Cost",
        "## Memory",
        "## Quality validation",
        "## Methodology",
    ):
        assert section in text
    assert "INSUFFICIENT DATA" in text and "plots/a.png" in text
    gpu = s.model_copy(update={"label": "GPU INFERENCE RESULTS", "simulated": False})
    assert "GPU INFERENCE RESULTS" in render_run_report(gpu, {})


def test_concurrency_timeline() -> None:
    t, series = concurrency_timeline(records(10))
    assert t == sorted(t) and max(series["in flight"]) >= 2 and "gateway queue" in series
    assert concurrency_timeline([]) == ([], {})


def test_record_parquet_roundtrip_and_run_loading(tmp_path: Path) -> None:
    recs = records(5)
    run = tmp_path / "20260101-000000-x-abcd"
    run.mkdir()
    write_records(run / "requests.parquet", recs)
    assert read_records(run / "requests.parquet") == recs
    s = summarize_records(recs, run_id=run.name, simulated=True)
    write_json(run / "summary.json", s.model_dump(mode="json"))
    write_json(run / "metadata.json", {"name": "x"})
    loaded = load_run(run)
    assert loaded.summary.counts.ok == 5 and loaded.config == {} and loaded.profile is None
    assert len(loaded.records()) == 5
    assert resolve_run_dir(run.name, tmp_path) == run and resolve_run_dir("abcd", tmp_path) == run
    assert resolve_run_dir(str(run), tmp_path) == run
    with pytest.raises(FileNotFoundError):
        resolve_run_dir("nope", tmp_path)


def test_plots_render_pngs(tmp_path: Path) -> None:
    assert plots.plots_available()
    colors = plots.assign_colors(["b", "a"])
    assert colors == {"a": plots.SERIES_COLORS[0], "b": plots.SERIES_COLORS[1]}
    with pytest.raises(ValueError):
        plots.assign_colors([str(i) for i in range(9)])
    p1 = plots.sweep(
        {"x": ([1, 2, 4], [1.0, None, 3.0])},
        path=tmp_path / "s.png",
        title="t",
        xlabel="c",
        ylabel="y",
        log_x=True,
    )
    p2 = plots.latency_cdf(
        {"a": [1.0, 2.0], "b": [3.0]},
        path=tmp_path / "c.png",
        title="t",
        xlabel="ms",
        note="SIMULATED",
    )
    p3 = plots.timeline(
        [0.0, 1.0], {"q": [0.0, 2.0]}, path=tmp_path / "t.png", title="t", ylabel="n"
    )
    p4 = plots.small_multiples_timeline(
        [0.0, 1.0],
        [("a", {"x": [1.0, 2.0]}), ("b", {"y": [0.0, 1.0], "z": [1.0, 0.0]})],
        path=tmp_path / "m.png",
        title="t",
        note="n",
    )
    for p in (p1, p2, p3, p4):
        assert p is not None and p.stat().st_size > 1000
    assert plots.latency_cdf({}, path=tmp_path / "e.png", title="t", xlabel="x") is None


# ---------------------------------------------------------------- quality
def test_quality_checks() -> None:
    assert prefix_agreement("a b c d", "a b x d") == 0.5
    assert prefix_agreement("", "") == 1.0 and prefix_agreement("", "x") == 0.0
    assert json_valid('```json\n{"name": "Ada", "age": 36}\n```', ["name", "age"])
    assert not json_valid('{"name": "Ada"}', ["age"]) and not json_valid("nope")
    add = next(c for c in BUILTIN_CASES if c.id == "add")
    assert check_output(add, "The answer is 42.") and not check_output(add, "41")
    assert check_output(QualityCase(id="e", prompt="p", kind="exact", expected="Yes"), " yes ")
    assert check_output(
        QualityCase(id="c", prompt="p", kind="contains", expected="paris"), "It is Paris."
    )


async def test_compare_backends_with_mocks() -> None:
    from inferencelab.backends.config import LatencySpec, MockBackendConfig
    from inferencelab.backends.mock.backend import MockBackend
    from inferencelab.clock import VirtualClock

    clock = VirtualClock()
    ref = MockBackend(
        MockBackendConfig(name="ref", latency=LatencySpec(ttft_ms=0, tpot_ms=0), seed=1),
        clock=clock,
    )
    same = MockBackend(
        MockBackendConfig(name="same", latency=LatencySpec(ttft_ms=0, tpot_ms=0), seed=1),
        clock=clock,
    )
    other = MockBackend(
        MockBackendConfig(name="other", latency=LatencySpec(ttft_ms=0, tpot_ms=0), seed=2),
        clock=clock,
    )
    identical = await clock.run(
        compare_backends(ref, same, reference_model="m", candidate_model="m", max_tokens=8)
    )
    assert identical.exact_match_rate == 1.0 and identical.mean_prefix_agreement == 1.0
    different = await clock.run(
        compare_backends(ref, other, reference_model="m", candidate_model="m", max_tokens=8)
    )
    assert different.exact_match_rate == 0.0 and different.accuracy_delta == 0.0
    assert json.loads(different.model_dump_json())["cases"] == len(BUILTIN_CASES)
