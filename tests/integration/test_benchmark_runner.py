"""Benchmark runner end to end in-process (virtual time): artifacts, profile, report, reload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from inferencelab.benchmarking.artifacts import load_run
from inferencelab.benchmarking.config import BenchmarkConfig
from inferencelab.benchmarking.runner import make_run_id, run_benchmark
from inferencelab.config.loader import load_model_file
from inferencelab.domain.hardware import HardwareRef
from inferencelab.experiments.plan import expand_experiment
from inferencelab.experiments.runner import run_local_experiment
from inferencelab.experiments.spec import ExperimentSpec
from tests.conftest import CONFIGS

pytestmark = pytest.mark.integration


async def test_smoke_benchmark_writes_every_artifact(tmp_path: Path) -> None:
    cfg = load_model_file(CONFIGS / "benchmarks" / "smoke_simulated.yaml", BenchmarkConfig)
    result = await run_benchmark(cfg, base_dir=CONFIGS / "benchmarks", output_root=tmp_path)
    run_dir = result.run_dir
    assert run_dir is not None
    for name in (
        "config.yaml",
        "metadata.json",
        "requests.parquet",
        "summary.json",
        "profile.json",
        "system.json",
        "report.md",
    ):
        assert (run_dir / name).exists(), name
    assert list((run_dir / "plots").glob("*.png"))
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["label"] == "SIMULATED BACKEND RESULTS" and meta["clock"] == "virtual"
    assert meta["privacy"]["capture_text"] is False
    assert "SIMULATED BACKEND RESULTS" in (run_dir / "report.md").read_text(encoding="utf-8")
    loaded = load_run(run_dir)
    assert loaded.summary.counts.total == 40
    assert loaded.summary.counts.ok == result.summary.counts.ok == 36  # 4 warmup requests excluded
    assert all(r.prompt_text is None for r in loaded.records())  # no prompt text by default
    assert result.summary.engine["engine_generation_tokens"] == 40 * 16
    assert (
        yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["name"]
        == "smoke-simulated"
    )
    assert result.profile.simulated and result.profile.points[0].load.kind == "concurrency"


async def test_gateway_target_records_backend_and_queue(tmp_path: Path) -> None:
    cfg = load_model_file(CONFIGS / "benchmarks" / "gateway_inprocess.yaml", BenchmarkConfig)
    result = await run_benchmark(
        cfg, base_dir=CONFIGS / "benchmarks", output_root=tmp_path, plots=False
    )
    backends = {r.backend for r in result.records}
    assert backends <= {"mock-a", "mock-b"} and backends
    assert all(r.gateway_queue_s is not None for r in result.records if r.ok)
    assert result.metadata["target"]["routing_policy"] == "latency_aware"


async def test_runs_are_deterministic_under_virtual_time(tmp_path: Path) -> None:
    cfg = load_model_file(CONFIGS / "benchmarks" / "smoke_simulated.yaml", BenchmarkConfig)
    a = await run_benchmark(cfg, base_dir=CONFIGS / "benchmarks", write=False)
    b = await run_benchmark(cfg, base_dir=CONFIGS / "benchmarks", write=False)
    assert [r.chunk_times for r in a.records] == [r.chunk_times for r in b.records]
    assert a.summary.latency["ttft_ms"] == b.summary.latency["ttft_ms"]


async def test_capture_text_is_opt_in(tmp_path: Path) -> None:
    cfg = load_model_file(CONFIGS / "benchmarks" / "smoke_simulated.yaml", BenchmarkConfig)
    result = await run_benchmark(
        cfg.model_copy(update={"capture_text": True}), base_dir=CONFIGS / "benchmarks", write=False
    )
    assert all(r.prompt_text and r.output_text for r in result.records)


async def test_pricing_produces_cost_report(tmp_path: Path) -> None:
    cfg = load_model_file(CONFIGS / "benchmarks" / "smoke_simulated.yaml", BenchmarkConfig)
    hardware = HardwareRef(gpu="cpu", count=1, hourly_cost_usd=3.6, price_source="test")
    priced = cfg.model_copy(update={"hardware": hardware})
    result = await run_benchmark(priced, base_dir=CONFIGS / "benchmarks", write=False)
    assert result.summary.cost is not None
    assert result.summary.cost["total_cost_usd"] == pytest.approx(
        result.summary.throughput.duration_s / 1000
    )


async def test_local_experiment_merges_profiles(tmp_path: Path) -> None:
    spec = load_model_file(CONFIGS / "experiments" / "local_prefix_cache.yaml", ExperimentSpec)
    plan = expand_experiment(spec, CONFIGS / "experiments")
    result = await run_local_experiment(plan, output_root=tmp_path, plots=False)
    assert len(result.runs) == 4 and result.report.exists()
    text = result.report.read_text(encoding="utf-8")
    assert "SIMULATED BACKEND RESULTS" in text and "Prefix-cache workloads" in text
    hits = {
        run.target: res.summary.engine.get("prefix_cache_hit_rate")
        for run, res in result.runs
        if run.workload.name == "local_shared_prefix"
    }
    assert hits["sim_a"] is not None and hits["sim_a"] > 0.5  # cache on: shared prefixes hit
    assert hits["sim_a_nocache"] == 0.0
    assert len(list((result.out_dir / "profiles").glob("*.json"))) == 4


def test_run_ids_are_unique_and_safe() -> None:
    ids = {make_run_id("My Run!") for _ in range(20)}
    assert len(ids) == 20 and all("my-run" in i and " " not in i for i in ids)
