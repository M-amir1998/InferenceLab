"""CLI commands end to end (CPU only, no network beyond loopback)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inferencelab import __version__
from inferencelab.cli.main import app
from tests.conftest import CONFIGS

pytestmark = pytest.mark.e2e
runner = CliRunner()


def run(*args: str, code: int = 0) -> str:
    result = runner.invoke(app, list(args), catch_exceptions=False, env={"COLUMNS": "200"})
    assert result.exit_code == code, result.output
    return result.output


def test_version_and_doctor() -> None:
    assert run("--version").strip() == __version__
    out = run("doctor")
    assert "python" in out and "gpu runtime" in out


def test_registry_and_inspection_commands() -> None:
    out = run("model", "list", "--models-dir", str(CONFIGS / "models"))
    assert "Qwen/Qwen2.5-7B-Instruct" in out and "Apache-2.0" in out
    caps = run("backend", "list", "--config-dir", str(CONFIGS / "backends"))
    assert "prefix_caching" in caps and "sim-a" in caps
    wl = run(
        "workload", "inspect", str(CONFIGS / "workloads" / "shared_prefix.yaml"), "--samples", "50"
    )
    assert "shared prefix" in wl and "yes" in wl
    plan = run(
        "backend",
        "launch-plan",
        str(CONFIGS / "deployments" / "sglang_qwen_7b_awq.yaml"),
        "--models-dir",
        str(CONFIGS / "models"),
        "--hardware-dir",
        str(CONFIGS / "hardware"),
    )
    assert "sglang serve" in plan and "--quantization" not in plan and "fits" in plan


def test_benchmark_run_compare_report_profile(tmp_path: Path) -> None:
    out = run(
        "benchmark",
        "run",
        str(CONFIGS / "benchmarks" / "smoke_simulated.yaml"),
        "--output",
        str(tmp_path),
    )
    run_dir = Path(re.search(r"run_dir=(\S+)", out).group(1))  # type: ignore[union-attr]
    assert (run_dir / "summary.json").exists() and "SIMULATED BACKEND RESULTS" in out
    out2 = run(
        "benchmark",
        "run",
        str(CONFIGS / "benchmarks" / "gateway_inprocess.yaml"),
        "--output",
        str(tmp_path),
        "--no-plots",
    )
    run2 = re.search(r"run_dir=(\S+)", out2).group(1)  # type: ignore[union-attr]
    table = run(
        "benchmark",
        "compare",
        str(run_dir),
        run2,
        "--results",
        str(tmp_path),
        "--output",
        str(tmp_path / "cmp.md"),
    )
    assert "| Backend | Config | TTFT p95" in table and (tmp_path / "cmp.md").exists()
    assert run("report", run_dir.name, "--results", str(tmp_path)).strip().endswith("report.md")
    profile = run("profile", "show", str(run_dir / "profile.json"))
    assert "SIMULATED" in profile and "concurrency=4" in profile


def test_gpu_benchmark_requires_execute() -> None:
    out = run("benchmark", "run", str(CONFIGS / "benchmarks" / "vllm_single_gpu.yaml"), code=2)
    assert "--execute" in out and "nothing was run" in out


def test_capacity_and_autoscale(tmp_path: Path) -> None:
    out = run("capacity", "plan", str(CONFIGS / "capacity" / "plan_simulated.yaml"))
    assert "replicas" in out and "SIMULATED" in out
    sim = run(
        "autoscale",
        "simulate",
        str(CONFIGS / "autoscaling" / "diurnal_utilization.yaml"),
        str(CONFIGS / "autoscaling" / "diurnal_slo_aware.yaml"),
        "--output",
        str(tmp_path),
    )
    assert "utilization" in sim and "slo_aware" in sim
    assert (tmp_path / "diurnal_utilization.csv").exists() and (
        tmp_path / "diurnal_slo_aware.png"
    ).exists()


def test_experiment_plan_and_local_run(tmp_path: Path) -> None:
    plan = run("experiment", "plan", str(CONFIGS / "experiments" / "gpu_first_comparison.yaml"))
    assert "runs: 32 (cap 40)" in plan and "remote run" in plan
    out = run(
        "experiment",
        "run",
        str(CONFIGS / "experiments" / "local_prefix_cache.yaml"),
        "--output",
        str(tmp_path),
        "--no-plots",
    )
    assert "experiment report" in out
    run(
        "experiment",
        "run",
        str(CONFIGS / "experiments" / "gpu_trtllm.yaml"),
        "--output",
        str(tmp_path),
        code=2,
    )


def test_remote_plan_is_a_dry_run() -> None:
    out = run(
        "remote",
        "plan",
        str(CONFIGS / "experiments" / "gpu_trtllm.yaml"),
        "--host",
        str(CONFIGS / "hosts" / "example.yaml"),
    )
    assert "docker run" in out and "trtllm-serve" in out and "vllm/vllm-openai:v0.33.0" in out
    assert "hf download" in out and "docker stop" in out


def test_remote_run_requires_execute(tmp_path: Path) -> None:
    wheel = tmp_path / "inferencelab-0.1.0.dev0-py3-none-any.whl"
    wheel.write_bytes(b"not used")
    out = run(
        "remote",
        "run",
        str(CONFIGS / "experiments" / "gpu_trtllm.yaml"),
        "--host",
        str(CONFIGS / "hosts" / "example.yaml"),
        "--wheel",
        str(wheel),
        code=2,
    )
    assert "require --execute" in out


def test_gpu_doctor_on_laptop() -> None:
    out = run("gpu", "doctor")
    assert "nvidia-smi" in out


def test_backend_check_and_quality_compare(tmp_path: Path) -> None:
    out = run("backend", "check", str(CONFIGS / "backends" / "mock_fast.yaml"))
    assert "infer: ok" in out and "stream: ok" in out
    report = tmp_path / "q.json"
    run(
        "quality",
        "compare",
        "--reference",
        str(CONFIGS / "backends" / "sim_a.yaml"),
        "--candidate",
        str(CONFIGS / "backends" / "sim_b.yaml"),
        "--max-tokens",
        "8",
        "--output",
        str(report),
    )
    data = json.loads(report.read_text(encoding="utf-8"))
    assert (
        data["exact_match_rate"] == 1.0
    )  # same simulator vocabulary/seeding: identical greedy output
