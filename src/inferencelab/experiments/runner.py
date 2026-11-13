"""Execute local experiment plans and build the experiment report.

Remote (GPU) plans are executed by :mod:`inferencelab.remote.campaign`; this module
runs local plans against in-process mock/simulated backends, merges each target's
load sweep into one :class:`PerformanceProfile`, and renders comparison tables and
sweep plots.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from inferencelab.benchmarking.config import BackendTargetConfig, BenchmarkConfig
from inferencelab.benchmarking.runner import RunResult, make_run_id, run_benchmark
from inferencelab.config.loader import load_backend_config
from inferencelab.experiments.plan import ExperimentPlan, PlannedRun
from inferencelab.metrics.summary import SIMULATED_LABEL, RunSummary
from inferencelab.profiles.model import PerformanceProfile
from inferencelab.profiles.registry import ProfileRegistry
from inferencelab.reporting import plots as plotlib
from inferencelab.reporting.tables import RunRow, comparison_table, prefix_cache_table

__all__ = ["ExperimentResult", "render_experiment_report", "run_local_experiment"]

Done = list[tuple[PlannedRun, RunResult]]


@dataclass(slots=True)
class ExperimentResult:
    experiment_id: str
    out_dir: Path
    runs: Done
    profiles: dict[str, PerformanceProfile]
    report: Path


async def run_local_experiment(
    plan: ExperimentPlan, *, output_root: Path, plots: bool = True
) -> ExperimentResult:
    spec = plan.spec
    if spec.mode != "local":
        raise ValueError("remote experiments run through `inferencelab remote run`")
    experiment_id = make_run_id(spec.name)
    out_dir = output_root / experiment_id
    runs_dir = out_dir / "runs"
    done: Done = []
    for run in plan.runs:
        config = BenchmarkConfig(
            name=run.label,
            target=BackendTargetConfig(backend=load_backend_config(run.target_path)),
            workload=run.workload,
            slo=spec.slo,
            virtual_time=spec.virtual_time,
            output_dir=str(runs_dir),
            tags={"experiment": experiment_id, "target": run.target, "load": str(run.load_value)},
        )
        result = await run_benchmark(
            config, base_dir=run.workload_path.parent, output_root=runs_dir, plots=False
        )
        done.append((run, result))
    profiles = _merge_profiles(done)
    registry = ProfileRegistry(out_dir / "profiles")
    for profile in profiles.values():
        registry.save(profile)
    images = await asyncio.to_thread(_sweep_plots, out_dir / "plots", done) if plots else []
    report = out_dir / "report.md"
    text = render_experiment_report(spec.name, experiment_id, done, images, out_dir)
    report.write_text(text, encoding="utf-8")
    return ExperimentResult(experiment_id, out_dir, done, profiles, report)


def _merge_profiles(done: Done) -> dict[str, PerformanceProfile]:
    merged: dict[str, PerformanceProfile] = {}
    for run, result in done:
        key = f"{run.target}--{run.workload.name}"
        profile = result.profile.model_copy(update={"profile_id": key})
        profile.deployment = {**profile.deployment, "name": run.target}
        merged[key] = merged[key].merge(profile) if key in merged else profile
    return merged


def _throughput(s: RunSummary) -> float | None:
    return s.throughput.output_token_throughput


def _ttft_p95(s: RunSummary) -> float | None:
    return s.latency["ttft_ms"].p95


def _tpot_p95(s: RunSummary) -> float | None:
    return s.latency["tpot_ms"].p95


def _goodput(s: RunSummary) -> float | None:
    return s.throughput.goodput_rps


_METRICS: tuple[tuple[str, str, Callable[[RunSummary], float | None]], ...] = (
    ("throughput", "output tokens/s", _throughput),
    ("ttft_p95", "TTFT p95 (ms)", _ttft_p95),
    ("tpot_p95", "TPOT p95 (ms)", _tpot_p95),
    ("goodput", "goodput (SLO-compliant req/s)", _goodput),
)
_LOADS = (
    ("concurrency", "concurrency (closed loop)"),
    ("request_rate", "offered load (req/s)"),
)


def _sweep_plots(directory: Path, done: Done) -> list[Path]:
    if not plotlib.plots_available():
        return []
    targets = sorted({run.target for run, _ in done})
    colors = plotlib.assign_colors(targets) if len(targets) <= len(plotlib.SERIES_COLORS) else None
    by_workload: dict[str, Done] = defaultdict(list)
    for run, result in done:
        by_workload[run.workload.name].append((run, result))
    note = SIMULATED_LABEL if all(r.summary.simulated for _, r in done) else None
    images: list[Path] = []
    for wname, rows in by_workload.items():
        for kind, xlabel in _LOADS:
            pts = [(r, res) for r, res in rows if r.load_kind == kind and r.load_value is not None]
            if not pts:
                continue
            for key, ylabel, getter in _METRICS:
                series: dict[str, tuple[list[float], list[float | None]]] = {}
                for run, res in pts:
                    xs, ys = series.setdefault(run.target, ([], []))
                    xs.append(float(run.load_value or 0))
                    ys.append(getter(res.summary))
                if all(v is None for _, ys in series.values() for v in ys):
                    continue
                path = plotlib.sweep(
                    series,
                    path=directory / f"{wname}_{key}_vs_{kind}.png",
                    title=f"{ylabel} vs {xlabel} — {wname}",
                    xlabel=xlabel,
                    ylabel=ylabel,
                    colors=colors,
                    note=note,
                    log_x=kind == "concurrency",
                )
                if path:
                    images.append(path)
    return images


def _config_label(run: PlannedRun) -> str:
    if run.load_value is None:
        return run.workload.name
    return f"{run.workload.name} · {run.load_kind} {run.load_value:g}"


def render_experiment_report(
    name: str, experiment_id: str, done: Done, images: list[Path], out_dir: Path
) -> str:
    lines = [f"# Experiment {name}", ""]
    if all(r.summary.simulated for _, r in done):
        lines += [
            "> **SIMULATED BACKEND RESULTS** — CPU mock/simulated backends. These numbers "
            "exercise the methodology and tooling; they are not GPU inference measurements.",
            "",
        ]
    lines += [f"Experiment `{experiment_id}` · {len(done)} runs", "", "## Comparison", ""]
    rows = [
        RunRow(
            backend=run.target,
            config=_config_label(run),
            summary=res.summary,
            gpu_memory_gb=(res.metadata.get("gpu_memory") or {}).get("peak_gb"),
        )
        for run, res in done
    ]
    lines += [comparison_table(rows), ""]
    prefix_rows = [
        row
        for row, (run, _) in zip(rows, done, strict=True)
        if run.workload.shared_prefix or run.workload.sessions
    ]
    if prefix_rows:
        lines += ["## Prefix-cache workloads", "", prefix_cache_table(prefix_rows), ""]
    if images:
        lines += ["## Plots", ""]
        lines += [f"![{p.stem}]({p.relative_to(out_dir).as_posix()})\n" for p in images]
    lines += ["## Runs", ""]
    for run, res in done:
        rel = res.run_dir.relative_to(out_dir).as_posix() if res.run_dir else "(not written)"
        slo = res.summary.slo.status if res.summary.slo else "n/a"
        lines.append(f"- `{run.label}` → `{rel}/report.md` (SLO: {slo})")
    lines.append("")
    return "\n".join(lines)
