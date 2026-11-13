"""Benchmark runner: build target → snapshot engine metrics → load → summarise → artifacts."""

from __future__ import annotations

import asyncio
import re
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, assert_never

import httpx

from inferencelab import __version__
from inferencelab.backends.base import BackendMetrics
from inferencelab.backends.factory import create_backend
from inferencelab.backends.openai_compatible.prom import (
    INFERENCELAB_SERVER_METRICS,
    MetricMap,
    build_metrics,
    engine_phase_means,
    parse_prometheus_text,
)
from inferencelab.benchmarking.artifacts import write_json, write_records, write_yaml
from inferencelab.benchmarking.config import (
    BackendTargetConfig,
    BenchmarkConfig,
    GatewayTargetConfig,
    HTTPTargetConfig,
    classify_backend_types,
)
from inferencelab.benchmarking.gpu_monitor import GpuMemorySampler
from inferencelab.benchmarking.loadgen import LoadGenerator
from inferencelab.benchmarking.system_info import collect_system_info, git_state
from inferencelab.benchmarking.targets import (
    BackendTarget,
    BenchmarkTarget,
    HTTPTarget,
    ServiceTarget,
)
from inferencelab.clock import Clock, SystemClock, VirtualClock
from inferencelab.cost.model import compute_cost
from inferencelab.domain.result import RequestRecord
from inferencelab.domain.workload import ClosedLoopArrival, ConstantRateArrival, PoissonArrival
from inferencelab.gateway.runtime import GatewayRuntime
from inferencelab.metrics.summary import RunSummary, summarize_records
from inferencelab.profiles.model import LoadLevel, PercentileSet, PerformanceProfile, ProfilePoint
from inferencelab.reporting import plots as plotlib
from inferencelab.reporting.report import render_run_report
from inferencelab.reporting.timeseries import concurrency_timeline
from inferencelab.workloads.generator import WorkloadPlan

__all__ = ["RunResult", "build_profile", "make_run_id", "run_benchmark"]


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    run_dir: Path | None
    summary: RunSummary
    profile: PerformanceProfile
    records: list[RequestRecord]
    metadata: dict[str, Any]


def make_run_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "run"
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{slug}-{secrets.token_hex(2)}"


@dataclass(slots=True)
class _Built:
    target: BenchmarkTarget
    cleanup: Callable[[], Awaitable[None]]
    snapshot: Callable[[], Awaitable[dict[str, BackendMetrics]]]
    phase_metrics: Mapping[str, str]
    simulated: bool | None
    info: dict[str, Any] = field(default_factory=dict)


def _metric_map(engine: str | None) -> tuple[MetricMap, Mapping[str, str]]:
    if engine == "vllm":
        from inferencelab.backends.vllm.backend import VLLMBackend

        return VLLMBackend.metric_map, VLLMBackend.phase_metrics
    if engine == "sglang":
        from inferencelab.backends.sglang.backend import SGLangBackend

        return SGLangBackend.metric_map, SGLangBackend.phase_metrics
    if engine == "tensorrt_llm":
        from inferencelab.backends.tensorrt_llm.backend import TensorRTLLMBackend

        return TensorRTLLMBackend.metric_map, TensorRTLLMBackend.phase_metrics
    return INFERENCELAB_SERVER_METRICS, {}


async def _probe_http(base_url: str) -> dict[str, Any]:
    """Ask the target what it is: an InferenceLab gateway lists its backends."""
    info: dict[str, Any] = {}
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        try:
            resp = await client.get("/inferencelab/backends")
            if resp.status_code == 200:
                info["gateway_backends"] = [
                    {"name": b["name"], "type": b["type"], "deployment": b.get("deployment")}
                    for b in resp.json().get("backends", [])
                ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            pass
        for path in ("/version", "/server_info"):
            try:
                resp = await client.get(path)
                if resp.status_code == 200 and isinstance(resp.json(), dict):
                    data = resp.json()
                    info["engine_version"] = data.get("version")
                    if isinstance(data.get("engine"), str):
                        info["engine"] = data["engine"]
                    break
            except (httpx.HTTPError, ValueError):
                continue
    return info


def _simulated_from_probe(info: Mapping[str, Any]) -> bool | None:
    backends = info.get("gateway_backends")
    if backends:
        return classify_backend_types({str(b["type"]) for b in backends})
    engine = info.get("engine")
    if isinstance(engine, str) and engine.startswith("inferencelab-"):
        return True
    return None


async def _build(config: BenchmarkConfig, clock: Clock) -> _Built:
    target_cfg = config.target
    if isinstance(target_cfg, HTTPTargetConfig):
        target = HTTPTarget(
            target_cfg.base_url,
            api_key_env=target_cfg.api_key_env,
            headers=target_cfg.headers,
            max_connections=target_cfg.max_connections,
            connect_timeout_s=target_cfg.connect_timeout_s,
        )
        info = await _probe_http(target_cfg.base_url)
        mapping, phases = _metric_map(target_cfg.engine)
        metrics_url = target_cfg.metrics_url

        async def snapshot() -> dict[str, BackendMetrics]:
            if metrics_url is None:
                return {}
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(metrics_url)
                if resp.status_code != 200:
                    return {}
                samples = parse_prometheus_text(resp.text)
            except (httpx.HTTPError, ValueError):
                return {}
            return {"engine": build_metrics(samples, mapping, source=f"prometheus:{metrics_url}")}

        return _Built(target, target.aclose, snapshot, phases, _simulated_from_probe(info), info)

    if isinstance(target_cfg, BackendTargetConfig):
        backend = create_backend(target_cfg.backend, clock=clock)
        phases = getattr(backend, "phase_metrics", {})

        async def snapshot_backend() -> dict[str, BackendMetrics]:
            return {backend.name: await backend.metrics()}

        return _Built(
            BackendTarget(backend),
            backend.aclose,
            snapshot_backend,
            phases,
            classify_backend_types({target_cfg.backend.type}),
            {"backend": {"name": backend.name, "type": target_cfg.backend.type}},
        )

    if not isinstance(target_cfg, GatewayTargetConfig):
        assert_never(target_cfg)  # the target union is exhaustive
    runtime = GatewayRuntime(target_cfg.gateway, clock=clock)
    await runtime.start()

    async def snapshot_gateway() -> dict[str, BackendMetrics]:
        return {h.name: await h.backend.metrics() for h in runtime.pool.handles}

    return _Built(
        ServiceTarget(runtime.service),
        runtime.stop,
        snapshot_gateway,
        {},
        config.local_backend_types(),
        {
            "gateway_backends": [
                {"name": b.name, "type": b.type, "deployment": b.deployment}
                for b in target_cfg.gateway.backends
            ],
            "routing_policy": target_cfg.gateway.routing.policy,
        },
    )


def engine_window(
    before: Mapping[str, BackendMetrics],
    after: Mapping[str, BackendMetrics],
    phase_metrics: Mapping[str, str],
) -> dict[str, float]:
    """Engine-reported activity *during* the run (counter deltas), summed over backends."""
    out: dict[str, float] = {}

    def delta(attr: str) -> float | None:
        total = None
        for name, a in after.items():
            b = before.get(name)
            va, vb = getattr(a, attr), getattr(b, attr) if b is not None else None
            if va is None or vb is None:
                continue
            total = (total or 0.0) + (va - vb)
        return total

    queries, hits = delta("prefix_cache_queries"), delta("prefix_cache_hits")
    if queries and hits is not None:
        out["prefix_cache_hit_rate"] = hits / queries
    else:
        rates = [
            m.prefix_cache_hit_rate for m in after.values() if m.prefix_cache_hit_rate is not None
        ]
        if rates:
            out["prefix_cache_hit_rate_gauge"] = sum(rates) / len(rates)
    for attr, key in (
        ("preemptions_total", "preemptions"),
        ("generation_tokens_total", "engine_generation_tokens"),
        ("prompt_tokens_total", "engine_prompt_tokens_computed"),
        ("spec_decode_accepted_tokens", "spec_accepted_tokens"),
        ("spec_decode_draft_tokens", "spec_draft_tokens"),
    ):
        d = delta(attr)
        if d is not None:
            out[key] = d
    if "spec_draft_tokens" in out and out["spec_draft_tokens"] > 0:
        out["spec_acceptance_rate"] = (
            out.get("spec_accepted_tokens", 0.0) / out["spec_draft_tokens"]
        )
    kv = [m.kv_cache_usage for m in after.values() if m.kv_cache_usage is not None]
    if kv:
        out["kv_cache_usage_end"] = max(kv)
    for name, a in after.items():
        b = before.get(name)
        if b is not None and phase_metrics:
            for phase, mean in engine_phase_means(phase_metrics, b.raw, a.raw).items():
                out[f"engine_mean_{phase}_ms"] = mean * 1000.0
    return out


def build_profile(
    config: BenchmarkConfig, summary: RunSummary, metadata: Mapping[str, Any]
) -> PerformanceProfile:
    arrival = config.workload.arrival
    if isinstance(arrival, ClosedLoopArrival):
        load = LoadLevel(kind="concurrency", value=float(arrival.concurrency))
    elif isinstance(arrival, ConstantRateArrival | PoissonArrival):
        load = LoadLevel(kind="request_rate", value=arrival.rate)
    else:
        offered = (
            summary.counts.measured / summary.throughput.duration_s
            if summary.throughput.duration_s
            else 0.0
        )
        load = LoadLevel(kind="request_rate", value=round(offered, 4))

    def pct(key: str) -> PercentileSet:
        d = summary.latency[key]
        return PercentileSet(p50=d.p50, p90=d.p90, p95=d.p95, p99=d.p99, mean=d.mean)

    point = ProfilePoint(
        load=load,
        run_id=summary.run_id,
        requests=summary.counts.measured,
        request_throughput=summary.throughput.request_throughput,
        input_token_throughput=summary.throughput.input_token_throughput,
        output_token_throughput=summary.throughput.output_token_throughput,
        goodput_rps=summary.throughput.goodput_rps,
        slo_attainment=summary.slo.attainment if summary.slo else None,
        error_rate=summary.errors.error_rate,
        ttft_ms=pct("ttft_ms"),
        tpot_ms=pct("tpot_ms"),
        itl_ms=pct("itl_ms"),
        e2e_ms=pct("e2e_ms"),
        mean_input_tokens=summary.tokens.mean_input_tokens,
        mean_output_tokens=summary.tokens.mean_output_tokens,
        gpu_memory_peak_gb=(metadata.get("gpu_memory") or {}).get("peak_gb"),
    )
    hw = config.hardware_ref
    dep = config.deployment
    target_name = (
        dep.name
        if dep
        else config.target.backend.name
        if isinstance(config.target, BackendTargetConfig)
        else config.name
    )
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{target_name}--{config.workload.name}")
    return PerformanceProfile(
        profile_id=slug,
        simulated=bool(summary.simulated),
        hardware={"gpu": hw.gpu if hw else "cpu", "count": hw.count if hw else 0},
        model={
            "name": config.model.name if config.model else config.workload.model,
            "checkpoint": config.model.checkpoint if config.model else None,
            "revision": config.model.revision if config.model else None,
        },
        backend={
            "kind": dep.backend.value if dep else metadata.get("target_kind"),
            "version": metadata.get("engine_version"),
            "image": dep.image if dep else None,
        },
        deployment={
            "name": target_name,
            "quantization": dep.quantization.value if dep else None,
            "tensor_parallel": dep.parallelism.tensor if dep else None,
            "prefix_caching": dep.prefix_caching if dep else None,
            "chunked_prefill": dep.chunked_prefill if dep else None,
        },
        workload={
            "name": config.workload.name,
            "arrival": config.workload.arrival.kind,
            "mean_input_tokens": summary.tokens.mean_input_tokens,
            "mean_output_tokens": summary.tokens.mean_output_tokens,
        },
        slo=config.slo,
        points=[point],
        notes=["SIMULATED: CPU model, not GPU measurements"] if summary.simulated else [],
    )


async def _measure(
    config: BenchmarkConfig,
    clock: Clock,
    plan: WorkloadPlan,
    run_id: str,
    target: BenchmarkTarget | None,
) -> tuple[list[RequestRecord], float, dict[str, Any], dict[str, float], bool | None]:
    built = await _build(config, clock) if target is None else None
    tgt = target if target is not None else built.target  # type: ignore[union-attr]
    sampler = GpuMemorySampler() if config.execution == "gpu" else None
    try:
        before = await built.snapshot() if built else {}
        if sampler is not None:
            await sampler.__aenter__()
        loadgen = LoadGenerator(
            tgt, plan, clock=clock, run_tag=run_id[-4:] + "r", capture_text=config.capture_text
        )
        result = await loadgen.run()
        after = await built.snapshot() if built else {}
    finally:
        if sampler is not None:
            await sampler.__aexit__(None, None, None)
        if built is not None:
            await built.cleanup()
    engine = engine_window(before, after, built.phase_metrics) if built else {}
    info = dict(built.info) if built else {"target": getattr(tgt, "description", "custom")}
    if sampler is not None:
        info["gpu_memory"] = sampler.summary()
    simulated = built.simulated if built else None
    return result.records, result.duration_s, info, engine, simulated


def _write_outputs(
    run_dir: Path,
    config: BenchmarkConfig,
    records: Sequence[RequestRecord],
    summary: RunSummary,
    profile: PerformanceProfile,
    metadata: dict[str, Any],
    *,
    plots: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config.yaml", config.model_dump(mode="json", exclude_none=True))
    write_json(run_dir / "metadata.json", metadata)
    write_records(run_dir / "requests.parquet", records)
    write_json(run_dir / "summary.json", summary.model_dump(mode="json"))
    write_json(run_dir / "profile.json", profile.model_dump(mode="json"))
    write_json(run_dir / "system.json", collect_system_info())
    images: list[Path] = []
    if plots and plotlib.plots_available():
        images = _run_plots(run_dir / "plots", records, summary)
    (run_dir / "report.md").write_text(
        render_run_report(summary, metadata, plots=images, run_dir=run_dir), encoding="utf-8"
    )


def _run_plots(
    directory: Path, records: Sequence[RequestRecord], summary: RunSummary
) -> list[Path]:
    from inferencelab.domain.result import Phase

    measured = [r for r in records if r.phase is Phase.MEASURE and r.ok]
    note = summary.label
    out: list[Path] = []
    by_backend: dict[str, list[float]] = {}
    for r in measured:
        if r.ttft_s is not None:
            by_backend.setdefault(r.backend or "target", []).append(r.ttft_s * 1000)
    if len(by_backend) <= 8:
        p = plotlib.latency_cdf(
            by_backend,
            path=directory / "ttft_cdf.png",
            title="TTFT distribution",
            xlabel="TTFT (ms)",
            note=note,
        )
        if p:
            out.append(p)
    e2e = [r.e2e_s * 1000 for r in measured if r.e2e_s is not None]
    p = plotlib.latency_cdf(
        {"E2E": e2e},
        path=directory / "e2e_cdf.png",
        title="End-to-end latency distribution",
        xlabel="E2E (ms)",
        note=note,
    )
    if p:
        out.append(p)
    t, series = concurrency_timeline(records)
    if t:
        p = plotlib.timeline(
            t,
            series,
            path=directory / "queue_depth.png",
            title="In-flight requests" + (" and gateway queue depth" if len(series) > 1 else ""),
            ylabel="requests",
            note=note,
        )
        if p:
            out.append(p)
    return out


async def run_benchmark(
    config: BenchmarkConfig,
    *,
    base_dir: Path | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    target: BenchmarkTarget | None = None,
    write: bool = True,
    plots: bool = True,
) -> RunResult:
    run_id = run_id or make_run_id(config.name)
    clock: Clock = VirtualClock() if config.virtual_time else SystemClock()
    plan = WorkloadPlan(config.workload, base_dir=base_dir)
    started = datetime.now(UTC)
    measure = _measure(config, clock, plan, run_id, target)
    if isinstance(clock, VirtualClock):
        records, duration, info, engine, simulated = await clock.run(measure)
    else:
        records, duration, info, engine, simulated = await measure
    if config.simulated is not None:
        simulated = config.simulated
    summary = summarize_records(
        records,
        run_id=run_id,
        simulated=simulated,
        slo=config.slo,
        min_tail=config.min_tail_samples,
    )
    summary.engine = engine
    hw = config.hardware_ref
    if hw is not None and hw.pricing is not None and summary.throughput.duration_s > 0:
        compliant = None
        if summary.slo is not None and summary.throughput.goodput_rps is not None:
            compliant = round(summary.throughput.goodput_rps * summary.throughput.duration_s)
        cost = compute_cost(
            hw.pricing,
            gpu_count=max(1, hw.count),
            duration_s=summary.throughput.duration_s,
            input_tokens=summary.tokens.input_tokens,
            output_tokens=summary.tokens.output_tokens,
            successful_requests=summary.counts.ok,
            slo_compliant_requests=compliant,
        )
        summary.cost = cost.model_dump(mode="json")
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "name": config.name,
        "description": config.description,
        "created_at": started.isoformat(timespec="seconds"),
        "inferencelab_version": __version__,
        "git": await asyncio.to_thread(git_state) if write else {},
        "label": summary.label,
        "simulated": simulated,
        "execution": config.execution,
        "clock": "virtual" if config.virtual_time else "system",
        "wall_duration_s": duration,
        "target_kind": config.target.type,
        "target": info,
        "engine_version": info.get("engine_version"),
        "deployment": _deployment_meta(config),
        "gpu_memory": info.pop("gpu_memory", None),
        "hardware": None if hw is None else hw.model_dump(mode="json", exclude_none=True),
        "model": None
        if config.model is None
        else config.model.model_dump(mode="json", exclude_none=True),
        "workload": {
            "name": config.workload.name,
            "api": config.workload.api,
            "arrival": config.workload.arrival.model_dump(mode="json"),
            "input_tokens": config.workload.input_tokens.model_dump(mode="json"),
            "output_tokens": config.workload.output_tokens.model_dump(mode="json"),
            "shared_prefix": config.workload.shared_prefix is not None,
            "sessions": config.workload.sessions is not None,
            "seed": config.workload.seed,
        },
        "privacy": {
            "capture_text": config.capture_text,
            "prompt_fingerprint": "HMAC-SHA256 with a per-run random salt that is not persisted",
        },
        "tags": config.tags,
    }
    profile = build_profile(config, summary, metadata)
    run_dir = None
    if write:
        root = output_root or Path(config.output_dir)
        run_dir = root / run_id
        await asyncio.to_thread(
            _write_outputs, run_dir, config, records, summary, profile, metadata, plots=plots
        )
    return RunResult(run_id, run_dir, summary, profile, records, metadata)


def _deployment_meta(config: BenchmarkConfig) -> dict[str, Any] | None:
    if config.deployment is not None:
        return config.deployment.model_dump(mode="json", exclude_none=True)
    if isinstance(config.target, BackendTargetConfig):
        b = config.target.backend
        return {"backend": b.name, "type": b.type}
    if isinstance(config.target, GatewayTargetConfig):
        g = config.target.gateway
        return {
            "gateway_routing": g.routing.policy,
            "backends": ", ".join(f"{b.name} ({b.type})" for b in g.backends),
        }
    return None
