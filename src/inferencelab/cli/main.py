"""InferenceLab command-line interface."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import platform
import shutil
import subprocess  # fixed argv only  # nosec B404
import sys
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any, TypeVar

import psutil
import typer
from rich.console import Console
from rich.table import Table

from inferencelab import __version__

T = TypeVar("T")

app = typer.Typer(
    name="inferencelab",
    help="Reproducible LLM inference engineering: serving, routing, benchmarking, SLOs, capacity.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
backend_app = typer.Typer(
    help="Inference backends: list, check, serve locally, launch plans.", no_args_is_help=True
)
model_app = typer.Typer(help="Model registry.", no_args_is_help=True)
gateway_app = typer.Typer(help="OpenAI-compatible gateway.", no_args_is_help=True)
workload_app = typer.Typer(help="Workload definitions.", no_args_is_help=True)
benchmark_app = typer.Typer(help="Benchmark runs and comparisons.", no_args_is_help=True)
profile_app = typer.Typer(help="Performance profiles.", no_args_is_help=True)
capacity_app = typer.Typer(help="Capacity planning from measured profiles.", no_args_is_help=True)
autoscale_app = typer.Typer(help="Autoscaling simulation (CPU only).", no_args_is_help=True)
experiment_app = typer.Typer(
    help="Experiment matrices (capped, cost-guarded).", no_args_is_help=True
)
remote_app = typer.Typer(help="Cloud-neutral remote GPU execution over SSH.", no_args_is_help=True)
gpu_app = typer.Typer(help="Local GPU checks.", no_args_is_help=True)
quality_app = typer.Typer(help="Quality / equivalence validation.", no_args_is_help=True)
for sub, name in (
    (backend_app, "backend"),
    (model_app, "model"),
    (gateway_app, "gateway"),
    (workload_app, "workload"),
    (benchmark_app, "benchmark"),
    (profile_app, "profile"),
    (capacity_app, "capacity"),
    (autoscale_app, "autoscale"),
    (experiment_app, "experiment"),
    (remote_app, "remote"),
    (gpu_app, "gpu"),
    (quality_app, "quality"),
):
    app.add_typer(sub, name=name)

console = Console()
ConfigArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="YAML config file")]


def _run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def _fail(message: str, code: int = 1) -> typer.Exit:
    console.print(f"[red]error:[/red] {message}")
    return typer.Exit(code)


def _table(title: str, columns: list[str], rows: list[list[str]]) -> Table:
    table = Table(title=title, title_justify="left", header_style="bold")
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    return table


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    """InferenceLab command-line interface."""


# ============================================================================ doctor
def _probe(argv: list[str], timeout: float = 5.0) -> str | None:
    exe = shutil.which(argv[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell  # nosec B603
            [exe, *argv[1:]], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (
        (out.stdout or out.stderr).strip().splitlines()[0]
        if (out.stdout or out.stderr).strip()
        else ""
    )


@app.command()
def doctor() -> None:
    """Check the local development environment (no GPU is required or expected)."""
    rows: list[list[str]] = []

    def add(name: str, status: str, detail: str) -> None:
        colour = {"ok": "green", "warn": "yellow", "fail": "red", "info": "cyan"}[status]
        rows.append([name, f"[{colour}]{status}[/{colour}]", detail])

    py_ok = sys.version_info >= (3, 11)
    add("python", "ok" if py_ok else "fail", sys.version.split()[0])
    add("os", "info", f"{platform.system()} {platform.release()} ({platform.machine()})")
    add(
        "cpu",
        "info",
        f"{psutil.cpu_count(logical=True)} logical / {psutil.cpu_count(logical=False)} physical",
    )
    ram = psutil.virtual_memory().total / 1e9
    add("ram", "ok" if ram >= 8 else "warn", f"{ram:.1f} GB")
    free = shutil.disk_usage(Path.cwd()).free / 1e9
    add("disk", "ok" if free >= 20 else "warn", f"{free:.0f} GB free here")
    for tool in ("uv", "git", "ssh", "scp", "curl"):
        out = _probe([tool, "--version"] if tool not in ("ssh", "scp") else [tool, "-V"])
        add(tool, "ok" if out is not None else "warn", out or "not found")
    docker = _probe(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=10)
    if docker is None:
        add("docker", "warn", "CLI not found (Docker builds/Compose demo unavailable)")
    elif not docker or "error" in docker.lower() or "cannot" in docker.lower():
        add("docker", "warn", "CLI present but the daemon is not reachable")
    else:
        add("docker", "ok", f"server {docker}")
    for extra, pkg in (("plots", "matplotlib"), ("tracing", "opentelemetry.sdk")):
        present = importlib.util.find_spec(pkg) is not None
        add(
            f"extra:{extra}",
            "ok" if present else "warn",
            "installed" if present else f"pip install 'inferencelab[{extra}]'",
        )
    prom = shutil.which("prometheus")
    add("prometheus", "info", (prom and "binary found") or "optional (use docker compose)")
    nvsmi = shutil.which("nvidia-smi")
    add(
        "gpu runtime",
        "info",
        "nvidia-smi found" if nvsmi else "none - normal for the CPU development laptop",
    )
    console.print(_table(f"InferenceLab {__version__} doctor", ["check", "status", "detail"], rows))


# ============================================================================ demo
@app.command()
def demo(
    quick: Annotated[bool, typer.Option(help="Smaller workloads (~30 s).")] = False,
    output: Annotated[Path, typer.Option(help="Results directory.")] = Path("results"),
    publish: Annotated[Path | None, typer.Option(help="Also copy report + plots here.")] = None,
) -> None:
    """Run the CPU-only demo: simulated engines + gateway + load + report (no GPU, no internet)."""
    from inferencelab.demo import run_demo

    start = time.perf_counter()
    result = _run(
        run_demo(
            output,
            quick=quick,
            publish_dir=publish,
            progress=lambda m: console.print(f"[cyan]-[/cyan] {m}"),
        )
    )
    console.print(
        f"\n[bold]SIMULATED BACKEND RESULTS[/bold] - report: {result.report} ({time.perf_counter() - start:.0f}s)"
    )


# ============================================================================ backend
@backend_app.command("list")
def backend_list(
    config_dir: Annotated[Path, typer.Option(help="Directory with backend YAML configs")] = Path(
        "configs/backends"
    ),
) -> None:
    """Show the capability matrix of backend kinds and the configured backends."""
    from inferencelab.backends.config import (
        BackendConfig,
        MockBackendConfig,
        OpenAIBackendConfig,
        SGLangBackendConfig,
        SimulatedBackendConfig,
        TensorRTLLMBackendConfig,
        VLLMBackendConfig,
    )
    from inferencelab.backends.factory import create_backend
    from inferencelab.config.loader import load_backend_config

    configs: list[BackendConfig] = [
        MockBackendConfig(name="mock"),
        SimulatedBackendConfig(name="simulated"),
        OpenAIBackendConfig(name="openai", base_url="http://127.0.0.1:1"),
        VLLMBackendConfig(name="vllm", base_url="http://127.0.0.1:1"),
        SGLangBackendConfig(name="sglang", base_url="http://127.0.0.1:1"),
        TensorRTLLMBackendConfig(name="tensorrt_llm", base_url="http://127.0.0.1:1"),
    ]

    async def caps() -> list[dict[str, Any]]:
        out = []
        for cfg in configs:
            backend = create_backend(cfg)
            out.append(backend.capabilities.as_dict())
            await backend.aclose()
        return out

    matrix = _run(caps())
    features = [k for k in matrix[0] if k != "quantization"]
    rows = [[f] + ["✓" if m[f] else "·" for m in matrix] for f in features]
    rows.append(["quantization"] + [",".join(m["quantization"]) or "·" for m in matrix])
    console.print(_table("Backend capabilities", ["feature"] + [c.name for c in configs], rows))
    if config_dir.exists():
        found = []
        for path in sorted(config_dir.glob("*.yaml")):
            try:
                cfg = load_backend_config(path)
                found.append([path.name, cfg.name, cfg.type, ", ".join(cfg.models)])
            except Exception as exc:  # show broken configs instead of crashing the listing
                found.append([path.name, "-", "[red]invalid[/red]", str(exc).splitlines()[0][:60]])
        console.print(
            _table(f"Configured backends ({config_dir})", ["file", "name", "type", "models"], found)
        )


@backend_app.command("check")
def backend_check(config: ConfigArg) -> None:
    """Health, models, one request and one streamed request against a backend."""
    from inferencelab.backends.base import BackendRequest
    from inferencelab.backends.factory import create_backend
    from inferencelab.config.loader import load_backend_config
    from inferencelab.domain.errors import InferenceError
    from inferencelab.domain.request import ChatMessage, RequestKind, SamplingParams

    cfg = load_backend_config(config)

    async def check() -> int:
        backend = create_backend(cfg)
        failures = 0
        try:
            health = await backend.health()
            console.print(f"health: {health.status.value} {health.detail or ''}")
            failures += not health.healthy
            model = cfg.served_model(cfg.models[0])
            req = BackendRequest(
                request_id="inferencelab-check",
                model=model,
                kind=RequestKind.CHAT,
                sampling=SamplingParams(max_tokens=16, temperature=0.0),
                messages=(ChatMessage("user", "Say hello in five words."),),
            )
            try:
                resp = await backend.infer(req)
                console.print(
                    f"infer: ok, {resp.usage.output_tokens} output tokens ({resp.usage.source.value})"
                )
            except InferenceError as exc:
                failures += 1
                console.print(f"[red]infer failed[/red]: {exc.kind.value}: {exc.message}")
            try:
                t0 = time.perf_counter()
                first = None
                chunks = 0
                async for chunk in backend.stream(req):
                    chunks += 1
                    if chunk.text and first is None:
                        first = time.perf_counter() - t0
                ttft = f"{first * 1000:.1f} ms" if first is not None else "n/a"
                console.print(f"stream: ok, {chunks} chunks, probe TTFT {ttft}")
            except InferenceError as exc:
                failures += 1
                console.print(f"[red]stream failed[/red]: {exc.kind.value}: {exc.message}")
            metrics = await backend.metrics()
            console.print(f"engine metrics: {metrics.source}")
            version = getattr(backend, "version", None)
            if version is not None:
                console.print(f"engine version: {await version()}")
            caps = [k for k, v in backend.capabilities.as_dict().items() if v is True]
            console.print(f"capabilities: {', '.join(caps)}")
        finally:
            await backend.aclose()
        return failures

    if _run(check()):
        raise typer.Exit(1)


@backend_app.command("serve")
def backend_serve(
    config: ConfigArg,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 9001,
    admin: Annotated[bool, typer.Option(help="Expose /admin/fault for fault injection.")] = False,
) -> None:
    """Serve a mock/simulated backend over an OpenAI-compatible HTTP API."""
    import uvicorn

    from inferencelab.backends.factory import create_backend
    from inferencelab.backends.server import create_backend_app
    from inferencelab.config.loader import load_backend_config

    cfg = load_backend_config(config)
    if cfg.type not in ("mock", "simulated"):
        raise _fail(
            "only mock/simulated backends can be served locally; real engines run in their own containers"
        )
    app_ = create_backend_app(create_backend(cfg), models=cfg.models, kind=cfg.type, admin=admin)
    uvicorn.run(app_, host=host, port=port, log_level="warning", access_log=False)


@backend_app.command("launch-plan")
def backend_launch_plan(
    deployment: ConfigArg,
    models_dir: Annotated[Path, typer.Option(help="Model registry directory")] = Path(
        "configs/models"
    ),
    hardware_dir: Annotated[Path, typer.Option(help="Hardware profiles")] = Path(
        "configs/hardware"
    ),
) -> None:
    """Print the pinned `docker run` command for a GPU deployment (nothing is executed)."""
    from inferencelab.backends.launch import ENGINE_VERSIONS, build_launch_plan
    from inferencelab.config.loader import load_model_file
    from inferencelab.cost.hardware import HardwareRegistry, weight_fit
    from inferencelab.domain.deployment import DeploymentSpec

    dep = load_model_file(deployment, DeploymentSpec)
    models = _load_models(models_dir)
    if dep.model not in models:
        raise _fail(f"model {dep.model!r} not found in {models_dir}")
    plan = build_launch_plan(dep, models[dep.model])
    console.print(
        f"[bold]{dep.name}[/bold] ({dep.backend.value}, verified against v{ENGINE_VERSIONS.get(dep.backend, '?')})"
    )
    console.print(plan.describe())
    for name, content in plan.config_files.items():
        console.print(f"\n[bold]{name}[/bold]\n{content}")
    for cmd in plan.prepare:
        console.print(f"prepare: {' '.join(cmd)}")
    for note in plan.notes:
        console.print(f"[yellow]note:[/yellow] {note}")
    if hardware_dir.exists():
        try:
            hw = HardwareRegistry.from_dir(hardware_dir).get(dep.hardware.gpu)
            console.print(
                f"memory: {weight_fit(models[dep.model], hw, dep.hardware.count, dep.quantization).describe()}"
            )
        except KeyError as exc:
            console.print(f"[yellow]{exc}[/yellow]")


def _load_models(models_dir: Path) -> dict[str, Any]:
    from inferencelab.config.loader import load_model_file
    from inferencelab.domain.model import ModelSpec

    models = {}
    for path in sorted(models_dir.glob("*.yaml")):
        model = load_model_file(path, ModelSpec)
        models[model.name] = model
    return models


# ============================================================================ model
@model_app.command("list")
def model_list(
    models_dir: Annotated[Path, typer.Option(help="Model registry directory")] = Path(
        "configs/models"
    ),
) -> None:
    """List registered models (references only - weights are never stored here)."""
    rows = [
        [
            m.name,
            m.checkpoint,
            (m.revision[:12] + ("" if m.revision_pinned else " [yellow](unpinned)[/yellow]")),
            f"{m.parameter_count_b:g}B" if m.parameter_count_b else "?",
            str(m.context_length or "?"),
            m.quantization.value,
            m.license.id if m.license else "?",
        ]
        for m in _load_models(models_dir).values()
    ]
    console.print(
        _table(
            "Models",
            ["name", "checkpoint", "revision", "params", "context", "weights", "license"],
            rows,
        )
    )


# ============================================================================ gateway
@gateway_app.command("serve")
def gateway_serve(
    config: ConfigArg,
    host: Annotated[str | None, typer.Option(help="Override server.host")] = None,
    port: Annotated[int | None, typer.Option(help="Override server.port")] = None,
) -> None:
    """Run the OpenAI-compatible gateway."""
    import uvicorn

    from inferencelab.config.loader import load_model_file
    from inferencelab.gateway.app import create_app
    from inferencelab.gateway.config import GatewayConfig
    from inferencelab.observability.logging import configure_logging

    cfg = load_model_file(config, GatewayConfig)
    configure_logging(cfg.observability.log_level, json_logs=cfg.observability.log_json)
    uvicorn.run(
        create_app(cfg),
        host=host or cfg.server.host,
        port=port if port is not None else cfg.server.port,
        log_level="warning",
        access_log=False,
    )


# ============================================================================ workload
@workload_app.command("inspect")
def workload_inspect(
    config: ConfigArg,
    samples: Annotated[int, typer.Option(help="Sessions to sample")] = 1000,
) -> None:
    """Summarise a workload's request shapes and arrival process (no prompt text shown)."""
    from inferencelab.config.loader import load_model_file
    from inferencelab.domain.workload import WorkloadSpec
    from inferencelab.metrics.stats import Distribution
    from inferencelab.workloads.generator import WorkloadPlan

    spec = load_model_file(config, WorkloadSpec)
    plan = WorkloadPlan(spec, base_dir=config.parent)
    sessions = []
    for i, s in enumerate(plan.sessions()):
        if i >= samples:
            break
        sessions.append(s)
    turns = [t for s in sessions for t in s.turns]
    ins = Distribution.of([t.input_tokens for t in turns])
    outs = Distribution.of([t.max_tokens for t in turns])
    offsets = [s.start_offset_s for s in sessions if s.start_offset_s is not None]
    rows = [
        [
            "arrival",
            spec.arrival.kind
            + (
                f" (concurrency {plan.closed_loop_concurrency})"
                if plan.closed_loop_concurrency
                else ""
            ),
        ],
        ["bounded by", f"num_requests={spec.num_requests} duration_s={spec.duration_s}"],
        ["sampled sessions / requests", f"{len(sessions)} / {len(turns)}"],
        ["new input tokens / turn", f"mean {ins.mean or 0:.1f}, p50 {ins.p(50)}, max {ins.max}"],
        ["max output tokens", f"mean {outs.mean or 0:.1f}, max {outs.max}"],
        ["shared prefix", "yes" if spec.shared_prefix else "no"],
        ["multi-turn sessions", "yes" if spec.sessions else "no"],
    ]
    if len(offsets) > 1 and offsets[-1] > 0:
        rows.append(
            [
                "offered rate (sampled)",
                f"{(len(offsets) - 1) / offsets[-1]:.2f} sessions/s over {offsets[-1]:.1f}s",
            ]
        )
    tenants = sorted({s.tenant_id for s in sessions if s.tenant_id})
    if tenants:
        rows.append(["tenants", ", ".join(tenants)])
    console.print(_table(f"Workload {spec.name}", ["property", "value"], rows))


# ============================================================================ benchmark
@benchmark_app.command("run")
def benchmark_run(
    config: ConfigArg,
    execute: Annotated[
        bool, typer.Option(help="Required for execution: gpu (cost guard).")
    ] = False,
    output: Annotated[
        Path | None, typer.Option(help="Results root (default: config output_dir)")
    ] = None,
    plots: Annotated[bool, typer.Option(help="Render PNG plots")] = True,
) -> None:
    """Run one benchmark and write results/<run-id>/ (report.md, summary.json, requests.parquet…)."""
    from inferencelab.benchmarking.config import BenchmarkConfig
    from inferencelab.benchmarking.runner import run_benchmark
    from inferencelab.config.loader import load_model_file

    cfg = load_model_file(config, BenchmarkConfig)
    if cfg.execution == "gpu":
        hw = cfg.hardware_ref
        est = cfg.estimated_duration_s or cfg.workload.duration_s
        console.print("[bold]GPU benchmark[/bold] - 1 run")
        console.print(f"  hardware: {hw.gpu if hw else 'unspecified'} × {hw.count if hw else '?'}")
        console.print(f"  estimated duration: {f'{est:.0f}s' if est else 'unknown (closed loop)'}")
        if hw is not None and hw.pricing is not None and est:
            cost = hw.pricing.hourly_cost_for(max(1, hw.count)) * est / 3600
            console.print(
                f"  estimated cost (upper bound): ${cost:.2f} ({hw.pricing.price_source})"
            )
        if not execute:
            raise _fail("execution: gpu requires --execute (nothing was run)", code=2)
    result = _run(run_benchmark(cfg, base_dir=config.parent, output_root=output, plots=plots))
    _print_summary(result.summary)
    typer.echo(f"run_dir={result.run_dir}")


def _print_summary(summary: Any) -> None:
    from inferencelab.reporting.tables import fmt

    s = summary
    rows = [
        ["label", s.label],
        [
            "requests (measured)",
            f"{s.counts.measured} ({s.counts.ok} ok, {s.counts.rejected} rejected, {s.counts.error + s.counts.timeout} failed)",
        ],
        [
            "throughput",
            f"{fmt(s.throughput.request_throughput, 2)} req/s, {fmt(s.throughput.output_token_throughput, 0)} output tok/s",
        ],
        ["goodput", fmt(s.throughput.goodput_rps, 2, " req/s") if s.slo else "no SLO"],
        [
            "TTFT p50/p95/p99",
            f"{fmt(s.latency['ttft_ms'].p50)} / {fmt(s.latency['ttft_ms'].p95)} / {fmt(s.latency['ttft_ms'].p99)} ms",
        ],
        [
            "TPOT p50/p95",
            f"{fmt(s.latency['tpot_ms'].p50, 2)} / {fmt(s.latency['tpot_ms'].p95, 2)} ms",
        ],
        ["E2E p50/p95", f"{fmt(s.latency['e2e_ms'].p50)} / {fmt(s.latency['e2e_ms'].p95)} ms"],
        ["SLO", s.slo.status if s.slo else "n/a"],
    ]
    console.print(_table(f"Run {s.run_id}", ["metric", "value"], rows))
    for w in s.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")


@benchmark_app.command("compare")
def benchmark_compare(
    runs: Annotated[list[str], typer.Argument(help="Run ids or run directories")],
    results: Annotated[Path, typer.Option(help="Results root")] = Path("results"),
    output: Annotated[Path | None, typer.Option(help="Write the markdown table here")] = None,
) -> None:
    """Comparison table across runs (only measured values; 'n/a' otherwise)."""
    from inferencelab.benchmarking.artifacts import load_run, resolve_run_dir
    from inferencelab.reporting.tables import RunRow, comparison_table, prefix_cache_table

    rows = []
    for ref in runs:
        loaded = load_run(resolve_run_dir(ref, results))
        dep = loaded.metadata.get("deployment") or {}
        backend = str(dep.get("backend") or dep.get("name") or loaded.metadata.get("target_kind"))
        rows.append(
            RunRow(
                backend=backend,
                config=loaded.metadata.get("name", loaded.summary.run_id),
                summary=loaded.summary,
                gpu_memory_gb=(loaded.metadata.get("gpu_memory") or {}).get("peak_gb"),
                prefix_caching=dep.get("prefix_caching") if isinstance(dep, dict) else None,
            )
        )
    labels = {r.summary.label for r in rows}
    text = "\n".join(
        [
            *(f"> {label}" for label in sorted(labels)),
            "",
            comparison_table(rows),
            "",
            "Prefix cache:",
            "",
            prefix_cache_table(rows),
        ]
    )
    typer.echo(text)
    if output is not None:
        output.write_text(text + "\n", encoding="utf-8")


# ============================================================================ report
@app.command()
def report(
    run: Annotated[str, typer.Argument(help="Run id or run directory")],
    results: Annotated[Path, typer.Option(help="Results root")] = Path("results"),
) -> None:
    """Re-generate report.md (and plots) for a run from its stored artifacts."""
    from inferencelab.benchmarking.artifacts import load_run, resolve_run_dir
    from inferencelab.benchmarking.runner import _run_plots
    from inferencelab.reporting.plots import plots_available
    from inferencelab.reporting.report import render_run_report

    run_dir = resolve_run_dir(run, results)
    loaded = load_run(run_dir)
    images = (
        _run_plots(run_dir / "plots", loaded.records(), loaded.summary) if plots_available() else []
    )
    text = render_run_report(loaded.summary, loaded.metadata, plots=images, run_dir=run_dir)
    (run_dir / "report.md").write_text(text, encoding="utf-8")
    typer.echo(str(run_dir / "report.md"))


# ============================================================================ profile
@profile_app.command("show")
def profile_show(profile: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Show a performance profile's measured load points."""
    from inferencelab.profiles.registry import ProfileRegistry
    from inferencelab.reporting.tables import fmt

    p = ProfileRegistry(profile.parent).load(profile)
    console.print(
        f"[bold]{p.profile_id}[/bold] {'[yellow](SIMULATED)[/yellow]' if p.simulated else ''}"
    )
    console.print(f"hardware {p.hardware} · model {p.model} · backend {p.backend}")
    rows = [
        [
            f"{pt.load.kind}={pt.load.value:g}",
            fmt(pt.request_throughput, 2),
            fmt(pt.output_token_throughput, 0),
            fmt(pt.ttft_ms.p50),
            fmt(pt.ttft_ms.p95),
            fmt(pt.tpot_ms.p95, 2),
            fmt(pt.goodput_rps, 2),
            f"{pt.error_rate * 100:.1f}%",
        ]
        for pt in p.sorted_points()
    ]
    console.print(
        _table(
            "Points",
            ["load", "req/s", "out tok/s", "TTFT p50", "TTFT p95", "TPOT p95", "goodput", "errors"],
            rows,
        )
    )


# ============================================================================ capacity
@capacity_app.command("plan")
def capacity_plan(config: ConfigArg) -> None:
    """Replicas needed for a target rate under an SLO, from measured profile points."""
    from inferencelab.capacity.planner import CapacityPlanConfig, plan_capacity
    from inferencelab.config.loader import load_model_file
    from inferencelab.profiles.registry import ProfileRegistry
    from inferencelab.reporting.tables import fmt

    cfg = load_model_file(config, CapacityPlanConfig)
    profile_path = (config.parent / cfg.profile).resolve()
    profile = ProfileRegistry(profile_path.parent).load(profile_path)
    plan = plan_capacity(profile, cfg)
    rows = [
        [
            f"{v.point.load.kind}={v.point.load.value:g}",
            fmt(v.point.request_throughput, 2),
            {
                True: "[green]meets[/green]",
                False: "[red]violates[/red]",
                None: "[yellow]insufficient data[/yellow]",
            }[v.meets],
            "; ".join(v.details),
        ]
        for v in plan.verdicts
    ]
    console.print(
        _table(
            f"SLO per measured point ({profile.profile_id})",
            ["load", "req/s", "verdict", "details"],
            rows,
        )
    )
    console.print(
        f"target: {cfg.target.request_rate:g} req/s · headroom {cfg.headroom:.0%} · redundancy +{cfg.redundancy}"
    )
    if plan.required_replicas is None:
        console.print("[red]no capacity estimate possible[/red]")
    else:
        console.print(
            f"[bold]{plan.required_replicas} replicas[/bold] ({plan.total_gpus} GPUs) - sustainable "
            f"{plan.sustainable_rps_per_replica:.2f} req/s per replica"
        )
        if plan.hourly_cost_usd is not None:
            console.print(
                f"cost: ${plan.hourly_cost_usd:.2f}/h, ${fmt(plan.cost_per_1m_output_tokens_usd, 3)} per 1M output tokens"
            )
    for w in plan.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")


# ============================================================================ autoscale
@autoscale_app.command("simulate")
def autoscale_simulate(
    configs: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help="One or more autoscaling configs"),
    ],
    output: Annotated[Path | None, typer.Option(help="Write timeseries CSV + plots here")] = None,
) -> None:
    """Simulate autoscaling policies against a load trace (fluid model; not a K8s validation)."""
    import csv

    from inferencelab.autoscaling.simulator import AutoscaleConfig, simulate
    from inferencelab.config.loader import load_model_file
    from inferencelab.profiles.registry import ProfileRegistry
    from inferencelab.reporting import plots as plotlib

    rows = []
    for path in configs:
        cfg = load_model_file(path, AutoscaleConfig)
        profile = None
        if cfg.replica.profile:
            ppath = (path.parent / cfg.replica.profile).resolve()
            profile = ProfileRegistry(ppath.parent).load(ppath)
        result = simulate(cfg, profile=profile, base_dir=path.parent)
        rows.append(
            [
                cfg.name,
                result.policy,
                f"{result.violation_fraction * 100:.2f}%",
                f"{result.mean_replicas:.2f}",
                f"{result.replica_hours:.2f}",
                "n/a" if result.cost_usd is None else f"${result.cost_usd:.2f}",
                f"{result.max_queue:.0f}",
                str(result.scale_events),
            ]
        )
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            with (output / f"{cfg.name}.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        "t_s",
                        "offered_rps",
                        "served_per_s",
                        "queue",
                        "ready",
                        "starting",
                        "latency_ms",
                    ]
                )
                for t in result.ticks:
                    writer.writerow(
                        [
                            t.t,
                            f"{t.offered_rps:.3f}",
                            f"{t.served / cfg.tick_s:.3f}",
                            f"{t.queue:.2f}",
                            t.ready,
                            t.starting,
                            f"{t.latency_ms:.1f}",
                        ]
                    )
            ts = [t.t for t in result.ticks]
            plotlib.small_multiples_timeline(
                ts,
                [
                    (
                        "req/s",
                        {
                            "offered": [t.offered_rps for t in result.ticks],
                            "served": [t.served / cfg.tick_s for t in result.ticks],
                        },
                    ),
                    (
                        "replicas",
                        {
                            "ready": [float(t.ready) for t in result.ticks],
                            "starting": [float(t.starting) for t in result.ticks],
                        },
                    ),
                    ("queued requests", {"queue": [t.queue for t in result.ticks]}),
                ],
                path=output / f"{cfg.name}.png",
                title=f"Autoscaling: {cfg.name} ({result.policy})",
                note="fluid-model simulation, not a Kubernetes measurement"
                + ("; SIMULATED profile" if profile and profile.simulated else ""),
            )
        for note in result.notes:
            console.print(f"[yellow]{cfg.name}:[/yellow] {note}")
    console.print(
        _table(
            "Autoscaling simulation",
            [
                "config",
                "policy",
                "SLO-violating requests",
                "mean replicas",
                "replica-hours",
                "cost",
                "max queue",
                "scale events",
            ],
            rows,
        )
    )


# ============================================================================ experiment
@experiment_app.command("plan")
def experiment_plan(config: ConfigArg) -> None:
    """Expand the matrix and show run count, duration and cost estimates (nothing runs)."""
    from inferencelab.config.loader import load_model_file
    from inferencelab.experiments.plan import ExperimentTooLargeError, expand_experiment
    from inferencelab.experiments.spec import ExperimentSpec

    spec = load_model_file(config, ExperimentSpec)
    try:
        plan = expand_experiment(spec, config.parent)
    except ExperimentTooLargeError as exc:
        raise _fail(str(exc)) from exc
    _print_plan(plan)


def _print_plan(plan: Any) -> None:
    rows = [
        [
            str(r.index),
            r.target,
            r.workload.name,
            f"{r.load_kind}={r.load_value:g}" if r.load_value is not None else "-",
            "unknown" if r.estimated_duration_s is None else f"{r.estimated_duration_s:.0f}s",
        ]
        for r in plan.runs
    ]
    console.print(
        _table(
            f"Experiment {plan.spec.name} ({plan.spec.mode})",
            ["#", "target", "workload", "load", "est. duration"],
            rows,
        )
    )
    console.print(f"runs: {len(plan.runs)} (cap {plan.spec.max_runs})")
    est = plan.estimated_duration_s
    console.print(
        f"estimated duration: {'unknown' if est is None else f'{est / 60:.1f} min'}; upper bound {plan.duration_upper_bound_s / 60:.1f} min"
    )
    if plan.pricing is not None and plan.pricing.pricing is not None:
        pr = plan.pricing.pricing
        console.print(
            f"hardware: {plan.pricing.gpu} × {plan.pricing.count} at ${pr.hourly_cost_usd}/h ({pr.unit}, source: {pr.price_source}, recorded {pr.recorded_at})"
        )
    if plan.estimated_cost_upper_bound_usd is not None:
        console.print(
            f"[bold]estimated cost upper bound: ${plan.estimated_cost_upper_bound_usd:.2f}[/bold]"
        )
    for w in plan.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    if plan.requires_execute_flag:
        console.print(
            "remote GPU plans run only with: inferencelab remote run CONFIG --host HOST --wheel WHEEL --execute"
        )


@experiment_app.command("run")
def experiment_run(
    config: ConfigArg,
    output: Annotated[Path, typer.Option(help="Results root")] = Path("results"),
    plots: Annotated[bool, typer.Option()] = True,
) -> None:
    """Run a local (mock/simulated) experiment; remote plans go through `remote run`."""
    from inferencelab.config.loader import load_model_file
    from inferencelab.experiments.plan import expand_experiment
    from inferencelab.experiments.runner import run_local_experiment
    from inferencelab.experiments.spec import ExperimentSpec

    spec = load_model_file(config, ExperimentSpec)
    plan = expand_experiment(spec, config.parent)
    _print_plan(plan)
    if spec.mode != "local":
        raise _fail("remote experiments: use `inferencelab remote plan/run`", code=2)
    result = _run(run_local_experiment(plan, output_root=output, plots=plots))
    console.print(f"experiment report: {result.report}")


# ============================================================================ remote / gpu
def _print_checks(checks: list[Any]) -> int:
    colour = {"ok": "green", "warn": "yellow", "fail": "red", "skip": "cyan"}
    rows = [
        [c.name, f"[{colour[c.status]}]{c.status}[/{colour[c.status]}]", c.detail] for c in checks
    ]
    console.print(_table("GPU doctor", ["check", "status", "detail"], rows))
    return sum(1 for c in checks if c.status == "fail")


@gpu_app.command("doctor")
def gpu_doctor_cmd() -> None:
    """Check NVIDIA driver, CUDA, GPUs and Docker GPU runtime on THIS machine."""
    from inferencelab.remote.doctor import gpu_doctor
    from inferencelab.remote.executor import LocalExecutor

    checks = _run(gpu_doctor(LocalExecutor(), expect_gpu=False))
    _print_checks(checks)


@remote_app.command("doctor")
def remote_doctor(
    host: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Remote host YAML")],
    container_test: Annotated[
        str | None, typer.Option(help="Also run `nvidia-smi -L` in this CUDA image")
    ] = None,
) -> None:
    """Audit a remote GPU host over SSH (read-only checks)."""
    from inferencelab.config.loader import load_model_file
    from inferencelab.remote.doctor import gpu_doctor
    from inferencelab.remote.executor import SSHExecutor
    from inferencelab.remote.host import RemoteHost

    spec = load_model_file(host, RemoteHost)
    checks = _run(
        gpu_doctor(SSHExecutor(spec), expect_gpu=True, container_test_image=container_test)
    )
    if _print_checks(checks):
        raise typer.Exit(1)


def _campaign(
    experiment: Path, host: Path, models_dir: Path, wheel: Path | None, executor: Any, results: Path
) -> Any:
    from inferencelab.config.loader import load_model_file
    from inferencelab.experiments.plan import expand_experiment
    from inferencelab.experiments.spec import ExperimentSpec
    from inferencelab.remote.campaign import RemoteCampaign
    from inferencelab.remote.host import RemoteHost

    spec = load_model_file(experiment, ExperimentSpec)
    plan = expand_experiment(spec, experiment.parent)
    host_spec = load_model_file(host, RemoteHost)
    models = _load_models(
        (experiment.parent / spec.models_dir).resolve() if models_dir == Path() else models_dir
    )
    return plan, RemoteCampaign(
        plan,
        host_spec,
        models=models,
        executor=executor,
        local_results=results,
        wheel=wheel,
        poll_interval_s=0.0
        if executor is not None and getattr(executor, "description", "") == "dry-run"
        else 10.0,
    )


@remote_app.command("plan")
def remote_plan(
    experiment: ConfigArg,
    host: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Remote host YAML")],
    models_dir: Annotated[
        Path, typer.Option(help="Model registry (default: from experiment)")
    ] = Path(),
) -> None:
    """Dry run: print every command a remote campaign would execute (nothing is executed)."""
    import shlex

    from inferencelab.remote.campaign import dry_run_responders

    executor = dry_run_responders()
    plan, campaign = _campaign(
        experiment,
        host,
        models_dir,
        Path("dist/inferencelab.whl"),
        executor,
        Path("results/remote"),
    )
    _print_plan(plan)
    _run(campaign.run())
    console.print("\n[bold]Commands (dry run, executed on the GPU host over SSH):[/bold]")
    for argv in executor.commands:
        typer.echo("  $ " + shlex.join(argv))
    for kind, a, b in executor.transfers:
        typer.echo(f"  {kind}: {a} -> {b}")


@remote_app.command("run")
def remote_run(
    experiment: ConfigArg,
    host: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Remote host YAML")],
    wheel: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="InferenceLab wheel to install remotely"),
    ],
    execute: Annotated[bool, typer.Option(help="Actually run (paid GPU time).")] = False,
    models_dir: Annotated[
        Path, typer.Option(help="Model registry (default: from experiment)")
    ] = Path(),
    results: Annotated[Path, typer.Option(help="Local results root")] = Path("results/remote"),
) -> None:
    """Execute a remote GPU campaign. Requires --execute; never creates or deletes machines."""
    from inferencelab.config.loader import load_model_file
    from inferencelab.remote.executor import SSHExecutor
    from inferencelab.remote.host import RemoteHost

    host_spec = load_model_file(host, RemoteHost)
    plan, campaign = _campaign(experiment, host, models_dir, wheel, SSHExecutor(host_spec), results)
    _print_plan(plan)
    if not execute:
        raise _fail(
            "remote GPU runs require --execute (nothing was run). Use `remote plan` for a dry run.",
            code=2,
        )
    collected = _run(campaign.run())
    for step in campaign.log:
        console.print(f"[cyan]{step.step}[/cyan] {step.detail}")
    console.print(f"collected {len(collected)} runs under {results}")


# ============================================================================ quality
@quality_app.command("compare")
def quality_compare(
    reference: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Reference backend YAML")
    ],
    candidate: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Candidate backend YAML")
    ],
    max_tokens: Annotated[int, typer.Option()] = 64,
    output: Annotated[Path | None, typer.Option(help="Write the JSON report here")] = None,
) -> None:
    """Greedy-output equivalence and task accuracy: candidate vs reference deployment."""
    from inferencelab.backends.factory import create_backend
    from inferencelab.config.loader import load_backend_config
    from inferencelab.quality.evaluators import compare_backends

    ref_cfg, cand_cfg = load_backend_config(reference), load_backend_config(candidate)

    async def go() -> Any:
        ref, cand = create_backend(ref_cfg), create_backend(cand_cfg)
        try:
            return await compare_backends(
                ref,
                cand,
                reference_model=ref_cfg.served_model(ref_cfg.models[0]),
                candidate_model=cand_cfg.served_model(cand_cfg.models[0]),
                max_tokens=max_tokens,
            )
        finally:
            await ref.aclose()
            await cand.aclose()

    rep = _run(go())
    console.print(
        f"cases {rep.cases} · exact match {rep.exact_match_rate:.0%} · prefix agreement {rep.mean_prefix_agreement:.2f} · "
        f"accuracy ref {rep.reference_accuracy:.0%} vs cand {rep.candidate_accuracy:.0%} (Δ {rep.accuracy_delta:+.0%})"
    )
    if output is not None:
        output.write_text(json.dumps(rep.model_dump(), indent=2), encoding="utf-8")


def main() -> None:
    # Never crash on a legacy console encoding (e.g. cp1252 on Windows).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    app()


if __name__ == "__main__":
    main()
