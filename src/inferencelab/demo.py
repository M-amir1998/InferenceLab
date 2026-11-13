"""``inferencelab demo``: complete CPU-only demonstration over real loopback HTTP.

Starts simulated engine servers and a gateway per scenario (all in-process, no GPU, no
internet), drives load through the gateway with the benchmark client, and writes a
combined report. Every number it produces is labelled SIMULATED BACKEND RESULTS.

Scenarios: routing policies (round robin / least outstanding / latency aware), prefix
affinity on multi-turn chat, circuit breaking under an injected fault, tenant rate
limits, and backpressure with SLO/goodput reporting.
"""

from __future__ import annotations

import asyncio
import shutil
from collections import Counter
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

from inferencelab.backends.config import (
    FaultMode,
    FaultWindow,
    LatencySpec,
    MockBackendConfig,
    SimulatedBackendConfig,
)
from inferencelab.backends.mock.backend import MockBackend
from inferencelab.backends.server import create_backend_app
from inferencelab.backends.simulated.backend import SimulatedInferenceBackend
from inferencelab.backends.simulated.engine import SimulatedEngineConfig
from inferencelab.benchmarking.config import BenchmarkConfig, HTTPTargetConfig
from inferencelab.benchmarking.runner import RunResult, make_run_id, run_benchmark
from inferencelab.domain.result import Phase, RequestStatus
from inferencelab.domain.slo import SLOSpec
from inferencelab.domain.workload import WorkloadSpec
from inferencelab.gateway.app import create_app
from inferencelab.gateway.config import GatewayConfig
from inferencelab.metrics.summary import SIMULATED_LABEL
from inferencelab.net import LoopbackServer
from inferencelab.reporting import plots as plotlib
from inferencelab.reporting.tables import fmt

__all__ = ["DemoResult", "run_demo"]

MODEL = "demo-model"
FAST = SimulatedEngineConfig(
    step_overhead_s=0.012, prefill_s_per_token=0.00003, decode_s_per_seq=0.0003, max_num_seqs=32
)
SLOW = SimulatedEngineConfig(
    step_overhead_s=0.026, prefill_s_per_token=0.00006, decode_s_per_seq=0.0006, max_num_seqs=32
)
#: A smaller/older accelerator: slower steps *and* only 8 concurrent sequences.
SMALL = SLOW.model_copy(update={"max_num_seqs": 8, "num_kv_blocks": 1024})
SLO = SLOSpec.model_validate(
    {"name": "demo-interactive", "ttft_p95_ms": 400, "tpot_p95_ms": 50, "e2e_p95_ms": 3000}
)

Progress = Callable[[str], None]


@dataclass(slots=True)
class ScenarioResult:
    key: str
    title: str
    runs: dict[str, RunResult] = field(default_factory=dict)
    gateway_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DemoResult:
    out_dir: Path
    report: Path
    scenarios: list[ScenarioResult]


@asynccontextmanager
async def _engines(
    specs: Sequence[tuple[str, SimulatedEngineConfig | MockBackendConfig]],
) -> AsyncIterator[dict[str, tuple[LoopbackServer, Any]]]:
    async with AsyncExitStack() as stack:
        out: dict[str, tuple[LoopbackServer, Any]] = {}
        for name, cfg in specs:
            backend: SimulatedInferenceBackend | MockBackend
            if isinstance(cfg, MockBackendConfig):
                backend = MockBackend(cfg)
                kind = "mock"
            else:
                backend = SimulatedInferenceBackend(
                    SimulatedBackendConfig(name=name, models=[MODEL], engine=cfg)
                )
                kind = "simulated"
            app = create_backend_app(backend, models=[MODEL], kind=kind, admin=True)
            server = await stack.enter_async_context(LoopbackServer(app))
            out[name] = (server, backend)
        yield out


def _gateway(engines: dict[str, tuple[LoopbackServer, Any]], **overrides: Any) -> GatewayConfig:
    data: dict[str, Any] = {
        "backends": [
            {
                "type": "openai",
                "name": name,
                "base_url": server.url,
                "models": [MODEL],
                "deployment": name,
            }
            for name, (server, _) in engines.items()
        ],
        "health": {"interval_s": 1.0},
        "slo": SLO.model_dump(),
    }
    data.update(overrides)
    return GatewayConfig.model_validate(data)


async def _bench(name: str, url: str, workload: WorkloadSpec, out: Path) -> RunResult:
    config = BenchmarkConfig(
        name=name,
        target=HTTPTargetConfig(base_url=url),
        workload=workload,
        slo=SLO,
        simulated=True,
        output_dir=str(out),
    )
    return await run_benchmark(config, output_root=out)


async def _gateway_counters(url: str) -> dict[str, float]:
    """Gateway counters keyed ``name{label=value,...}`` (labels kept, unlike the summed view)."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        text = (await client.get(f"{url}/metrics")).text
    keys = (
        "inferencelab_retries_total",
        "inferencelab_circuit_transitions_total",
        "inferencelab_rejections_total",
        "inferencelab_errors_total",
    )
    out: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name.startswith(keys):
                labels = ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items()))
                out[f"{sample.name}{{{labels}}}"] = float(sample.value)
    return out


def _wl(name: str, scale: float, **kw: Any) -> WorkloadSpec:
    base: dict[str, Any] = {"name": name, "model": MODEL, "seed": 7}
    if "num_requests" in kw:
        kw["num_requests"] = max(16, int(kw["num_requests"] * scale))
    if "duration_s" in kw:
        kw["duration_s"] = max(2.0, kw["duration_s"] * scale)
    base.update(kw)
    return WorkloadSpec.model_validate(base)


async def _routing(out: Path, scale: float, progress: Progress) -> ScenarioResult:
    res = ScenarioResult("routing", "Routing policies under heterogeneous backends")
    workload = _wl(
        "routing-mixed",
        scale,
        duration_s=12.0,
        input_tokens={"kind": "uniform", "min": 128, "max": 512},
        output_tokens={"kind": "fixed", "value": 32},
        arrival={"kind": "poisson", "rate": 20},
        warmup={"seconds": 1.0},
    )
    for policy in ("round_robin", "least_outstanding", "latency_aware"):
        progress(f"routing: {policy}")
        async with _engines([("sim-fast", FAST), ("sim-small", SMALL)]) as engines:
            app = create_app(_gateway(engines, routing={"policy": policy}))
            async with LoopbackServer(app) as gw:
                res.runs[policy] = await _bench(f"demo-routing-{policy}", gw.url, workload, out)
    return res


async def _affinity(out: Path, scale: float, progress: Progress) -> ScenarioResult:
    res = ScenarioResult("affinity", "Prefix/session affinity on multi-turn chat")
    workload = _wl(
        "chat-sessions",
        scale,
        num_requests=192,
        sessions={"turns": {"kind": "fixed", "value": 4}, "system_prompt_tokens": 256},
        input_tokens={"kind": "uniform", "min": 48, "max": 96},
        output_tokens={"kind": "fixed", "value": 48},
        arrival={"kind": "closed_loop", "concurrency": 16},
    )
    for label, routing in (
        ("round_robin", {"policy": "round_robin"}),
        (
            "affinity",
            {"policy": "round_robin", "affinity": {"enabled": True, "keys": ["conversation_id"]}},
        ),
    ):
        progress(f"affinity: {label}")
        async with _engines([("sim-a", FAST), ("sim-b", FAST)]) as engines:
            app = create_app(_gateway(engines, routing=routing))
            async with LoopbackServer(app) as gw:
                res.runs[label] = await _bench(f"demo-affinity-{label}", gw.url, workload, out)
    return res


async def _circuit(out: Path, scale: float, progress: Progress) -> ScenarioResult:
    res = ScenarioResult("circuit", "Circuit breaking and retries under an injected backend fault")
    progress("circuit breaker: injecting a fault mid-run")
    flaky = MockBackendConfig(
        name="mock-flaky", models=[MODEL], latency=LatencySpec(ttft_ms=30, tpot_ms=20)
    )
    workload = _wl(
        "steady-poisson",
        scale,
        duration_s=8.0,
        input_tokens={"kind": "fixed", "value": 128},
        output_tokens={"kind": "fixed", "value": 24},
        arrival={"kind": "poisson", "rate": 20},
    )
    async with _engines([("sim-fast", FAST), ("mock-flaky", flaky)]) as engines:
        cfg = _gateway(
            engines,
            routing={"policy": "least_outstanding"},
            circuit_breaker={"failure_threshold": 3, "open_duration_s": 2.0},
            retry={"max_attempts": 3, "backoff_initial_s": 0.01},
        )
        async with LoopbackServer(create_app(cfg)) as gw:
            server, _ = engines["mock-flaky"]
            fault_at = 1.5 * max(scale, 0.5)

            async def inject() -> None:
                await asyncio.sleep(fault_at)
                window = FaultWindow(mode=FaultMode.ERROR, start_s=0.0, end_s=4.0 * max(scale, 0.5))
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{server.url}/admin/fault", json=window.model_dump(mode="json")
                    )

            injector = asyncio.create_task(inject())
            res.runs["with-fault"] = await _bench("demo-circuit-breaker", gw.url, workload, out)
            await injector
            res.gateway_metrics["with-fault"] = await _gateway_counters(gw.url)
    res.notes.append(
        f"error fault injected into mock-flaky at t≈{fault_at:.1f}s for {4.0 * max(scale, 0.5):.1f}s"
    )
    return res


async def _rate_limits(out: Path, scale: float, progress: Progress) -> ScenarioResult:
    res = ScenarioResult("rate_limits", "Per-tenant rate limiting (token bucket)")
    progress("rate limits: tenant quotas")
    workload = _wl(
        "two-tenants",
        scale,
        duration_s=6.0,
        input_tokens={"kind": "fixed", "value": 64},
        output_tokens={"kind": "fixed", "value": 16},
        arrival={"kind": "poisson", "rate": 16},
        tenants=[{"tenant_id": "free", "weight": 1}, {"tenant_id": "pro", "weight": 1}],
    )
    async with _engines([("sim-fast", FAST)]) as engines:
        cfg = _gateway(
            engines,
            tenants={
                "tenants": {
                    "free": {"requests_per_second": 3, "burst": 3},
                    "pro": {"requests_per_second": 50, "burst": 50},
                }
            },
        )
        async with LoopbackServer(create_app(cfg)) as gw:
            res.runs["tenants"] = await _bench("demo-rate-limits", gw.url, workload, out)
    return res


async def _backpressure(out: Path, scale: float, progress: Progress) -> ScenarioResult:
    res = ScenarioResult(
        "backpressure", "Backpressure, load shedding and SLO/goodput under overload"
    )
    progress("backpressure: overload with a bounded queue")
    slow16 = SLOW.model_copy(update={"max_num_seqs": 16})
    workload = _wl(
        "overload",
        scale,
        duration_s=12.0,
        input_tokens={"kind": "fixed", "value": 256},
        output_tokens={"kind": "fixed", "value": 32},
        arrival={"kind": "poisson", "rate": 40},
    )
    async with _engines([("sim-slow", slow16)]) as engines:
        cfg = _gateway(
            engines,
            queue={"max_concurrency": 16, "max_pending_requests": 24, "max_queue_wait_s": 1.5},
        )
        async with LoopbackServer(create_app(cfg)) as gw:
            res.runs["overload"] = await _bench("demo-backpressure", gw.url, workload, out)
            res.gateway_metrics["overload"] = await _gateway_counters(gw.url)
    return res


async def run_demo(
    output_root: Path,
    *,
    quick: bool = False,
    publish_dir: Path | None = None,
    progress: Progress = print,
) -> DemoResult:
    scale = 0.35 if quick else 1.0
    out = output_root / make_run_id("demo")
    out.mkdir(parents=True, exist_ok=True)
    scenarios = [
        await _routing(out, scale, progress),
        await _affinity(out, scale, progress),
        await _circuit(out, scale, progress),
        await _rate_limits(out, scale, progress),
        await _backpressure(out, scale, progress),
    ]
    images = await asyncio.to_thread(_demo_plots, out / "plots", scenarios)
    report = out / "report.md"
    report.write_text(render_demo_report(scenarios, images, out, quick=quick), encoding="utf-8")
    if publish_dir is not None:
        await asyncio.to_thread(_publish, out, report, images, publish_dir)
    return DemoResult(out, report, scenarios)


def _publish(out: Path, report: Path, images: list[Path], dest: Path) -> None:
    """Copy the human-readable report and plots (never request-level data)."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "plots").mkdir(exist_ok=True)
    for img in images:
        shutil.copy2(img, dest / "plots" / img.name)
    text = report.read_text(encoding="utf-8")
    (dest / "report.md").write_text(text, encoding="utf-8")


def _demo_plots(directory: Path, scenarios: list[ScenarioResult]) -> list[Path]:
    images: list[Path] = []
    routing = next(s for s in scenarios if s.key == "routing")
    series = {
        policy: [
            r.ttft_s * 1000
            for r in run.records
            if r.phase is Phase.MEASURE and r.ok and r.ttft_s is not None
        ]
        for policy, run in routing.runs.items()
    }
    p = plotlib.latency_cdf(
        series,
        path=directory / "routing_ttft_cdf.png",
        title="TTFT by routing policy",
        xlabel="TTFT (ms)",
        note=SIMULATED_LABEL,
    )
    if p:
        images.append(p)
    affinity = next(s for s in scenarios if s.key == "affinity")
    series = {
        label: [
            r.ttft_s * 1000
            for r in run.records
            if r.phase is Phase.MEASURE and r.ok and r.ttft_s is not None
        ]
        for label, run in affinity.runs.items()
    }
    p = plotlib.latency_cdf(
        series,
        path=directory / "affinity_ttft_cdf.png",
        title="Multi-turn chat TTFT: round robin vs session affinity",
        xlabel="TTFT (ms)",
        note=SIMULATED_LABEL,
    )
    if p:
        images.append(p)
    return images


def _routing_takeaways(routing: ScenarioResult) -> list[str]:
    """Narrative computed from the measured numbers (never hard-coded claims)."""
    runs = routing.runs
    out: list[str] = []
    goodput = {k: r.summary.throughput.goodput_rps for k, r in runs.items()}
    known = {k: v for k, v in goodput.items() if v is not None}
    if known:
        best = max(known, key=lambda k: known[k])
        base = known.get("round_robin")
        line = f"Measured: highest goodput with **{best}** ({known[best]:.2f} req/s"
        if base is not None and best != "round_robin":
            line += f" vs {base:.2f} req/s for round_robin"
        out.append(line + ").")
    p95 = {k: r.summary.latency["ttft_ms"].p95 for k, r in runs.items()}
    if all(v is not None for v in p95.values()):
        ranked = sorted(p95, key=lambda k: p95[k] or 0.0)
        out.append(
            "TTFT p95 from best to worst: "
            + ", ".join(f"{k} {p95[k]:.0f} ms" for k in ranked)
            + "."
        )
    out.append(
        "Round robin sends half of the traffic to the small engine regardless of its backlog; "
        "load-aware policies (fewest in-flight requests, peak-EWMA latency × load) route around "
        "it. Compare the traffic-share column with the latency columns."
    )
    return out


def _share(run: RunResult) -> str:
    return ", ".join(f"{name} {b.share * 100:.0f}%" for name, b in run.summary.per_backend.items())


def render_demo_report(
    scenarios: list[ScenarioResult], images: list[Path], out: Path, *, quick: bool
) -> str:
    by = {s.key: s for s in scenarios}
    lines = [
        "# InferenceLab local demo",
        "",
        "> **SIMULATED BACKEND RESULTS** — everything below was produced on a CPU laptop by "
        "InferenceLab's simulated engines behind the real gateway over loopback HTTP. The "
        "numbers demonstrate gateway behaviour (routing, caching affinity, reliability, "
        "admission control, SLO accounting). They are **not** GPU inference measurements and "
        "say nothing about vLLM, SGLang or TensorRT-LLM performance.",
        "",
        f"Mode: {'quick' if quick else 'full'} · scenarios: {len(scenarios)} · output: `{out.name}`",
        "",
        "Simulated engines: `sim-fast` (12 ms step overhead, 32 sequences), `sim-small` (26 ms "
        "step overhead, 2× prefill/decode cost, only 8 sequences — a smaller accelerator), "
        "`sim-slow` (as sim-small with 16 sequences); `mock-flaky` is a fixed-latency mock with "
        "fault injection.",
        "",
    ]
    routing = by["routing"]
    lines += [
        "## 1. Routing policies on a heterogeneous fleet (sim-fast + sim-small, Poisson 20 req/s)",
        "",
        "| Policy | TTFT p50 | TTFT p95 | TPOT p95 | Req/s | Output tok/s | Goodput | Traffic share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for policy, run in routing.runs.items():
        s = run.summary
        lines.append(
            f"| {policy} | {fmt(s.latency['ttft_ms'].p50)} ms | {fmt(s.latency['ttft_ms'].p95)} ms "
            f"| {fmt(s.latency['tpot_ms'].p95, 2)} ms | {fmt(s.throughput.request_throughput, 2)} "
            f"| {fmt(s.throughput.output_token_throughput, 0)} | {fmt(s.throughput.goodput_rps, 2)} | {_share(run)} |"
        )
    lines += ["", *_routing_takeaways(routing), ""]
    aff = by["affinity"]
    lines += [
        "## 2. Session affinity on multi-turn chat (2 identical engines with prefix caching)",
        "",
        "| Routing | Engine-reported cached input | TTFT p50 | TTFT p95 | Req/s |",
        "|---|---|---|---|---|",
    ]
    for label, run in aff.runs.items():
        s = run.summary
        cached = s.tokens.cached_input_ratio
        lines.append(
            f"| {label} | {'n/a' if cached is None else f'{cached * 100:.1f}%'} | {fmt(s.latency['ttft_ms'].p50)} ms "
            f"| {fmt(s.latency['ttft_ms'].p95)} ms | {fmt(s.throughput.request_throughput, 2)} |"
        )
    lines += [
        "",
        "Each turn resends the whole conversation. With affinity (rendezvous hashing on "
        "`conversation_id`, bounded load) later turns land on the engine that already holds "
        "the conversation's KV blocks, so more of the prompt is served from cache.",
        "",
    ]
    circ = by["circuit"]
    run = circ.runs["with-fault"]
    m = circ.gateway_metrics.get("with-fault", {})
    retries = sum(v for k, v in m.items() if k.startswith("inferencelab_retries_total"))
    opened = sum(
        v
        for k, v in m.items()
        if k.startswith("inferencelab_circuit_transitions_total") and "to_state=open}" in k
    )
    s = run.summary
    lines += [
        "## 3. Circuit breaking and bounded retries",
        "",
        f"{'; '.join(circ.notes)}.",
        "",
        "| Client-observed success | Error rate | Gateway retries | Circuit opened | TTFT p95 |",
        "|---|---|---|---|---|",
        f"| {s.counts.ok}/{s.errors.attempted} | {s.errors.error_rate * 100:.2f}% | {retries:.0f} "
        f"| {opened:.0f}× | {fmt(s.latency['ttft_ms'].p95)} ms |",
        "",
        "Failures before the first token are retried on the healthy engine (never after output "
        "has been streamed); the breaker opens after consecutive failures and probes again in "
        "half-open state once the cooldown expires.",
        "",
    ]
    rl = by["rate_limits"].runs["tenants"]
    counts: Counter[tuple[str, str]] = Counter(
        (r.tenant_id or "?", r.status.value) for r in rl.records if r.phase is Phase.MEASURE
    )
    lines += [
        "## 4. Tenant rate limits (free: 3 req/s, pro: 50 req/s; offered ≈ 8 req/s each)",
        "",
        "| Tenant | Sent | OK | Rejected (429) |",
        "|---|---|---|---|",
    ]
    for tenant in ("free", "pro"):
        sent = sum(v for (t, _), v in counts.items() if t == tenant)
        lines.append(
            f"| {tenant} | {sent} | {counts[(tenant, RequestStatus.OK.value)]} | "
            f"{counts[(tenant, RequestStatus.REJECTED.value)]} |"
        )
    bp = by["backpressure"].runs["overload"]
    s = bp.summary
    q = s.latency["gateway_queue_ms"]
    lines += [
        "",
        "## 5. Backpressure, load shedding and SLO accounting (one sim-slow engine, offered Poisson 40 req/s)",
        "",
        "| Throughput | Goodput | Queue-full rejections | Shed (max queue wait) | Gateway queue p50 / p90 | TTFT p95 | SLO |",
        "|---|---|---|---|---|---|---|",
        f"| {fmt(s.throughput.request_throughput, 2)} req/s | {fmt(s.throughput.goodput_rps, 2)} req/s "
        f"| {s.errors.by_kind.get('queue_full', 0)} | {s.errors.by_kind.get('load_shed', 0)} "
        f"| {fmt(q.p50)} / {fmt(q.p90)} ms | {fmt(s.latency['ttft_ms'].p95)} ms "
        f"| {(s.slo.status.upper() if s.slo else 'n/a')} |",
        "",
        "The queue is bounded (16 executing + 24 waiting, 1.5 s max wait): excess load is rejected "
        "quickly with 503 instead of queueing without bound, so admitted requests keep a bounded "
        "latency. Throughput and goodput are reported separately.",
        "",
    ]
    if images:
        lines += ["## Plots", ""]
        lines += [f"![{p.stem}](plots/{p.name})\n" for p in images]
    lines += [
        "## Per-run reports (local `results/` directory; request-level data is not committed)",
        "",
    ]
    for scen in scenarios:
        for label, r in scen.runs.items():
            rel = r.run_dir.name if r.run_dir else "(not written)"
            lines.append(f"- {scen.key} / {label}: `{rel}/report.md`")
    lines.append("")
    return "\n".join(lines)
