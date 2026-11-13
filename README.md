# InferenceLab

> InferenceLab is a reproducible LLM inference engineering platform for benchmarking serving engines, routing traffic, enforcing SLOs, studying caching and batching, and optimizing latency, throughput and cost.

It is the layer **around** vLLM, SGLang and TensorRT-LLM: an OpenAI-compatible gateway
(routing, admission control, circuit breaking, bounded retries, streaming), a benchmark
harness with precise latency/goodput definitions, capped and cost-guarded experiment
matrices, profile-driven capacity planning and autoscaling simulation, and a cloud-neutral
SSH runner for rented GPUs. All of it is developed and tested on a CPU-only laptop.

**GPU benchmarks: pending.** No GPU numbers are claimed anywhere in this repository. The
results below come from InferenceLab's CPU simulator and are labelled as such.

`0.1.0.dev0` · Python 3.11+ · Apache-2.0 · CI: Jenkins

---

## Why InferenceLab

Serving LLMs well is an engineering problem with many coupled knobs: engine choice,
precision, batch limits, prefix caching, chunked prefill, speculative decoding, routing,
admission control and fleet size — all under latency SLOs and a budget. Published numbers
rarely transfer: different GPUs, versions, workloads and measurement code. InferenceLab
makes each question a reproducible experiment:

* the **same** workload (seeded), model revision, GPU and measurement code for every engine;
* metrics defined like `vllm bench serve` (TTFT, TPOT, ITL, E2E, goodput) with
  sample-size-aware percentiles;
* engine-reported metrics when available, never fabricated ones;
* explicit SIMULATED vs GPU labelling, pinned versions and recorded prices.

## Architecture

```text
 client workloads ──► gateway: limits ─ auth ─ quotas ─ admission ─ tiered router ─ attempts ──► SSE
                                                                       │
                                              ┌────────────────────────┼────────────────────────┐
                                              ▼                        ▼                        ▼
                                         vLLM (HTTP)             SGLang (HTTP)        TensorRT-LLM (HTTP)
                                              └──────── engine /metrics (Prometheus) ───────────┘
 gateway /metrics ─► Prometheus ─► Grafana          optional OpenTelemetry spans
 control plane: models · deployments · workloads · SLOs · experiments · profiles · capacity · cost · remote
```

The **data plane** (gateway, routing, admission, reliability, backends) is separated from
the **control plane** (configuration, experiments, benchmarking, planning, reporting).
Engines are reached only over their OpenAI-compatible HTTP APIs (ADR-001), so the core never
imports CUDA code. See [docs/architecture/overview.md](docs/architecture/overview.md).

## Serving backends

| Backend | Use | Integration |
|---|---|---|
| `vllm` | GPU engine (v0.33.0 verified) | HTTP adapter + launch builder + `vllm:*` metrics |
| `sglang` | GPU engine (v0.5.23 verified) | HTTP adapter + launch builder + `sglang:*` metrics |
| `tensorrt_llm` | GPU engine (trtllm-serve 1.2.1 verified) | HTTP adapter + launch builder + `trtllm_*` metrics |
| `openai` | any OpenAI-compatible endpoint | generic HTTP adapter |
| `mock` | deterministic latency + fault injection | in-process or served over HTTP |
| `simulated` | CPU continuous-batching engine model (paged KV, prefix cache, chunked prefill, preemption) | in-process or served over HTTP |

Model ≠ deployment: one checkpoint (`configs/models/`) can be deployed as vLLM+BF16,
vLLM+AWQ or SGLang+AWQ (`configs/deployments/`), each with pinned images, parallelism,
context and batch limits, caching, LoRA and speculative-decoding settings. Launch builders
reject combinations an engine cannot serve (e.g. TensorRT-LLM with an AutoAWQ checkpoint).
See [docs/backends/](docs/backends/).

## Gateway

`POST /v1/chat/completions`, `POST /v1/completions` (streaming with usage),
`GET /v1/models`, `/health`, `/ready`, `/metrics`, `/inferencelab/backends`.
Optional API-key auth, request-size limits, tenant quotas (req/s, concurrency, tokens/min),
a bounded priority queue with load shedding, per-backend circuit breakers, first-token /
idle / overall timeouts, and retries only while no output has reached the client.
[Compatibility matrix and headers](docs/architecture/gateway.md).

## Routing

`round_robin`, `random`, `least_outstanding`, `latency_aware` (peak-EWMA × load),
`cost_aware` (cheapest with spare capacity), `slo_aware` (cheapest predicted to meet the
TTFT budget) — plus **prefix/session affinity** (rendezvous hashing with bounded loads over
`prefix_key` / `conversation_id` ids, never prompt text) and **tiered fallback**
(preferred → secondary → fallback). [docs/architecture/routing.md](docs/architecture/routing.md)

## Workloads

Fixed / uniform / weighted-bucket / trace-driven lengths; closed-loop, constant, Poisson,
burst, step and trace arrivals; shared-prefix and multi-turn chat workloads; tenants,
priorities, client cancellation; seeded and deterministic. Ready-made profiles:
short/short, short/long, long/short, long/long, shared prefix, chat sessions, mixed Poisson,
step load, structured JSON, trace replay. [docs/methodology/workloads.md](docs/methodology/workloads.md)

## Metrics

TTFT, TPOT, ITL, E2E, E2E-from-schedule (coordinated omission), gateway queue time,
engine queue/prefill/decode time (engine-reported only), request and token throughput,
**goodput**, error taxonomy (timeouts, backend errors, invalid responses, rejections,
circuit rejections, cancellations), token accounting with reported/estimated provenance.
p50/p90/p95/p99 are withheld when fewer than 5 samples lie beyond them.
[docs/methodology/latency.md](docs/methodology/latency.md)

## Benchmarking

```bash
inferencelab benchmark run configs/benchmarks/smoke_simulated.yaml       # → results/<run-id>/
inferencelab benchmark compare <run-a> <run-b>                             # measured-only table
inferencelab experiment plan configs/experiments/gpu_first_comparison.yaml  # 32 runs, cost guard
```

Each run writes `config.yaml`, `metadata.json`, `requests.parquet`, `summary.json`,
`profile.json`, `system.json`, `report.md` and plots. Prompts are not stored by default
(ADR-007). Experiment matrices are refused above `max_runs`; paid runs need `--execute`.

## SLOs

```yaml
slo: {ttft_p95_ms: 500, tpot_p95_ms: 50, e2e_p95_ms: 10000, availability: 0.99}
```

Objectives are reported PASS / FAIL / INSUFFICIENT DATA per run; per-request thresholds
drive goodput and SLO attainment. [docs/methodology/goodput.md](docs/methodology/goodput.md)

## Observability

Prometheus metrics for requests, TTFT/TPOT/E2E histograms, queue depth, backend health,
circuit state, retries, errors, rejections, tokens and router decisions; a provisioned
Grafana dashboard; optional OpenTelemetry spans; JSON logs without prompt text.
[docs/architecture/observability.md](docs/architecture/observability.md)

## Capacity planning

```bash
inferencelab capacity plan configs/capacity/plan_simulated.yaml
inferencelab autoscale simulate configs/autoscaling/*.yaml --output build/autoscale
```

The planner sizes a fleet from the highest **measured** load point that meets the SLO
(never from parameter counts); the autoscaling simulator compares queue-length,
utilisation, request-rate and predictive SLO-aware policies on a trace with startup delay
and replica-hour cost. Both run on CPU from stored profiles.

## Cloud GPU experiments

InferenceLab never provisions machines. Rent a GPU, point a host file at it, and:

```bash
inferencelab remote doctor configs/hosts/my-gpu.local.yaml
inferencelab remote plan   configs/experiments/gpu_first_comparison.yaml --host configs/hosts/my-gpu.local.yaml
inferencelab remote run    configs/experiments/gpu_first_comparison.yaml --host configs/hosts/my-gpu.local.yaml \
                           --wheel dist/inferencelab-0.1.0.dev0-py3-none-any.whl --execute
```

The campaign audits the host, pulls pinned images (recording digests), starts engines on
`127.0.0.1`, waits for health, smoke-tests, runs the load generator **on the host**, copies
results back and removes its containers — leaving the machine for you to terminate.
[Runbook](docs/cloud/gpu-runbook.md) · [first experiment](docs/benchmarks/gpu-first-experiment.md)

## Jenkins

The official CI is the declarative [`Jenkinsfile`](Jenkinsfile): lock check, format, Ruff,
strict mypy, unit / property / integration tests, mock-serving and gateway E2E, benchmark
smoke (including the quick demo), secret scan + bandit, Docker build, package build — all on
CPU, one build at a time. There are no GitHub Actions workflows.
[docs/ci/jenkins.md](docs/ci/jenkins.md)

## Quickstart

```bash
git clone https://github.com/M-amir1998/InferenceLab && cd InferenceLab
uv sync --all-extras            # Python 3.12 venv with dev tools
uv run inferencelab doctor      # CPU-only is normal
uv run inferencelab demo        # simulated engines + gateway + load + report (~2 min, offline)
uv run pytest -q                # ~300 tests, no GPU, no network beyond loopback
```

Run the gateway locally against two simulated engines:

```bash
uv run inferencelab backend serve configs/backends/sim_a.yaml --port 9001 &
uv run inferencelab backend serve configs/backends/sim_b.yaml --port 9002 &
uv run inferencelab gateway serve configs/gateway/local.yaml
curl -N http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"demo-model","messages":[{"role":"user","content":"hello"}],"max_tokens":16,"stream":true}'
```

Or the full CPU stack with Prometheus and Grafana: `docker compose up --build`
(Grafana at http://127.0.0.1:3000; `docker compose --profile load up loadgen` drives a 10-minute simulated session workload).

## Results

### SIMULATED BACKEND RESULTS (CPU, `inferencelab demo`, 2026-11-13)

Produced by the real gateway over loopback HTTP in front of InferenceLab's **simulated**
engines on a laptop. They demonstrate gateway behaviour, not engine performance.
Full report: [reports/local-simulated/report.md](reports/local-simulated/report.md).

Routing on a heterogeneous simulated fleet (fast engine + small engine, Poisson 20 req/s):

| Policy | TTFT p50 | TTFT p95 | Req/s | Goodput |
|---|---|---|---|---|
| round_robin | 1,855 ms | 5,197 ms | 13.36 | 2.97 |
| least_outstanding | 196 ms | 872 ms | 18.92 | 13.99 |
| latency_aware | 147 ms | 276 ms | 19.89 | 19.80 |

Session affinity on multi-turn chat (two identical simulated engines with prefix caching):
engine-reported cached input rose from 73.5% to 81.9% and TTFT p50 fell from 217 ms to
109 ms. Under an injected backend fault, 170/170 requests succeeded via retries and the
circuit breaker. Overload against a bounded queue was shed with 503s while admitted
requests waited at most 1.5 s, and the SLO was reported as FAIL with goodput (1.1 req/s)
far below throughput (9.8 req/s).

### GPU results

Pending. The first comparison (vLLM vs SGLang, Qwen2.5-7B-Instruct BF16, one GPU,
4 workloads × 4 concurrency levels) is fully prepared: [plan](docs/benchmarks/gpu-first-experiment.md).

## Limitations

* No GPU measurements yet; simulator numbers are not predictions of engine performance.
* The simulator models scheduling/caching mechanics with a linear cost model — no kernels,
  CUDA graphs, memory fragmentation or tensor parallelism.
* Gateway quotas, routing statistics and circuit breakers are per process (no shared
  state across gateway replicas in v0.1).
* OpenAI API subset only (no `n>1`, `echo`, multimodal, Assistants/Responses).
* Disaggregated (prefill/decode) serving and multi-node parallelism are represented in
  deployment metadata and launch flags, but not orchestrated end to end.
* The autoscaling simulator is a fluid queue model for comparing policies, not a validated
  Kubernetes model. The Kubernetes manifests are future work.
* Some TensorRT-LLM behaviours (e.g. `ignore_eos` support, structured output without
  `guided_decoding_backend`) are unverified and marked so in the docs.

## Roadmap

1. Run the first GPU comparison (vLLM vs SGLang) and publish measured profiles.
2. Prefix-cache, quantization (with quality checks) and chunked-prefill GPU studies.
3. TensorRT-LLM comparison with a validated configuration.
4. Speculative decoding and multi-LoRA experiments.
5. Optional shared state (Redis) for multi-replica gateways; minimal Kubernetes manifests.
6. Disaggregated serving experiments on multi-GPU hosts.

## Repository layout

```text
src/inferencelab/   domain · backends · openai_api · gateway · routing · admission · limits · reliability
                    workloads · benchmarking · metrics · profiles · capacity · autoscaling · cost
                    experiments · remote · observability · reporting · cli
configs/            models · hardware · deployments · backends · gateway · workloads · slos
                    benchmarks · experiments · capacity · autoscaling · profiles · hosts
tests/              unit · property · integration · e2e
deployment/         docker (GPU engine examples) · prometheus · grafana
scripts/            remote · gpu · ci · dev
docs/               architecture · methodology · backends · benchmarks · cloud · ci · adr
reports/            curated reports (simulated today)
```

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) · Security: [SECURITY.md](SECURITY.md) ·
Changes: [CHANGELOG.md](CHANGELOG.md) · Status: [PROJECT_STATUS.md](PROJECT_STATUS.md)
