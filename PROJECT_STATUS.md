# Project status

Snapshot at the end of the local (CPU) build phase, 2026-11-13. Every figure below comes
from a command that was actually run; nothing is extrapolated.

| | |
|---|---|
| **Repository** | https://github.com/M-amir1998/InferenceLab (public, branch `main`) |
| **Version** | `0.1.0.dev0`, unreleased. `v0.1.0` is not tagged |
| **Licence** | Apache-2.0 |
| **GPU benchmarks** | **Pending.** No GPU run has been performed and no GPU numbers exist |

## Architecture

```text
client ─HTTP/SSE─▶ gateway: auth → body/request limits → tenant quotas → admission queue
                   → router (policy + affinity + tiers) → circuit breaker → retries, timeouts
                   ─HTTP─▶ engines: vLLM · SGLang · TensorRT-LLM · OpenAI-compatible · simulated · mock

control plane:  workloads → load generator → runner → metrics + SLOs → profiles
                → capacity planner · autoscaling simulator · experiments → remote campaigns (SSH)
observability:  Prometheus · Grafana · JSON logs · optional OpenTelemetry
```

Engines are reached over HTTP only (ADR-001); the data plane never imports engine code.
Details: [docs/architecture/](docs/architecture/) (8 documents), [docs/methodology/](docs/methodology/)
(5), [docs/backends/](docs/backends/) (4) and ADR-001 to ADR-008 in [docs/adr/](docs/adr/).

## Components

| Area | Implemented | Verified by |
|---|---|---|
| Backends | Deterministic mock with fault injection; simulated engine (continuous batching, paged KV blocks, prefix caching, chunked prefill, preemption by recompute); generic OpenAI-compatible adapter; vLLM 0.33.0, SGLang 0.5.23 and TensorRT-LLM 1.2.1 adapters with metric maps and launch-plan builders | Mock and simulator: unit, property and end-to-end tests. Engine adapters: tests against mocked HTTP and flags checked against the engines' docs. **Not yet run against live engines** |
| Gateway | `/v1/chat/completions`, `/v1/completions`, `/v1/models` with SSE streaming and usage chunks; API keys (constant-time); body-size, message and `max_tokens` limits; `/health`, `/ready`, `/metrics`, backend state; client disconnect cancels engine work | Integration and end-to-end tests; Compose smoke test |
| Routing | `round_robin`, `random`, `least_outstanding`, `latency_aware` (peak EWMA), `cost_aware`, `slo_aware`; prefix/session affinity (rendezvous hashing, bounded load); tiered fallback | Unit and property tests; demo |
| Reliability | Circuit breaker (consecutive and failure-rate trips, half-open probes); bounded retries with backoff, jitter and a retry budget (streams retried only before the first output); connect, first-token, idle and overall timeouts | Failure injection over loopback HTTP; demo |
| Admission and limits | Token buckets; tenant quotas (requests/s, concurrency, tokens/min); bounded priority queue with backpressure; shedding by newest, lowest priority or predicted SLO miss | Unit and property tests; demo |
| Workloads | 14 workload configs; closed-loop, constant, Poisson, burst, step and trace arrivals; shared prefixes; multi-turn sessions; trace replay without prompt text | Unit tests |
| Metrics | TTFT, TPOT, ITL, E2E; a percentile is reported only with at least 5 samples beyond it (p50 needs 10, p99 needs 500); request and token throughput; goodput; error rates; schedule lag (coordinated omission); engine metric deltas; GPU memory sampling; cost per 1M tokens, per request and per SLO-compliant request | Unit and property tests |
| SLO system | Per-metric percentile objectives, per-request compliance, PASS / FAIL / INSUFFICIENT_DATA, attainment; 4 SLO configs | Unit and property tests |
| Capacity planner | Profile-driven replica count with headroom, redundancy, optional interpolation, workload-mismatch and SIMULATED warnings | Unit tests; result below |
| Autoscaling simulator | Queue-length, utilisation, request-rate and SLO-aware policies over piecewise, diurnal or file traces with replica startup delay | Unit tests; results below |
| Observability | Gateway and engine Prometheus metrics, Grafana dashboard (15 panels), JSON logs, optional OpenTelemetry tracing | Compose smoke test |
| Remote GPU tooling | SSH executor, GPU doctor, campaign runner with dry-run plans. Never creates, resizes or deletes machines | Unit tests with dry-run executors; `remote plan` dry run |

## Quality gates

The full Jenkins stage list ran on a fresh clone at `58224c1` (Windows 11, Git Bash, Docker
Desktop 29.8.1). Lint, types, tests and secret scans were re-run at `0cdf724`, which differs
only in documentation and the gitleaks config.

| Gate | Result |
|---|---|
| Tests | **297 passed, 0 failed, 0 skipped**: 239 unit, 15 property, 24 integration, 19 end-to-end |
| Coverage | 88% (line + branch); the pipeline gate is 85% |
| Ruff 0.17.5 | `ruff format --check` and `ruff check` clean |
| mypy 2.4.1 | strict, 151 files, no issues |
| Package build | `uv build` produces `inferencelab-0.1.0.dev0.tar.gz` and `inferencelab-0.1.0.dev0-py3-none-any.whl` |
| Secret scan | detect-secrets 1.5.0 against an audited baseline (6 false positives: env-var names, public model revision SHAs, dummy test keys): clean. gitleaks 8.28.0 (container) over the full history: no leaks |
| bandit 1.9.4 | `src`: no issues |
| Lock file | `uv lock --check` passes |

## Docker

| Check | Result |
|---|---|
| Image | Builds (multi-stage, `python:3.12.16-slim-bookworm`, uv 0.11.21); 583 MB; runs as uid 10001; `--version` prints `0.1.0.dev0`; no tests or `.env` in the image |
| Compose stack | Two simulated engines, gateway, Prometheus v3.15.1 (3/3 targets up) and Grafana 12.4.13 (dashboard and datasource provisioned) start; health, models, chat, streaming and `/metrics` checked |
| Compose load profile | 10-minute session workload: 7,367 requests, all OK (simulated) |
| GPU engine Compose files | `deployment/docker/{vllm,sglang,trtllm}.compose.yaml`: not run (no GPU) |

## Jenkins

`Jenkinsfile` (declarative): Checkout, Environment Validation, Dependency Lock Validation,
Format, Ruff, Strict mypy, Unit Tests, Property Tests, Integration Tests, Mock Serving E2E,
Gateway E2E (with the coverage gate), Benchmark Smoke Test, Security Scan, Docker Build
(skipped without a Docker daemon) and Python Package Build. JUnit reports, coverage and
artifacts are archived. It needs no GPU, CUDA, cloud credentials or model downloads.

Status: **not yet run on a Jenkins controller** (none was available). Every stage command
passed locally, as listed above. GitHub Actions is not used; there is no `.github/workflows`.
Setup: [docs/ci/jenkins.md](docs/ci/jenkins.md).

## Git

All commits on `main` are authored and committed by `M-amir1998`, the existing local Git
identity. There are no co-author trailers or generated-by metadata, and history has never
been force-pushed.

## Local simulated results

> **SIMULATED BACKEND RESULTS.** Produced on a CPU laptop by InferenceLab's simulated
> engines behind the real gateway over loopback HTTP. They show gateway behaviour, not
> engine performance, and say nothing about vLLM, SGLang or TensorRT-LLM.

From `inferencelab demo` ([full report](reports/local-simulated/report.md)):

Routing on a heterogeneous fleet (fast engine + small engine, Poisson 20 req/s):

| Policy | TTFT p50 | TTFT p95 | Req/s | Goodput (req/s) |
|---|---|---|---|---|
| round_robin | 1,855 ms | 5,197 ms | 13.36 | 2.97 |
| least_outstanding | 196.1 ms | 871.9 ms | 18.92 | 13.99 |
| latency_aware | 146.9 ms | 275.5 ms | 19.89 | 19.80 |

| Scenario | Result |
|---|---|
| Session affinity, 2 identical engines, multi-turn chat | Engine-reported cached input 73.5% → 81.9%; TTFT p50 216.5 → 108.6 ms; p95 396.8 → 291.1 ms |
| Injected backend fault (4 s of errors) | 170/170 requests succeeded; 5 gateway retries; circuit opened twice |
| Tenant rate limits (free 3 req/s, pro 50 req/s, ≈ 8 req/s offered each) | free: 54 sent, 20 OK, 34 rejected (429); pro: 58 sent, 58 OK |
| Overload, one slow engine, Poisson 40 req/s | 9.79 req/s served, goodput 1.10 req/s; 266 queue-full and 69 shed; queue wait p50/p90 1,360 / 1,484 ms; TTFT p95 2,435 ms; SLO FAIL |
| Compose load profile (2 simulated engines, Poisson 3 sessions/s) | 12.06 req/s; TTFT p50/p95/p99 44.2 / 84.9 / 108.0 ms; TPOT p95 32.86 ms; SLO pass |

Capacity plan (`inferencelab capacity plan configs/capacity/plan_simulated.yaml`): for 60
req/s with 20% headroom and +1 redundancy, **6 replicas** (17.08 req/s sustainable per
replica, at concurrency 32). The profile is simulated, so this is not a GPU capacity estimate.

Autoscaling on a diurnal trace (`inferencelab autoscale simulate configs/autoscaling/diurnal_<policy>.yaml`):

| Policy | SLO-violating requests | Mean replicas | Replica-hours | Scale events |
|---|---|---|---|---|
| queue_length | 0.00% | 4.56 | 9.12 | 18 |
| request_rate | 0.00% | 5.51 | 11.01 | 16 |
| slo_aware | 0.83% | 3.72 | 7.45 | 12 |
| utilization | 0.00% | 4.73 | 9.46 | 14 |

The dollar amounts these commands print use an explicit `$1.00/h` example placeholder,
not a real price.

## GPU benchmark status

Pending. Nothing has been run on a GPU, and InferenceLab creates no cloud resources. The
first comparison is fully prepared: vLLM 0.33.0 vs SGLang 0.5.23, Qwen2.5-7B-Instruct at
revision `a09a35458c702b33eeacc393d103063234e8bc28`, BF16, one GPU; workloads short/short,
long/short, short/long and shared-prefix at concurrency 1, 8, 32 and 128 (≈ saturation).
It reports TTFT, TPOT, throughput, goodput, GPU memory and cost. See
[docs/benchmarks/gpu-first-experiment.md](docs/benchmarks/gpu-first-experiment.md).
TensorRT-LLM follows once its configuration is validated as comparable.

## Cloud experiment configs

| Experiment | Runs | Upper-bound duration | Varies |
|---|---|---|---|
| `gpu_first_comparison` | 32 (cap 40) | 516 min | vLLM vs SGLang × 4 workloads × 4 concurrency levels |
| `gpu_prefix_cache` | 8 | 164 min | Prefix caching on/off, both engines |
| `gpu_chunked_prefill` | 8 | 164 min | Prefill chunking on long prompts |
| `gpu_quantization` | 16 | 288 min | BF16 vs AWQ, both engines |
| `gpu_trtllm` | 12 | 206 min | vLLM vs TensorRT-LLM, BF16 |

Upper bounds come from `inferencelab experiment plan` (per-run cap of 900 s plus per-run
overhead and model loads). They also need 9 GPU deployments plus 3 examples (speculative decoding, LoRA,
SGLang prefill/decode), digest-pinned engine images, 8 hardware profiles without prices,
a host template ([configs/hosts/example.yaml](configs/hosts/example.yaml)), a
[runbook](docs/cloud/gpu-runbook.md) and [provider notes](docs/cloud/providers.md).

## Exact commands for the first GPU benchmark

Rent a Linux GPU host yourself (driver R580+, Docker, NVIDIA container toolkit). Then:

```bash
# 0. Host file (git-ignored): set hostname, user and key path
cp configs/hosts/example.yaml configs/hosts/my-gpu.local.yaml

# 1. Machine audit
scp scripts/remote/bootstrap.sh my-gpu:~ && ssh my-gpu 'bash bootstrap.sh'
inferencelab remote doctor configs/hosts/my-gpu.local.yaml \
    --container-test nvidia/cuda:13.0.0-base-ubuntu24.04

# 2. Record your price in configs/experiments/gpu_first_comparison.yaml (pricing block), then plan
uv build
inferencelab experiment plan configs/experiments/gpu_first_comparison.yaml
inferencelab remote plan configs/experiments/gpu_first_comparison.yaml \
    --host configs/hosts/my-gpu.local.yaml      # dry run: prints every command

# 3. Smoke one deployment first (configs/experiments/gpu_smoke.local.yaml, see runbook step 3)
inferencelab remote run configs/experiments/gpu_smoke.local.yaml \
    --host configs/hosts/my-gpu.local.yaml \
    --wheel dist/inferencelab-0.1.0.dev0-py3-none-any.whl --execute

# 4. The comparison (32 runs)
inferencelab remote run configs/experiments/gpu_first_comparison.yaml \
    --host configs/hosts/my-gpu.local.yaml \
    --wheel dist/inferencelab-0.1.0.dev0-py3-none-any.whl --execute

# 5. Compare and report
inferencelab benchmark compare <vllm run> <sglang run> --results results/remote/my-gpu/…
inferencelab report <run>
```

`remote run` refuses to start without `--execute`. Stopping or deleting the machine is
left to you.

## Known limitations

* No GPU measurements yet, and the engine adapters have not been run against live
  engines. Simulator numbers are not predictions of engine performance.
* The simulator uses a linear cost model: no kernels, CUDA graphs, memory fragmentation
  or tensor parallelism.
* The Jenkins pipeline has not run on a Jenkins controller. It was checked locally on
  Windows, whereas the target agents are Linux.
* Gateway quotas, routing statistics and circuit breakers are per process; there is no
  shared state across gateway replicas.
* Only a subset of the OpenAI API is supported (no `n>1`, `echo`, multimodal,
  Assistants or Responses).
* Prefill/decode disaggregation and multi-node parallelism exist as deployment metadata
  and launch flags only; they are not orchestrated end to end.
* The autoscaling simulator is a fluid queue model for comparing policies, not a
  validated Kubernetes model. Kubernetes manifests are future work.
* Some TensorRT-LLM behaviours (e.g. `ignore_eos`, structured output) are unverified and
  marked as such in the docs.
* Coverage gaps:
  * `demo.py` runs in the CI smoke stage, not under pytest.
  * `gpu_monitor` (48%, needs `nvidia-smi`), `system_info` (61%) and
    `experiments/runner` (62%) are the least-covered modules.
