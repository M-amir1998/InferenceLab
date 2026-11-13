# Architecture overview

InferenceLab is the layer **around** production inference engines. It never reimplements
vLLM, SGLang or TensorRT-LLM; it turns *models + engines + deployment configurations +
traffic + routing + SLOs + cost* into reproducible engineering experiments, and provides
the data-plane pieces (gateway, router, admission control) needed to study them.

```text
                     Client workloads (benchmark client / real users)
                                     │  OpenAI-compatible HTTP (SSE)
                                     ▼
 ┌─────────────────────────── Inference gateway (data plane) ───────────────────────────┐
 │ body limit ─► auth ─► request validation ─► tenant quotas ─► admission (bounded       │
 │ priority queue, load shedding) ─► tiered router ─► attempt loop (circuit breakers,    │
 │ first-token/idle/overall timeouts, bounded retries) ─► SSE stream to the client       │
 └───────────────┬───────────────────────┬──────────────────────────┬────────────────────┘
                 ▼                       ▼                          ▼
           vLLM server             SGLang server            trtllm-serve        (GPU hosts,
       (OpenAI HTTP API)         (OpenAI HTTP API)        (OpenAI HTTP API)      HTTP only)
                 │                       │                          │
                 └──────────── engine /metrics (Prometheus) ────────┘
                                     │
          Gateway /metrics ──► Prometheus ──► Grafana      OpenTelemetry spans (optional)
                                     │
 ┌──────────────────────────────── Control plane ─────────────────────────────────────────┐
 │ model registry · deployments · hardware profiles · workloads · SLOs · experiment       │
 │ matrices (capped, cost-guarded) · benchmark runner · profiles · capacity planner ·     │
 │ autoscaling simulator · cost model · reports · remote (SSH) campaign runner            │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

## Two subsystems

| | Data plane | Control plane |
|---|---|---|
| Handles | requests, streaming, routing, timeouts, retries, admission, load shedding, backend I/O | models, deployments, backend configs, traffic policies, experiments, benchmarks, SLOs, results, cost |
| Runs | continuously (gateway process) | per experiment (CLI, CI, remote campaign) |
| Code | `gateway/`, `routing/`, `admission/`, `limits/`, `reliability/`, `backends/` | `domain/`, `config/`, `workloads/`, `benchmarking/`, `experiments/`, `profiles/`, `capacity/`, `autoscaling/`, `cost/`, `remote/`, `reporting/` |

The boundary is enforced in code: the data plane depends only on `domain` contracts and
backend interfaces; nothing in the request path imports experiment, reporting or
remote-execution code. See [data-plane.md](data-plane.md) and [control-plane.md](control-plane.md).

## Package map

| Package | Responsibility |
|---|---|
| `domain/` | Typed contracts: requests, responses, errors, models, deployments, hardware/pricing, workloads, SLOs, per-request records |
| `backends/` | `InferenceBackend` protocol; mock, CPU simulator, generic OpenAI HTTP adapter, vLLM/SGLang/TensorRT-LLM adapters + launch builders; serving app for local backends |
| `openai_api/` | Wire format shared by every hop: schemas, SSE encode/parse, response builders |
| `gateway/` | FastAPI shell + framework-independent `InferenceService`, backend pool, auth, limits |
| `routing/` | Policies, prefix affinity (rendezvous hashing with bounded loads), tiered fallback, EWMA stats |
| `reliability/` | Circuit breaker, bounded retries + retry budget, timeouts |
| `admission/`, `limits/` | Bounded priority queue with shedding; token buckets and tenant quotas |
| `workloads/`, `benchmarking/` | Workload generation and the async load generator; runner and artifacts |
| `metrics/` | Metric definitions, percentiles with sample-size rules, summaries, goodput, SLO evaluation |
| `profiles/`, `capacity/`, `autoscaling/`, `cost/` | Measured profiles and the planning tools that consume them |
| `experiments/`, `remote/` | Capped experiment matrices; cloud-neutral SSH execution |
| `observability/`, `reporting/` | Prometheus, OpenTelemetry, logging; reports, tables, plots |
| `clock.py` | `SystemClock` and the deterministic `VirtualClock` used by tests and simulations |

## Design principles

1. **HTTP to engines** (ADR-001): version independence, failure isolation, no GPU
   dependencies in the core package.
2. **No GPU required for development or CI** (ADR-004): everything is testable with the
   mock backend and the CPU continuous-batching simulator.
3. **Injected time**: all timing goes through a `Clock`; the `VirtualClock` makes
   timeouts, rate limits, breakers and the simulator deterministic in tests.
4. **Never fabricate**: engine metrics that are not exposed stay `None`; estimated token
   counts are labelled; simulated results are labelled; percentiles are withheld when the
   sample is too small; cost is computed only from recorded prices.
5. **Bounded everything**: queues, retries, in-flight client tasks, experiment matrices.
