# Benchmarks

## Status

| Kind | Status |
|---|---|
| **GPU benchmarks** (vLLM, SGLang, TensorRT-LLM) | **pending** — no GPU results have been measured yet; none are claimed anywhere in this repository |
| **Simulated local results** | available: [reports/local-simulated/report.md](../../reports/local-simulated/report.md) (CPU simulator, SIMULATED BACKEND RESULTS) |

## What exists today

* **Local demo** (`inferencelab demo`): routing policies on a heterogeneous simulated fleet,
  session affinity and prefix-cache reuse, circuit breaking under an injected fault, tenant
  rate limits, backpressure and SLO/goodput accounting — real runs of the real gateway over
  loopback HTTP against simulated engines.
* **Simulated profiles** (`configs/profiles/`): load sweeps of the two simulated engines,
  produced by `configs/experiments/local_routing_capacity.yaml`; consumed by the capacity
  and autoscaling examples. They demonstrate the method only.

## First GPU experiment (prepared, not run)

See [gpu-first-experiment.md](gpu-first-experiment.md) and the
[GPU runbook](../cloud/gpu-runbook.md).

## Reading results

* `SIMULATED BACKEND RESULTS` — CPU models; never comparable with GPU numbers.
* `GPU INFERENCE RESULTS` — set only when every serving backend is a GPU engine type.
* `UNVERIFIED BACKEND TYPE` — the target could not be identified (e.g. a generic OpenAI
  endpoint); treat with caution.
* `n/a` — not measured or too few samples (see docs/methodology/latency.md).
