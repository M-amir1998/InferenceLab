# Control plane

The control plane describes *what* is being served and measured. All of it is declarative,
validated YAML (`configs/`) backed by Pydantic models that reject unknown keys, so a typo
such as `prefix_cachng: true` fails loudly instead of silently changing an experiment.

## Objects

| Object | Module | Holds |
|---|---|---|
| Model | `domain/model.py` | logical name, checkpoint, **commit-SHA revision**, tokenizer, architecture, parameters, context, weight format, license |
| Deployment | `domain/deployment.py` | model + backend + pinned image + hardware + quantization + parallelism (TP/PP/DP/EP) + context/concurrency/batch limits + prefix caching + chunked prefill + LoRA + speculative decoding + role (unified/prefill/decode) |
| Hardware profile | `domain/hardware.py` | specifications only (VRAM, bandwidth, architecture) — never prices |
| Pricing | `HardwareRef` | hourly price **with its source and observation date**, supplied per experiment |
| Workload | `domain/workload.py` | length distributions, arrival process, shared prefixes, sessions, traces, tenants, priorities, warmup/cooldown, seed |
| SLO | `domain/slo.py` | distribution objectives (`ttft_p95_ms`…), availability, per-request thresholds |
| Experiment | `experiments/spec.py` | matrix of deployments/backends × workloads × load levels × repeats, `max_runs`, per-run duration cap |
| Profile | `profiles/model.py` | measured curve of load → throughput/latency for one deployment and workload |

**Model ≠ deployment.** One checkpoint can be served as vLLM+BF16, vLLM+AWQ and
SGLang+AWQ; those are three deployments, benchmarked and compared separately.

## Capability discovery

Every backend exposes `BackendCapabilities` (streaming, structured output, tool calling,
LoRA, multi-LoRA, prefix caching, chunked prefill, speculative decoding, quantization
formats, embeddings, distributed inference, disaggregated serving, cancellation, engine
metrics). Launch builders additionally reject deployments an engine cannot express —
e.g. TensorRT-LLM with a Hugging Face AutoAWQ checkpoint, vLLM disaggregation without a KV
connector — instead of producing a different experiment than the one configured.

## Experiment lifecycle

```text
experiment.yaml ─► expand (product, excludes) ─► refuse if > max_runs
               ─► estimate duration (open-loop schedules) and upper-bound cost (recorded price)
               ─► local mode: run in-process against mock/simulated backends (optionally virtual time)
               ─► remote mode: requires --execute; SSH campaign on a GPU host (cloud-execution.md)
               ─► per run: results/<run-id>/ artifacts ─► merged profiles ─► experiment report
```

## Planning tools (CPU-only, profile-driven)

* **Capacity planner** — replicas for a target rate under an SLO, from the highest measured
  load point that meets the SLO (never from parameter counts or datasheets).
* **Autoscaling simulator** — compares queue-length, utilisation, request-rate and
  predictive SLO-aware policies on a load trace with startup delay, stabilisation windows
  and replica-hour cost. A policy-comparison tool, not a Kubernetes model.
* **Cost model** — all-in cost per 1M input/output/total tokens, per successful request and
  per SLO-compliant request, from recorded prices only.
