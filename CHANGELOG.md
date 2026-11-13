# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow PEP 440.

## [0.1.0.dev0] — unreleased

Local (CPU) framework. `v0.1.0` will be tagged only after the local framework is complete
and reviewed; GPU benchmark results may justify a later release.

### Added
- Domain contracts: requests, responses, error taxonomy, models (pinned revisions),
  deployments, hardware profiles and recorded pricing, workloads, SLOs, request records.
- Backends: `InferenceBackend` protocol and capability discovery; deterministic mock with
  fault injection; CPU continuous-batching simulator (paged KV, prefix caching, chunked
  prefill, preemption); generic OpenAI-compatible HTTP adapter; vLLM 0.33.0, SGLang 0.5.23
  and TensorRT-LLM 1.2.1 adapters with metric maps and verified launch builders; serving
  app for local backends.
- Gateway: OpenAI-compatible chat/completions with SSE streaming, auth, request limits,
  tenant quotas, bounded priority admission with load shedding, circuit breakers,
  first-token/idle/overall timeouts, bounded retries with a retry budget, client-disconnect
  cancellation, Prometheus metrics, optional OpenTelemetry tracing.
- Routing: round robin, random, least outstanding, peak-EWMA latency-aware, cost-aware,
  SLO-aware; prefix/session affinity (rendezvous hashing, bounded loads); tiered fallback.
- Benchmarking: async load generator (closed/open loop, Poisson, burst, step, trace;
  sessions; cancellation; coordinated-omission accounting), runner, artifacts, metrics with
  sample-size-aware percentiles, goodput, SLO evaluation, cost model, reports and plots.
- Planning: performance profiles and registry, capacity planner, autoscaling simulator.
- Experiments: capped matrices, duration/cost estimates, local execution.
- Remote: cloud-neutral SSH execution, GPU doctor, campaign runner with dry-run planning.
- CLI covering doctor, demo, backends, models, gateway, workloads, benchmarks, profiles,
  capacity, autoscaling, experiments, remote, GPU doctor, quality, reports.
- Jenkins pipeline, Docker image (non-root), Compose stack with Prometheus and Grafana.
- Documentation: architecture, methodology, backends, cloud runbook, CI, ADR-001…008.
- Simulated local demo results (`reports/local-simulated/`).
