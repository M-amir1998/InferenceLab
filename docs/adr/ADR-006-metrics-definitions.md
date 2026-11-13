# ADR-006 — Metric definitions

**Status:** accepted · **Date:** 2026-11-13

## Context

Latency metrics are defined inconsistently across tools. Small differences — whether E2E
includes the trailing usage chunk, how ITL treats multi-token chunks, what duration
throughput divides by — make results incomparable.

## Decision

* Match `vllm bench serve`: TTFT to the first content chunk; E2E to the last content chunk;
  TPOT = (E2E − TTFT)/(output − 1) per request; ITL per content-chunk gap; throughput =
  successful work / makespan of the measurement window.
* Measure on the client with a monotonic clock; also record the intended send time and
  report schedule lag and E2E-from-schedule to expose coordinated omission.
* Report a percentile only when at least 5 samples lie beyond it (p99 needs 500 samples).
* Goodput = successful SLO-compliant requests per second, always next to throughput.
* Token counts come from the engine's usage; estimates are labelled `estimated`.
* Engine-internal timings come only from the engine (the simulator per request; real
  engines via Prometheus histogram deltas) and are never derived from client timings.

## Consequences

* Results are comparable with the ecosystem, and small samples cannot masquerade as tails.
* Short runs show `n/a` for tail percentiles; experiments must be sized accordingly
  (docs/methodology/latency.md).
