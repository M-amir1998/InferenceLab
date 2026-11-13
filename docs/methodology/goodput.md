# Goodput and SLOs

## Two views of an SLO

```yaml
slo:
  name: interactive
  ttft_p95_ms: 500        # distribution-level objectives…
  tpot_p95_ms: 50
  e2e_p95_ms: 10000
  availability: 0.99
  # request_thresholds: {ttft_ms: 500, tpot_ms: 50}   # optional explicit per-request bounds
```

* **Objectives** are evaluated on the run's measured distribution: each is `PASS`,
  `FAIL` or `INSUFFICIENT DATA` (see the sample-size rule in latency.md). Availability needs
  roughly `1 / (1 − target)` attempts before it can be judged at all (≈1,000 for 99.9%).
* **Per-request thresholds** classify every successful request as compliant or not.
  Unless set explicitly they come from the highest-percentile objective of each metric
  (`ttft_p95_ms: 500` ⇒ each request needs TTFT ≤ 500 ms). ITL objectives are
  distribution-level only.

A request is compliant when it succeeded and meets every configured threshold; a
threshold on an undefined metric (TPOT of a one-token answer) is not violated, but a
missing TTFT/E2E is.

## Goodput

> **Goodput** = successful, SLO-compliant requests per second of the measurement window.

* Throughput and goodput are always reported side by side; a system that answers fast
  but late has high throughput and low goodput.
* **SLO attainment** = compliant / eligible requests, where eligible excludes requests the
  workload cancelled deliberately and requests the client never sent.
* Goodput is also reported in output tokens/s of compliant requests.

## Sweeps: maximum sustainable load

A single run gives one point. Capacity questions need the **largest load that still meets
the SLO**, found from a sweep (closed-loop concurrency or open-loop request rate). The
capacity planner uses the highest-throughput *measured* point whose percentiles meet the
objectives — the same idea as DistServe's per-GPU goodput — and optionally interpolates
toward the first failing point.

## Wording

Reports state what was *measured in this run* ("TTFT p95 was 320 ms; the objective was
met"). They never claim guarantees. SIMULATED results are labelled as such.
