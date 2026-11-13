# Cost methodology

## Prices are inputs, never constants

Cloud GPU prices change constantly and differ by provider, region, commitment and spot
market. InferenceLab therefore ships **no prices**. A price is supplied per experiment,
with where and when it was observed:

```yaml
hardware:
  gpu: H100-80GB-SXM
  count: 1
  hourly_cost_usd: <the price you were quoted>
  price_unit: per_instance_hour      # or per_gpu_hour
  price_source: <provider console / invoice URL / "manual">
  price_recorded_at: <YYYY-MM-DD>
```

A price without `price_source` is a validation error. Every cost figure in a report carries
the source and date.

## Formulas

```text
effective hourly = hourly_cost_usd × (gpu_count if per_gpu_hour else 1)
run cost         = effective hourly × measurement duration (hours)

cost / 1M input tokens          = run cost / input tokens × 1e6
cost / 1M output tokens         = run cost / output tokens × 1e6
cost / 1M total tokens          = run cost / (input + output) × 1e6
cost / successful request       = run cost / successful requests
cost / SLO-compliant request    = run cost / compliant requests
```

Per-token figures are **all-in**: prefill and decode share the same GPUs, so cost cannot
be split between input and output honestly. Each figure answers "what did a million of
these tokens cost in this run, everything included".

## What run cost does not include

Idle time between runs, model download and load (minutes for a 7B model), failed attempts,
storage, egress and the machine's minimum billing increment. The experiment planner's cost
estimate is therefore an explicit **upper bound** for the campaign:

```text
upper bound = hourly × (runs × max_run_duration_s + deployments × model_load_estimate_s
                        + runs × per_run_overhead_s) / 3600
```

and is shown before any paid run, which requires `--execute`.

## Capacity cost

The capacity planner multiplies required replicas by the per-replica hourly price and
reports cost per 1M output tokens at the target rate. It inherits the profile's caveats:
a simulated profile produces a simulated cost.
