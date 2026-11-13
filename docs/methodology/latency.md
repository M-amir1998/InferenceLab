# Latency and throughput definitions

All timestamps are taken by the **client** (the benchmark load generator) on a monotonic
clock (`time.perf_counter`), relative to the run start. Definitions match
`vllm bench serve` so numbers are comparable with the wider ecosystem.

| Metric | Definition | Notes |
|---|---|---|
| **TTFT** — time to first token | first *content-bearing* SSE chunk − request send | role-only and usage-only chunks are ignored |
| **E2E** — end-to-end latency | last content-bearing chunk − request send | the trailing usage chunk and `[DONE]` are excluded |
| **TPOT** — time per output token | (E2E − TTFT) / (output_tokens − 1), per request | undefined for single-token outputs; `output_tokens` from the engine's usage when reported |
| **ITL** — inter-token latency | gaps between consecutive content-bearing chunks, pooled over requests | a chunk carrying several tokens (speculative decoding, batched streaming) counts as one gap, so ITL ≠ TPOT in general |
| **E2E from schedule** | last content chunk − *intended* send time | equals E2E unless the client lagged; the gap exposes coordinated omission |
| **Schedule lag** | actual send − intended send | reported, and a warning is raised when it is large |
| **Queue time (gateway)** | time in the gateway's admission queue | from the `x-inferencelab-queue-ms` response header |
| **Queue / prefill / decode time (engine)** | as reported by the engine | per request from the simulator; for vLLM/SGLang/TRT-LLM, window means from the engines' Prometheus histograms (`_sum`/`_count` deltas over the run). Never estimated. |

## Throughput

* **Measurement window**: requests in the `measure` phase (warmup/cooldown excluded).
* **Duration**: first send → last token among measured requests (makespan).
* **Request throughput**: successful requests / duration.
* **Token throughput**: input, output and total tokens of successful requests / duration.
* Failed, rejected, cancelled and dropped requests never count as throughput.

## Percentiles and sample size

Percentiles use linear interpolation between order statistics (NumPy `linear`,
Hyndman–Fan type 7). A percentile *q* is reported only when at least **5 samples lie
beyond it**, i.e. `n · (1 − q/100) ≥ 5`:

| Percentile | Minimum samples |
|---|---|
| p50 | 10 |
| p90 | 50 |
| p95 | 100 |
| p99 | 500 |

Below that, the value is shown as `n/a` and SLO objectives on it are
`INSUFFICIENT DATA`. A "p99" from 40 requests is the maximum of a tiny sample, not a tail
estimate, and reporting it would invite false conclusions.

## Clock and timer caveats

* On Windows the default timer granularity is ≈15.6 ms; local demo latencies inherit
  that jitter. Linux CI agents and GPU hosts do not have this limitation.
* The CPU simulator keeps its own engine time and stays drift-free, so simulated
  throughput is not distorted by coarse timers; only token *delivery* is.
* GPU benchmarks run the load generator **on the GPU host** so WAN latency does not enter
  TTFT.
