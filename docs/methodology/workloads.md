# Workloads

A workload fixes everything about the traffic, so two deployments are compared under
exactly the same requests (`configs/workloads/`).

## Dimensions

| Dimension | Options |
|---|---|
| Prompt / output length | `fixed`, `uniform`, weighted `buckets`, or a JSONL **trace** of shapes |
| Arrival | `closed_loop` (N users), `constant`, `poisson`, `burst`, `step` (piecewise rate, optionally Poisson), `trace` timestamps |
| Shared prefix | `num_prefixes` × `prefix_tokens`, uniform or Zipf popularity, `prefix_key` routing hint |
| Sessions | multi-turn chat: number of turns, think time, system prompt; history (including generated answers) is resent each turn and `conversation_id` is set |
| Tenants / priorities | weighted mixes for quota and admission studies |
| Sampling | temperature (default 0), top_p, `ignore_eos` (exact output lengths) |
| Phases | warmup and cooldown by request count or seconds |
| Bounds | `num_requests`, `duration_s` or the trace length (unbounded runs are rejected), `max_in_flight`, `request_timeout_s` |
| Cancellation | `cancel_probability` + `cancel_after_tokens` (client disconnects mid-stream) |

Every random choice derives from `seed` through *separate* streams (lengths, arrivals,
prefixes, assignment), so changing the arrival rate does not change the prompts.

## Provided profiles

| Workload | Shape | Studies |
|---|---|---|
| `short_short` | 128–256 in / 64–128 out | interactive chat, latency |
| `short_long` | 128–256 in / 768–1024 out | decode-bound throughput, TPOT |
| `long_short` | 3–4k in / 64–128 out | prefill-bound, TTFT, chunked prefill |
| `long_long` | 3–4k in / 768–1024 out | KV-cache capacity, preemption |
| `shared_prefix` | 4 × 2048-token prefixes (Zipf) + 32–128 token queries | prefix caching on vs off |
| `chat_sessions` | 4–8 turns, Poisson conversations | session affinity, history reuse |
| `mixed_poisson` | 70/25/5 % length buckets, open loop | production-like latency under load |
| `step_load` | 1→2→4→8 req/s steps | finding the breaking point |
| `structured_json` | `response_format: json_schema` | constrained decoding (valid-JSON rate with `capture_text`) |
| `trace_replay` | JSONL shapes + timestamps | replaying anonymised production shapes |

## Synthetic prompts

Prompts are built from a fixed vocabulary of common short English words that are single
tokens in typical BPE vocabularies, so *N* requested tokens ≈ *N* real tokens. The engine's
reported `prompt_tokens` is the ground truth and is what the summary uses; the requested
target is kept alongside for comparison. Real prompt data is never required.

## Traces

```json
{"timestamp_s": 0.42, "input_tokens": 812, "output_tokens": 128, "conversation_id": "c-17"}
```

Only token counts, timestamps and opaque ids are read. A trace line with `prompt`,
`messages`, `text`, `completion` or similar fields is rejected, so private logs cannot be
ingested (or committed) by accident. Trace files are git-ignored except the synthetic
example.

## Measuring prefix caching fairly

* Use `shared_prefix` (or `chat_sessions`) and compare cache-on vs cache-off deployments of
  the **same** engine on the **same** GPU.
* Warm up: the first request for each prefix is always a miss.
* Report TTFT (where caching helps most), throughput, and the engine's own hit ratio
  (`vllm:prefix_cache_hits_total / queries_total`, `sglang:cache_hit_rate`,
  `trtllm_kv_cache_hit_rate`), plus per-request `cached_tokens` when the engine reports it.
