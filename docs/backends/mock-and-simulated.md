# Mock and simulated backends

Both run on CPU, need no model weights, and power the tests, the CI smoke stage, the
Compose stack and `inferencelab demo`. Results they produce are always labelled
**SIMULATED BACKEND RESULTS**.

## MockBackend

Fixed TTFT and per-token latency with optional seeded jitter. Every random decision is
drawn from a PRNG seeded by `(seed, request_id)`, so the same request always gets the same
latency and the same injected fault regardless of concurrency.

```yaml
type: mock
name: mock-flaky
latency: {ttft_ms: 40, tpot_ms: 15, jitter: 0.1}
output_tokens: null            # honour max_tokens
failure_probability: 0.01      # HTTP 500 before the first token
timeout_probability: 0.005     # hang until the caller's timeout fires
disconnect_probability: 0.0    # break the stream mid-way
malformed_probability: 0.0     # emit an unparseable chunk mid-way
max_concurrency: null          # 503 when exceeded
faults:                        # time-windowed faults relative to backend start
  - {mode: down, start_s: 30, end_s: 60}   # health fails, requests refused
```

Served over HTTP (`inferencelab backend serve CONFIG --admin`), faults are realistic at the
protocol level: `malformed` writes broken SSE JSON and `disconnect` drops the connection,
so the HTTP adapter sees exactly what a misbehaving engine produces.
`POST /admin/fault {"mode": "error", "end_s": 5}` injects a fault at runtime;
`{"clear": true}` removes it.

## SimulatedInferenceBackend

A deterministic **systems model** of a continuous-batching engine — not a GPU performance
predictor:

* **Continuous batching** with `max_num_seqs` and a per-step token budget
  (`max_num_batched_tokens`).
* **Paged KV cache** of `num_kv_blocks × block_size` tokens.
* **Prefix caching** keyed by chained hashes of full blocks (vLLM-style), with shared
  reference-counted blocks and an LRU of evictable cached blocks; blocks holding *generated*
  tokens are cached too, so multi-turn conversations that resend history hit the cache.
* **Chunked prefill** (decodes co-scheduled with prefill chunks) vs **prefill-first**
  scheduling (prefill steps stall decodes — visible as ITL spikes).
* **Preemption by recompute** when decode growth exhausts KV blocks.
* Step time = `overhead + prefill_tokens × prefill_cost + decode_seqs × decode_cost +
  attended_kv_tokens × kv_cost`.

```yaml
type: simulated
name: sim-a
engine:
  block_size: 16
  num_kv_blocks: 4096
  max_num_seqs: 32
  max_num_batched_tokens: 2048
  max_model_len: 8192
  chunked_prefill: true
  prefix_caching: true
  step_overhead_s: 0.012
  prefill_s_per_token: 0.00003
  decode_s_per_seq: 0.0003
  decode_s_per_kv_token: 1.0e-7
```

The engine core (`backends/simulated/engine.py`) is pure and synchronous, so unit and
property tests drive it step by step. The async shell keeps engine time causal (a request
enters the engine only once engine time reaches its arrival) and drift-free (sleeps target
absolute engine time), so coarse OS timers delay delivery slightly but never lower the
simulated throughput. It reports real *simulated* engine metrics (running/waiting, KV
usage, prefix-cache queries/hits, preemptions, per-request queue/prefill/decode time) —
values the model actually computed, not fabricated estimates.

Its "tokenizer" is whitespace splitting plus a fixed chat template, and its output words
come from the shared vocabulary, so resent assistant messages map back to identical token
ids.
