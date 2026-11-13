# SGLang

Verified against **SGLang v0.5.23** (released 2026-11-02) using the v0.5.23 source,
docs.sglang.io (docs.sglang.ai now redirects there) and Docker Hub on 2026-11-13.

## Image

`lmsysorg/sglang:v0.5.23@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f`
(same digest as `v0.5.23-cu130`; amd64 + arm64; CUDA 13 — the CUDA 12 lane ended with
`v0.5.19-cu129`). The image has no ENTRYPOINT; InferenceLab appends `sglang serve …`.

## Launch (generated)

```bash
docker run -d --gpus all --ipc=host -p 127.0.0.1:30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface --env-file ~/.inferencelab/env \
  lmsysorg/sglang:v0.5.23@sha256:f832b5c3… \
  sglang serve --model-path Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 --host 0.0.0.0 --port 30000 \
  --served-model-name qwen2.5-7b-instruct --enable-metrics --enable-cache-report \
  --sampling-defaults openai --dtype bfloat16 --context-length 8192 \
  --max-running-requests 256 --mem-fraction-static 0.85 --chunked-prefill-size 8192
```

| Deployment field | SGLang flag |
|---|---|
| `max_context` | `--context-length` |
| `max_concurrency` | `--max-running-requests` |
| `max_batched_tokens` | `--chunked-prefill-size` |
| `chunked_prefill: false` | `--chunked-prefill-size -1` |
| `gpu_memory_utilization` | `--mem-fraction-static` (static weights + KV share — not the same quantity as vLLM's) |
| `parallelism` | `--tp-size`, `--pp-size`, `--dp-size`, `--ep-size` (full names only; `--tp` works only by argparse prefix matching) |
| `prefix_caching: false` | `--disable-radix-cache` |
| `lora` | `--enable-lora --max-loras-per-batch --max-lora-rank --lora-paths name=path` |
| `speculative` | `--speculative-algorithm` (`EAGLE`, `EAGLE3`, `NEXTN`, `STANDALONE`, `NGRAM`) + `--speculative-num-steps N --speculative-eagle-topk 1 --speculative-num-draft-tokens N+1` (+ draft path); topk 1 is required with the default overlap scheduler |
| `role: prefill/decode` | `--disaggregation-mode prefill|decode` (+ transfer backend; combine with `sglang_router --pd-disaggregation`) |
| `quantization` | none for pre-quantized checkpoints (AWQ auto-upgraded to `awq_marlin` on SM80+; forcing `awq` selects the slower kernel); `fp8` for online FP8 |

`--sampling-defaults openai` prevents the checkpoint's generation config from changing
sampling; `--enable-cache-report` adds `cached_tokens` to usage; `--enable-metrics` is
required for `/metrics`. The default schedule policy is now **`fcfs`** (not `lpm`); prefix-
aware `lpm` must be requested explicitly and falls back to FCFS above 128 waiting requests.

## HTTP behaviour

* `/health` runs a 1-token generation (503 while starting, or when the scheduler does not
  answer within 20 s); `/ready` is a cheap readiness check. `/health*`, `/ready*` and
  `/metrics*` bypass `--api-key`.
* `/server_info` (current) and `/get_server_info` (deprecated alias) report the version.
* `stream_options.include_usage` is supported (final `choices: []` chunk with usage).
* Client disconnect aborts the request (streaming: ~2 s; non-streaming: polled every 4 s).
* `ignore_eos`, `min_tokens`, `top_k`, `min_p`, `regex`, `ebnf` extensions; structured
  output via `response_format` (`json_schema`, `json_object`, `structural_tag`).

## Metrics mapping (`backends/sglang/backend.py`)

| BackendMetrics | SGLang sample |
|---|---|
| running / waiting | `sglang:num_running_reqs`, `sglang:num_queue_reqs` |
| KV usage | `sglang:token_usage` |
| prefix hit rate | `sglang:cache_hit_rate` (gauge); `sglang:cached_tokens_total` / `sglang:prompt_tokens_total` |
| generation tokens | `sglang:generation_tokens_total` |
| speculative acceptance | `sglang:spec_accept_rate` (+ `sglang:spec_accept_length` kept raw) |
| phase means | `sglang:queue_time_seconds`, `sglang:time_to_first_token_seconds`, `sglang:inter_token_latency_seconds`, `sglang:e2e_request_latency_seconds` |

## SGLang Model Gateway (context)

SGLang ships a router whose default `cache_aware` policy keeps an approximate radix tree
of request *text* per worker and routes to the longest prefix match unless the fleet is
imbalanced. InferenceLab's `PrefixAffinityRouter` reaches a similar goal without inspecting
prompts (rendezvous hashing on ids with bounded loads) and works for every engine.

## Model support

Qwen2.5-7B-Instruct and the official AWQ checkpoint are natively supported
(`Qwen2ForCausalLM`). 28 query / 4 KV heads: `--tp-size` must be 1, 2 or 4.

Sources: https://docs.sglang.io/docs/advanced_features/server_arguments ·
https://docs.sglang.io/docs/references/production_metrics ·
https://github.com/sgl-project/sglang/releases/tag/v0.5.23 ·
https://hub.docker.com/r/lmsysorg/sglang/tags
