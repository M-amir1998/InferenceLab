# TensorRT-LLM

Verified against **TensorRT-LLM v1.2.1** (latest stable, 2026-04-20) using the v1.2.1 and
v1.4.0rc3 sources and the official docs on 2026-11-13. The docs site's "latest" is
1.4.0rc3, a **pre-release**; its CLI differs (noted below). InferenceLab pins 1.2.1.

## Image

`nvcr.io/nvidia/tensorrt-llm/release:1.2.1` (NGC; amd64 + arm64; there is **no** `latest`
tag and the Docker Hub `nvidia/tensorrt-llm` repository is an empty placeholder). The
container needs `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864`.

## Launch (generated)

```bash
# trtllm-serve has no revision flag: download the pinned snapshot first
uvx --from huggingface_hub hf download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28

docker run -d --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface --env-file ~/.inferencelab/env \
  -v ./cfg:/inferencelab:ro --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.1 \
  trtllm-serve serve /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a3545… \
  --host 0.0.0.0 --port 8000 --tp_size 1 --pp_size 1 \
  --config /inferencelab/trtllm-extra-options.yaml \
  --max_batch_size 256 --max_num_tokens 8192 --max_seq_len 8192 \
  --kv_cache_free_gpu_memory_fraction 0.85
```

`trtllm-extra-options.yaml` (generated from the deployment):

```yaml
enable_chunked_prefill: true        # default false
enable_iter_perf_stats: true
kv_cache_config:
  enable_block_reuse: true          # KV-cache (prefix) reuse; default true
return_perf_metrics: true           # required for Prometheus at /prometheus/metrics
```

| Deployment field | TensorRT-LLM |
|---|---|
| `max_concurrency` | `--max_batch_size` |
| `max_batched_tokens` | `--max_num_tokens` (prompts longer than this are rejected unless chunked prefill is on) |
| `max_context` | `--max_seq_len` |
| `gpu_memory_utilization` | `--kv_cache_free_gpu_memory_fraction` |
| `parallelism` | `--tp_size`, `--pp_size`, `--ep_size` (underscores) |
| `prefix_caching` | `kv_cache_config.enable_block_reuse` |
| `chunked_prefill` | `enable_chunked_prefill` |
| `speculative` | `speculative_config: {decoding_type, max_draft_len, speculative_model_dir}`; 1.2.1 names: `Eagle` (= EAGLE-3), `MTP`, `NGram`, `DraftTarget` (1.4 renames `Eagle` → `Eagle3`) |
| `lora` | `lora_config: {lora_dir, max_lora_rank, max_loras}` |

`--host` defaults to `localhost`, so `0.0.0.0` is required inside the container. In 1.2.1
`--backend` defaults to `pytorch`; in 1.4 only `pytorch` exists, so InferenceLab omits it.
`--config` and `--extra_llm_api_options` are the same flag in both versions.

## Quantization caveat

The PyTorch backend supports FP8, NVFP4, W4A16/W4A8 AWQ and GPTQ **from NVIDIA ModelOpt
(or llm-compressor FP8) checkpoints**. Hugging Face **AutoAWQ checkpoints such as
`Qwen/Qwen2.5-7B-Instruct-AWQ` do not load** (they are treated as unquantized). There is no
NVIDIA-published FP8 Qwen2.5-7B-Instruct. InferenceLab's launch builder rejects such
deployments; an AWQ comparison including TensorRT-LLM needs a ModelOpt export
(`examples/llm_ptq/scripts/huggingface_example.sh --quant w4a16_awq`), registered as a
separate model with `quantization_tool: modelopt`.

## HTTP behaviour

* `/health` (200 when ready; 503 on fatal engine error), `/health_generate`, `/v1/models`
  (id = snapshot directory name unless `--served_model_name`, which only 1.4 has — the
  campaign discovers the id), `/version`.
* `/metrics` returns **JSON iteration stats** consumed once per poll; Prometheus text lives at
  `/prometheus/metrics` (needs `return_perf_metrics: true`). The adapter scrapes the latter.
* `stream_options.include_usage` supported; final chunk includes `cached_tokens`.
* Client disconnect cancels the request (checked about once a second).
* Structured output needs `guided_decoding_backend: xgrammar` in the options YAML; without
  it `response_format` is likely ignored (UNVERIFIED — test before relying on it).
* `ignore_eos` support was not verified for `trtllm-serve` (UNVERIFIED); if a run reports
  output lengths below `max_tokens`, set `ignore_eos: false` in the workload and compare
  token throughput rather than request latency at fixed length.

## Metrics mapping

Only verified names are mapped: `trtllm_kv_cache_hit_rate` → prefix hit rate,
`trtllm_time_to_first_token_seconds` → engine TTFT mean. Every `trtllm_*` sample is kept in
`BackendMetrics.raw` for provenance; unmapped fields stay `None`.

## Model support

`Qwen2ForCausalLM` is registered in the PyTorch backend (Qwen2.5-7B-Instruct supported,
max context 32,768). It is not part of the per-feature support matrix and the 1.4.0rc3
accuracy tests contain no Qwen2.5 results — run `inferencelab quality compare` against the
vLLM BF16 deployment before trusting comparisons.

## Disaggregated serving

Context and generation workers are separate `trtllm-serve` processes (each with a
`cache_transceiver_config` backend), fronted by `trtllm-serve disaggregated -c config.yaml`.
Not automated by InferenceLab v0.1 (the launch builder rejects `role: prefill|decode`).

Sources: https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html ·
https://nvidia.github.io/TensorRT-LLM/features/quantization.html ·
https://nvidia.github.io/TensorRT-LLM/features/kvcache.html ·
https://github.com/NVIDIA/TensorRT-LLM/releases
