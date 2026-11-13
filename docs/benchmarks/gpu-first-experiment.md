# First GPU benchmark plan

**Status: prepared, not run.** No numbers exist yet.

## Question

On the same GPU, model revision, precision, context limits and workloads, how do
**vLLM 0.33.0** and **SGLang 0.5.23** compare on latency (TTFT, TPOT), throughput,
goodput under an interactive SLO, GPU memory and cost?

## Fixed factors

| Factor | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` (Apache-2.0) |
| Precision | BF16 |
| GPU | one GPU of a single type, same host for both engines (recorded: type, VRAM, driver, CUDA) |
| Engine settings | max context 8192, max concurrency 256, 8192 batched tokens/step, prefix caching **on**, chunked prefill **on**, memory fraction 0.90 (vLLM) / 0.85 static (SGLang) |
| Sampling | temperature 0, `ignore_eos`, fixed output lengths; checkpoint generation defaults disabled on both |
| Client | InferenceLab load generator on the GPU host, closed loop |
| Warmup | per workload (16 or 8 requests) |
| Images | `vllm/vllm-openai:v0.33.0@sha256:d825baf9…`, `lmsysorg/sglang:v0.5.23@sha256:f832b5c3…` |

## Matrix (32 runs, cap 40)

| Workloads | Concurrency |
|---|---|
| `short_short` (128–256 in / 64–128 out) | 1, 8, 32, 128 (≈ saturation) |
| `long_short` (3–4k in / 64–128 out) | 1, 8, 32, 128 |
| `short_long` (128–256 in / 768–1024 out) | 1, 8, 32, 128 |
| `shared_prefix` (4 × 2048-token prefixes) | 1, 8, 32, 128 |

SLO: `interactive` (TTFT p95 ≤ 500 ms, TPOT p95 ≤ 50 ms, E2E p95 ≤ 10 s, availability 0.99).

## Metrics reported

TTFT, TPOT, ITL and E2E distributions; request and token throughput; goodput and SLO
attainment; errors; engine-reported queue/prefill/decode means, KV usage, preemptions and
prefix-cache hit ratio; peak GPU memory; cost per 1M output tokens and per SLO-compliant
request (with the recorded price).

## Commands

```bash
uv build
inferencelab experiment plan configs/experiments/gpu_first_comparison.yaml
inferencelab remote plan configs/experiments/gpu_first_comparison.yaml --host configs/hosts/my-gpu.local.yaml
# after the smoke steps in the runbook succeed:
inferencelab remote run configs/experiments/gpu_first_comparison.yaml \
    --host configs/hosts/my-gpu.local.yaml \
    --wheel dist/inferencelab-0.1.0.dev0-py3-none-any.whl --execute
inferencelab benchmark compare <runs…> --results results/remote/my-gpu
```

Duration upper bound with the defaults: 32 runs × 900 s cap + 2 model loads ≈ 8.6 h; most
runs finish far sooner (closed-loop runs end when their requests complete). The plan
command prints the bound and, once a price is configured, the cost upper bound.

## TensorRT-LLM

Follows as a separate experiment (`gpu_trtllm.yaml`, vLLM vs TensorRT-LLM in BF16) once its
configuration has been validated with `backend check` and `quality compare`; TensorRT-LLM's
AWQ path needs a ModelOpt checkpoint and is out of the first comparison.
