# GPU runbook: the first real experiment

Nothing here runs automatically. You rent the machine, you run each step, and every paid
step requires `--execute`. Follow the order: **never launch the full matrix first.**

## 0. Prerequisites

* A rented Linux GPU machine (e.g. 1× H100 80GB, A100 80GB or L40S) with the NVIDIA driver
  (R580+ for the CUDA 13 images pinned here), Docker and the NVIDIA container toolkit.
* SSH access from the laptop (key in your agent or `~/.ssh/config`).
* A host file, git-ignored:

```bash
cp configs/hosts/example.yaml configs/hosts/my-gpu.local.yaml   # edit hostname/user
```

## 1. Machine audit

```bash
scp scripts/remote/bootstrap.sh my-gpu:~ && ssh my-gpu 'bash bootstrap.sh'
inferencelab remote doctor configs/hosts/my-gpu.local.yaml \
    --container-test nvidia/cuda:13.0.0-base-ubuntu24.04
```

Put `HF_TOKEN=…` in `~/.inferencelab/env` **on the host** only if you use gated models
(Qwen2.5-7B-Instruct is not gated).

## 2. Price and plan

Record the price you are paying (InferenceLab ships none), then review the plan:

```yaml
# in configs/experiments/gpu_first_comparison.yaml (or a copy of it)
pricing:
  gpu: H100-80GB-SXM
  count: 1
  hourly_cost_usd: <your price>
  price_unit: per_instance_hour
  price_source: <provider console URL>
  price_recorded_at: <YYYY-MM-DD>
```

```bash
uv build                                    # the wheel installed on the host
inferencelab experiment plan configs/experiments/gpu_first_comparison.yaml
inferencelab remote plan configs/experiments/gpu_first_comparison.yaml \
    --host configs/hosts/my-gpu.local.yaml   # dry run: prints every command
```

## 3. Backend container smoke, model load, one request, one stream

Start with **one deployment and one tiny run**. Copy the experiment and narrow it:

```yaml
# configs/experiments/gpu_smoke.local.yaml
name: gpu-smoke
mode: remote
max_runs: 1
matrix:
  deployments: [../deployments/vllm_qwen_7b_bf16.yaml]
  workloads: [../workloads/short_short.yaml]
  concurrency: [1]
slo: ../slos/interactive.yaml
models_dir: ../models
max_run_duration_s: 300
```

```bash
inferencelab remote run configs/experiments/gpu_smoke.local.yaml \
    --host configs/hosts/my-gpu.local.yaml --wheel dist/inferencelab-0.1.0.dev0-py3-none-any.whl --execute
```

This pulls the pinned image (digest recorded), starts the container on `127.0.0.1`, waits
for `/health`, discovers the served model id, runs `inferencelab backend check` (one request
+ one streamed request), runs the benchmark on the host and copies results to
`results/remote/my-gpu/…`. Repeat with `sglang_qwen_7b_bf16.yaml`.

## 4–7. Low concurrency → saturation → prefix cache → quantization → SLO/goodput

| Step | Experiment | What to look at |
|---|---|---|
| 4. low-concurrency baseline | `gpu_first_comparison` with `concurrency: [1]` | single-stream TTFT/TPOT |
| 5. saturation | `gpu_first_comparison` (1, 8, 32, 128) | throughput knee, TTFT growth, preemptions, GPU memory peak |
| 6. prefix cache | `gpu_prefix_cache` | TTFT and throughput, engine hit ratio, `cached_tokens` |
| 7. quantization | `gpu_quantization` + `inferencelab quality compare` | speed **and** memory **and** quality regression |
| 8. SLO / goodput | sweep request rate with `mixed_poisson`, `interactive` SLO | max rate at SLO; feed the profile to `capacity plan` |
| 9. long context | `gpu_chunked_prefill` | TTFT vs ITL trade-off of chunking |
| 10. TensorRT-LLM | `gpu_trtllm` | only after the vLLM/SGLang comparison is solid |

Each step uses the same host, model revision, workloads and seeds.

## Memory safety

Every remote run records GPU memory (peak per GPU, sampled by nvidia-smi), the deployment's
`gpu_memory_utilization`/`mem-fraction`, context and concurrency limits. Settings are chosen
to fit (`inferencelab backend launch-plan` prints a weight-fit check); never configure a run
to OOM on purpose. If a container exits during startup, the campaign prints its last logs
and stops.

## Cleanup and spend

Campaigns stop and remove their containers and temporary configs but leave the machine
running (the Hugging Face cache stays to avoid re-downloads). **Terminate the machine
yourself** in the provider console when you are done. `scripts/gpu/stop_engines.sh`
removes any leftover `inferencelab-*` containers.

## After the run

```bash
inferencelab benchmark compare <vllm run> <sglang run> --results results/remote/my-gpu/…
inferencelab report <run>
```

Commit only curated reports (`reports/`), never `requests.parquet` dumps, and keep GPU and
SIMULATED results apart.
