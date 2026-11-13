# Cloud execution (remote GPU)

InferenceLab never provisions cloud resources. It drives a GPU machine **you** rented, over
SSH, with no provider SDK and no stored credentials (ADR-008).

```text
Laptop (inferencelab remote run --execute)
   │ ssh (BatchMode, your agent / ~/.ssh/config)
   ▼
GPU host
   1. audit: driver, CUDA, GPUs/VRAM, Docker, NVIDIA runtime, uv, disk
   2. bootstrap: venv + pinned InferenceLab wheel (uploaded from the laptop)
   3. per deployment:
        prepare (e.g. download the pinned HF snapshot for TensorRT-LLM)
        docker pull <pinned image>; record RepoDigest
        docker run … -p 127.0.0.1:PORT:PORT (never public)
        wait for /health (container exit ⇒ logs + abort)
        discover served model id (/v1/models); smoke check (1 request + 1 stream)
        for each run: upload benchmark config ─► inferencelab benchmark run --execute
            (load generated ON the host; GPU memory sampled with nvidia-smi)
        copy results back (summary, profile, parquet, report)
        docker stop + rm; delete temp configs        ← machine left running
```

## Why the load generator runs on the GPU host

TTFT is often tens of milliseconds. A laptop→cloud round trip adds similar or larger,
variable latency and would dominate the measurement. Running the client on the host
measures the serving stack, not the internet.

## Host configuration

```yaml
# configs/hosts/<name>.local.yaml   (*.local.yaml is git-ignored)
name: my-gpu
hostname: 203.0.113.10        # example address
user: ubuntu
identity_file: ~/.ssh/id_ed25519   # a path; key material never lives in configs
workdir: ~/inferencelab
hf_cache: ~/.cache/huggingface
env_file: ~/.inferencelab/env      # created on the host by scripts/remote/bootstrap.sh
```

Hostname and user are validated against shell-injection characters; dangerous SSH options
(`ProxyCommand`, `LocalCommand`) are rejected; every remote command is shell-quoted.

## Safety rails

* `inferencelab remote plan` prints the exact command sequence without executing anything.
* `inferencelab remote run` refuses to start without `--execute`, after showing run count,
  duration upper bound, hardware and the cost upper bound (if a price is configured).
* Each run has a hard duration cap (`max_run_duration_s`).
* Cleanup removes containers and temporary configs but **never** shuts down, resizes or
  terminates the machine; the Hugging Face cache is kept to avoid re-downloading.
* Version pinning: images are pinned by tag (+ digest where verified); run metadata records
  the image digest, driver, GPU, engine version (`/version`, `/server_info`) and model revision.

See [docs/cloud/gpu-runbook.md](../cloud/gpu-runbook.md) for the step-by-step first experiment.
