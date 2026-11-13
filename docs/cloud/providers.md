# Using a GPU provider

InferenceLab is provider-neutral: it needs **SSH access to a Linux machine with an NVIDIA
GPU, the NVIDIA driver, Docker and the NVIDIA container toolkit**. It never calls a provider
API and never creates or deletes machines (ADR-008).

## What to rent for the first experiment

A single GPU with ≥ 40 GB is comfortable for Qwen2.5-7B-Instruct in BF16 (≈15 GB of weights
plus KV cache for 8k context at high concurrency); 24 GB cards (L4, A10G, RTX 4090) work
with lower concurrency/context limits. Prefer the same GPU type for every engine you
compare, and note the exact type, driver and region.

## Generic steps (any provider)

1. Choose an image with a recent NVIDIA driver (R580+ for the CUDA 13 images pinned here),
   Docker and the NVIDIA container toolkit — most providers offer a "PyTorch/CUDA" or
   "Docker + NVIDIA" image.
2. Attach ≥ 150 GB of disk (engine images are 10–16 GB each; model weights ≈15 GB per
   checkpoint).
3. Add your SSH public key in the provider console; open **only** port 22. Engines bind to
   `127.0.0.1` and are never exposed.
4. Record the hourly price shown in the console together with the date — you will need it
   for the cost guard.
5. Fill in `configs/hosts/<name>.local.yaml` and follow docs/cloud/gpu-runbook.md.
6. **Terminate the machine in the console** when finished. InferenceLab will not.

## Notes by provider type

| Type | Examples | Notes |
|---|---|---|
| Hyperscaler VM | AWS (G5/G6/P4/P5), GCP (A2/A3/G2), Azure (NC/ND) | stable networking, per-second/minute billing varies; GPU quotas may need a request |
| GPU cloud | Lambda, CoreWeave, Crusoe, Nebius | GPU-first images; check whether Docker is preinstalled |
| Marketplace | RunPod, Vast.ai | cheapest, but some "pods" are containers without Docker-in-Docker — you need a VM-style machine where `docker run --gpus all` works |

Provider names are examples only; check current offerings, prices and terms yourself.
InferenceLab does not record or recommend prices.
