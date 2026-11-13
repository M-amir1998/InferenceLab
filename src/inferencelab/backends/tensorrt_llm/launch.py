"""TensorRT-LLM launch builder (verified against TensorRT-LLM v1.2.1 stable, 2026-11-13).

* Command: ``trtllm-serve serve MODEL`` with underscore flags (``--tp_size``...).
  ``--host`` defaults to ``localhost``, so ``0.0.0.0`` is required inside Docker.
* Engine features are configured through an extra-options YAML (``--config``):
  ``kv_cache_config.enable_block_reuse`` (prefix reuse, default on),
  ``enable_chunked_prefill`` (default off), ``speculative_config``, ``lora_config``.
  ``return_perf_metrics: true`` is required for Prometheus text at /prometheus/metrics.
* ``trtllm-serve`` has no revision flag, so the pinned snapshot is downloaded first
  and served from its local Hugging Face cache path.
* The PyTorch backend does **not** load Hugging Face AutoAWQ checkpoints; quantized
  serving needs NVIDIA ModelOpt (or llm-compressor FP8) checkpoints.
* Containers need ``--ulimit memlock=-1 --ulimit stack=67108864``.
"""

from __future__ import annotations

from typing import Any

import yaml

from inferencelab.backends.launch import (
    CONTAINER_BIND_ALL,
    CONTAINER_CONFIG_DIR,
    CONTAINER_HF_CACHE,
    LaunchPlan,
    UnsupportedDeploymentError,
    engine_arg_flags,
    require_image,
)
from inferencelab.domain.deployment import DeploymentRole, DeploymentSpec, SpeculativeMethod
from inferencelab.domain.model import ModelSpec

__all__ = ["build_trtllm_plan", "hf_snapshot_path"]

# v1.2.1 spellings ("Eagle" means EAGLE-3 there; v1.4 renames it to "Eagle3").
_DECODING_TYPE = {
    SpeculativeMethod.EAGLE3: "Eagle",
    SpeculativeMethod.MTP: "MTP",
    SpeculativeMethod.NGRAM: "NGram",
    SpeculativeMethod.DRAFT_MODEL: "DraftTarget",
}
_LOADABLE_QUANT_TOOLS = {"modelopt", "llm-compressor"}
EXTRA_OPTIONS_FILE = "trtllm-extra-options.yaml"


def hf_snapshot_path(repo_id: str, revision: str, root: str = CONTAINER_HF_CACHE) -> str:
    """Path of a downloaded snapshot in the standard Hugging Face cache layout."""
    return f"{root}/hub/models--{repo_id.replace('/', '--')}/snapshots/{revision}"


def build_trtllm_plan(spec: DeploymentSpec, model: ModelSpec) -> LaunchPlan:
    image = require_image(spec)
    if spec.role is not DeploymentRole.UNIFIED:
        raise UnsupportedDeploymentError(
            "TensorRT-LLM disaggregated serving uses `trtllm-serve disaggregated` with "
            "context/generation server configs; not automated in v0.1"
        )
    if spec.quantization.value not in ("bf16", "fp16"):
        if spec.quantization != model.quantization:
            raise UnsupportedDeploymentError(
                "trtllm-serve does not quantize online; export a ModelOpt checkpoint first"
            )
        if model.quantization_tool not in _LOADABLE_QUANT_TOOLS:
            raise UnsupportedDeploymentError(
                f"TensorRT-LLM cannot load {model.quantization_tool or 'unknown'}-produced "
                f"{model.quantization.value} checkpoints (HF AutoAWQ is treated as unquantized); "
                "use a ModelOpt or llm-compressor checkpoint"
            )
    if not model.revision_pinned:
        raise UnsupportedDeploymentError("TensorRT-LLM runs need a commit-SHA model revision")

    options: dict[str, Any] = {"return_perf_metrics": True, "enable_iter_perf_stats": True}
    if spec.prefix_caching is not None:
        options["kv_cache_config"] = {"enable_block_reuse": spec.prefix_caching}
    if spec.chunked_prefill is not None:
        options["enable_chunked_prefill"] = spec.chunked_prefill
    if spec.speculative is not None:
        decoding = _DECODING_TYPE.get(spec.speculative.method)
        if decoding is None:
            method = spec.speculative.method.value
            raise UnsupportedDeploymentError(
                f"speculative method {method!r} has no TensorRT-LLM v1.2 mapping"
            )
        sc: dict[str, Any] = {
            "decoding_type": decoding,
            "max_draft_len": spec.speculative.num_speculative_tokens,
        }
        if spec.speculative.draft_model:
            sc["speculative_model_dir"] = spec.speculative.draft_model
        options["speculative_config"] = sc
    if spec.lora is not None:
        options["lora_config"] = {
            "lora_dir": [a.source for a in spec.lora.adapters],
            "max_lora_rank": spec.lora.max_lora_rank,
            "max_loras": spec.lora.max_loras,
        }

    cmd: list[str] = [
        "trtllm-serve", "serve", hf_snapshot_path(model.checkpoint, model.revision),
        "--host", CONTAINER_BIND_ALL,
        "--port", str(spec.port),
        "--tp_size", str(spec.parallelism.tensor),
        "--pp_size", str(spec.parallelism.pipeline),
        "--config", f"{CONTAINER_CONFIG_DIR}/{EXTRA_OPTIONS_FILE}",
    ]  # fmt: skip
    if spec.parallelism.expert:
        cmd += ["--ep_size", str(spec.parallelism.tensor)]
    if spec.max_concurrency is not None:
        cmd += ["--max_batch_size", str(spec.max_concurrency)]
    if spec.max_batched_tokens is not None:
        cmd += ["--max_num_tokens", str(spec.max_batched_tokens)]
    if spec.max_context is not None:
        cmd += ["--max_seq_len", str(spec.max_context)]
    if spec.gpu_memory_utilization is not None:
        cmd += ["--kv_cache_free_gpu_memory_fraction", str(spec.gpu_memory_utilization)]
    cmd += engine_arg_flags(spec.engine_args, dash="_")

    notes = [
        "served model id is the snapshot directory name; the campaign discovers it via /v1/models",
        "prompts longer than max_num_tokens are rejected unless chunked prefill is enabled",
    ]
    if spec.parallelism.data > 1:
        notes.append("data parallelism is realised as separate replicas behind the gateway")
    return LaunchPlan(
        backend=spec.backend,
        deployment=spec.name,
        image=image,
        port=spec.port,
        command=tuple(cmd),
        health_path="/health",
        extra_docker_args=("--ulimit", "memlock=-1", "--ulimit", "stack=67108864"),
        config_files={EXTRA_OPTIONS_FILE: yaml.safe_dump(options, sort_keys=True)},
        prepare=(
            (
                "uvx",
                "--from",
                "huggingface_hub",
                "hf",
                "download",
                model.checkpoint,
                "--revision",
                model.revision,
            ),
        ),
        notes=tuple(notes),
    )
