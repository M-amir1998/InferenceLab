"""vLLM launch builder (verified against vLLM v0.33.0, docs.vllm.ai stable, 2026-11-13).

* The ``vllm/vllm-openai`` image's ENTRYPOINT is ``vllm serve``: the model goes first,
  positionally, followed by flags.
* Booleans use ``--enable-X`` / ``--no-enable-X`` (the ``--no-`` form is accepted by
  v0.32 and v0.33, unlike the newer ``--disable-X`` spelling).
* ``--generation-config vllm`` stops the checkpoint's ``generation_config.json`` from
  silently changing sampling defaults between engines.
* ``--enable-prompt-tokens-details`` makes usage report ``cached_tokens`` (prefix-cache
  hits per request), which the prefix-caching experiments rely on.
* Never pass ``--disable-log-stats``: it removes every ``vllm:*`` engine metric.
"""

from __future__ import annotations

import json

from inferencelab.backends.launch import (
    CONTAINER_BIND_ALL,
    LaunchPlan,
    UnsupportedDeploymentError,
    engine_arg_flags,
    quantization_flag_needed,
    require_image,
)
from inferencelab.domain.deployment import DeploymentRole, DeploymentSpec, SpeculativeMethod
from inferencelab.domain.model import ModelSpec

__all__ = ["build_vllm_plan"]

_SPEC_METHOD = {
    SpeculativeMethod.NGRAM: "ngram",
    SpeculativeMethod.DRAFT_MODEL: "draft_model",
    SpeculativeMethod.EAGLE: "eagle",
    SpeculativeMethod.EAGLE3: "eagle3",
    SpeculativeMethod.MTP: "mtp",
}


def build_vllm_plan(spec: DeploymentSpec, model: ModelSpec) -> LaunchPlan:
    image = require_image(spec)
    if spec.role is not DeploymentRole.UNIFIED and "kv-transfer-config" not in spec.engine_args:
        raise UnsupportedDeploymentError(
            "disaggregated vLLM needs a KV connector: set engine_args['kv-transfer-config']"
        )
    cmd: list[str] = [
        model.checkpoint,
        "--host", CONTAINER_BIND_ALL,
        "--port", str(spec.port),
        "--served-model-name", spec.served_name,
        "--revision", model.revision,
        "--generation-config", "vllm",
        "--enable-prompt-tokens-details",
    ]  # fmt: skip
    notes: list[str] = []
    if spec.dtype != "auto":
        cmd += ["--dtype", spec.dtype]
    if spec.max_context is not None:
        cmd += ["--max-model-len", str(spec.max_context)]
    if spec.max_concurrency is not None:
        cmd += ["--max-num-seqs", str(spec.max_concurrency)]
    else:
        notes.append("max-num-seqs defaults depend on GPU memory; pin it for reproducible runs")
    if spec.max_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(spec.max_batched_tokens)]
    if spec.gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(spec.gpu_memory_utilization)]
    p = spec.parallelism
    if p.tensor > 1:
        cmd += ["--tensor-parallel-size", str(p.tensor)]
    if p.pipeline > 1:
        cmd += ["--pipeline-parallel-size", str(p.pipeline)]
    if p.data > 1:
        cmd += ["--data-parallel-size", str(p.data)]
    if p.expert:
        cmd.append("--enable-expert-parallel")
    quant = quantization_flag_needed(spec, model)
    if quant is not None:
        cmd += ["--quantization", quant]
    if spec.prefix_caching is True:
        cmd.append("--enable-prefix-caching")
    elif spec.prefix_caching is False:
        cmd.append("--no-enable-prefix-caching")
    if spec.chunked_prefill is True:
        cmd.append("--enable-chunked-prefill")
    elif spec.chunked_prefill is False:
        cmd.append("--no-enable-chunked-prefill")
        notes.append("vLLM warns that disabling chunked prefill is not officially supported")
    if spec.lora is not None:
        cmd += [
            "--enable-lora",
            "--max-loras", str(spec.lora.max_loras),
            "--max-lora-rank", str(spec.lora.max_lora_rank),
            "--lora-modules", *[f"{a.name}={a.source}" for a in spec.lora.adapters],
        ]  # fmt: skip
    if spec.speculative is not None:
        sc: dict[str, object] = {
            "method": _SPEC_METHOD[spec.speculative.method],
            "num_speculative_tokens": spec.speculative.num_speculative_tokens,
        }
        if spec.speculative.draft_model:
            sc["model"] = spec.speculative.draft_model
        cmd += ["--speculative-config", json.dumps(sc, separators=(",", ":"))]
    cmd += engine_arg_flags(spec.engine_args)
    return LaunchPlan(
        backend=spec.backend,
        deployment=spec.name,
        image=image,
        port=spec.port,
        command=tuple(cmd),
        health_path="/health",
        notes=tuple(notes),
    )
