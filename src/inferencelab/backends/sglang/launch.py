"""SGLang launch builder (verified against SGLang v0.5.23 source and docs.sglang.io).

* ``lmsysorg/sglang`` images have no ENTRYPOINT: the command is ``sglang serve ...``.
* Full flag names only (``--tp-size``, not argparse prefix matches like ``--tp``).
* Pre-quantized checkpoints are auto-detected; AWQ is upgraded to ``awq_marlin`` on
  SM80+ unless ``--quantization awq`` is forced, so no flag is passed for them.
* ``--chunked-prefill-size -1`` disables chunked prefill.
* ``--sampling-defaults openai`` stops the checkpoint's generation_config from changing
  sampling defaults; ``--enable-cache-report`` adds ``cached_tokens`` to usage.
* The default schedule policy is now ``fcfs``; prefix-aware ``lpm`` must be requested.
"""

from __future__ import annotations

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

__all__ = ["build_sglang_plan"]

_SPEC_ALGO = {
    SpeculativeMethod.EAGLE: "EAGLE",
    SpeculativeMethod.EAGLE3: "EAGLE3",
    SpeculativeMethod.MTP: "NEXTN",
    SpeculativeMethod.DRAFT_MODEL: "STANDALONE",
    SpeculativeMethod.NGRAM: "NGRAM",
}


def build_sglang_plan(spec: DeploymentSpec, model: ModelSpec) -> LaunchPlan:
    image = require_image(spec)
    cmd: list[str] = [
        "sglang", "serve",
        "--model-path", model.checkpoint,
        "--revision", model.revision,
        "--host", CONTAINER_BIND_ALL,
        "--port", str(spec.port),
        "--served-model-name", spec.served_name,
        "--enable-metrics",
        "--enable-cache-report",
        "--sampling-defaults", "openai",
    ]  # fmt: skip
    notes: list[str] = []
    if spec.dtype != "auto":
        cmd += ["--dtype", spec.dtype]
    if spec.max_context is not None:
        cmd += ["--context-length", str(spec.max_context)]
    if spec.max_concurrency is not None:
        cmd += ["--max-running-requests", str(spec.max_concurrency)]
    if spec.gpu_memory_utilization is not None:
        cmd += ["--mem-fraction-static", str(spec.gpu_memory_utilization)]
        notes.append("SGLang's mem-fraction-static is the static (weights+KV) share, not total use")
    if spec.chunked_prefill is False:
        cmd += ["--chunked-prefill-size", "-1"]
        if spec.max_batched_tokens is not None:
            notes.append("max_batched_tokens is ignored when chunked prefill is disabled")
    elif spec.max_batched_tokens is not None:
        cmd += ["--chunked-prefill-size", str(spec.max_batched_tokens)]
    p = spec.parallelism
    if p.tensor > 1:
        cmd += ["--tp-size", str(p.tensor)]
    if p.pipeline > 1:
        cmd += ["--pp-size", str(p.pipeline)]
    if p.data > 1:
        cmd += ["--dp-size", str(p.data)]
    if p.expert:
        cmd += ["--ep-size", str(p.tensor)]
    quant = quantization_flag_needed(spec, model)
    if quant is not None:
        cmd += ["--quantization", quant]
    if spec.prefix_caching is False:
        cmd.append("--disable-radix-cache")
    if spec.lora is not None:
        cmd += [
            "--enable-lora",
            "--max-loras-per-batch", str(spec.lora.max_loras),
            "--max-lora-rank", str(spec.lora.max_lora_rank),
            "--lora-paths", *[f"{a.name}={a.source}" for a in spec.lora.adapters],
        ]  # fmt: skip
    if spec.speculative is not None:
        n = spec.speculative.num_speculative_tokens
        cmd += [
            "--speculative-algorithm", _SPEC_ALGO[spec.speculative.method],
            "--speculative-num-steps", str(n),
            "--speculative-eagle-topk", "1",  # required with the default overlap scheduler
            "--speculative-num-draft-tokens", str(n + 1),
        ]  # fmt: skip
        if spec.speculative.draft_model:
            cmd += ["--speculative-draft-model-path", spec.speculative.draft_model]
    if spec.role is DeploymentRole.PREFILL:
        cmd += ["--disaggregation-mode", "prefill"]
    elif spec.role is DeploymentRole.DECODE:
        cmd += ["--disaggregation-mode", "decode"]
    if spec.role is not DeploymentRole.UNIFIED:
        notes.append(
            "PD disaggregation needs a router in front (sglang_router --pd-disaggregation)"
        )
    if spec.role is not DeploymentRole.UNIFIED and spec.speculative is not None:
        raise UnsupportedDeploymentError(
            "speculative decoding with PD disaggregation is not configured"
        )
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
