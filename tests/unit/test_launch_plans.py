from __future__ import annotations

import json

import pytest
import yaml

from inferencelab.backends.launch import (
    UnsupportedDeploymentError,
    build_launch_plan,
    engine_arg_flags,
)
from inferencelab.backends.tensorrt_llm.launch import hf_snapshot_path
from inferencelab.domain.deployment import DeploymentSpec
from inferencelab.domain.model import ModelSpec

SHA = "a" * 40
BF16 = ModelSpec(
    name="m", checkpoint="Qwen/Qwen2.5-7B-Instruct", revision=SHA, parameter_count_b=7.6
)
AWQ = ModelSpec.model_validate(
    {
        "name": "m",
        "checkpoint": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "revision": SHA,
        "quantization": "awq",
        "quantization_tool": "autoawq",
    }
)


def dep(backend: str, image: str | None = "img/x:1.0", **kw: object) -> DeploymentSpec:
    base: dict[str, object] = {
        "name": "d",
        "model": "m",
        "backend": backend,
        "image": image,
        "hardware": {"gpu": "H100-80GB-SXM", "count": 2},
        "served_model_name": "qwen",
    }
    base.update(kw)
    return DeploymentSpec.model_validate(base)


def test_vllm_plan_flags() -> None:
    plan = build_launch_plan(
        dep(
            "vllm",
            max_context=8192,
            max_concurrency=256,
            max_batched_tokens=8192,
            gpu_memory_utilization=0.9,
            prefix_caching=False,
            chunked_prefill=True,
            parallelism={"tensor": 2},
            speculative={"method": "ngram", "num_speculative_tokens": 3},
            engine_args={"enforce-eager": True, "seed": 0},
        ),
        BF16,
    )
    cmd = plan.command
    assert cmd[0] == "Qwen/Qwen2.5-7B-Instruct"  # image entrypoint is `vllm serve`
    for flag in (
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--generation-config",
        "--enforce-eager",
    ):
        assert flag in cmd
    assert cmd[cmd.index("--revision") + 1] == SHA
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert spec == {"method": "ngram", "num_speculative_tokens": 3}
    assert "--quantization" not in cmd
    argv = plan.docker_run(hf_cache="/h", env_file="/e", config_dir=None)
    assert "127.0.0.1:8000:8000" in argv and "--env-file" in argv and argv[-len(cmd) :] == list(cmd)


def test_awq_checkpoint_is_auto_detected_but_online_awq_is_rejected() -> None:
    assert "--quantization" not in build_launch_plan(dep("vllm", quantization="awq"), AWQ).command
    with pytest.raises(UnsupportedDeploymentError):
        build_launch_plan(dep("vllm", quantization="awq"), BF16)
    fp8 = build_launch_plan(dep("sglang", quantization="fp8"), BF16).command
    assert fp8[fp8.index("--quantization") + 1] == "fp8"


def test_sglang_plan_flags() -> None:
    plan = build_launch_plan(
        dep(
            "sglang",
            prefix_caching=False,
            chunked_prefill=False,
            max_concurrency=64,
            lora={"adapters": [{"name": "a", "source": "org/a"}], "max_loras": 2},
            speculative={
                "method": "eagle3",
                "draft_model": "org/draft",
                "num_speculative_tokens": 4,
            },
        ),
        BF16,
    )
    cmd = plan.command
    assert cmd[:2] == ("sglang", "serve")
    assert "--disable-radix-cache" in cmd and "--enable-metrics" in cmd
    assert cmd[cmd.index("--chunked-prefill-size") + 1] == "-1"
    assert cmd[cmd.index("--speculative-algorithm") + 1] == "EAGLE3"
    assert cmd[cmd.index("--speculative-num-draft-tokens") + 1] == "5"
    assert "a=org/a" in cmd


def test_trtllm_plan_and_config_file() -> None:
    plan = build_launch_plan(
        dep("tensorrt_llm", prefix_caching=True, chunked_prefill=True, max_concurrency=128), BF16
    )
    assert plan.command[:3] == ("trtllm-serve", "serve", hf_snapshot_path(BF16.checkpoint, SHA))
    options = yaml.safe_load(plan.config_files["trtllm-extra-options.yaml"])
    assert options["kv_cache_config"] == {"enable_block_reuse": True}
    assert options["enable_chunked_prefill"] is True and options["return_perf_metrics"] is True
    assert "--ulimit" in plan.extra_docker_args
    assert plan.prepare and plan.prepare[0][-1] == SHA
    with pytest.raises(ValueError, match="config_dir"):
        plan.docker_run(hf_cache="/h", env_file=None, config_dir=None)


def test_trtllm_rejects_autoawq_unpinned_and_disaggregated() -> None:
    with pytest.raises(UnsupportedDeploymentError, match="AutoAWQ"):
        build_launch_plan(dep("tensorrt_llm", quantization="awq"), AWQ)
    with pytest.raises(UnsupportedDeploymentError, match="SHA"):
        build_launch_plan(dep("tensorrt_llm"), BF16.model_copy(update={"revision": "main"}))
    with pytest.raises(UnsupportedDeploymentError):
        build_launch_plan(dep("tensorrt_llm", role="prefill"), BF16)


def test_vllm_disaggregation_requires_kv_connector() -> None:
    with pytest.raises(UnsupportedDeploymentError, match="kv-transfer-config"):
        build_launch_plan(dep("vllm", role="decode"), BF16)


def test_local_backends_have_no_launch_plan() -> None:
    with pytest.raises(UnsupportedDeploymentError):
        build_launch_plan(dep("mock", image=None, hardware={"gpu": "cpu", "count": 0}), BF16)


def test_engine_arg_flags_rendering() -> None:
    assert engine_arg_flags({"a_b": True, "c": False, "d": 3}) == ["--a-b", "--no-c", "--d", "3"]
    assert engine_arg_flags({"tp_size": 2}, dash="_") == ["--tp_size", "2"]
