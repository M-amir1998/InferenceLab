"""Engine launch plans: DeploymentSpec + ModelSpec -> pinned ``docker run`` command.

Launch builders translate engine-neutral deployment settings into each engine's CLI.
Flags were checked against the engines' official documentation and source for the
pinned versions recorded in ``ENGINE_VERSIONS`` (see docs/backends/). Settings an engine
cannot honour raise :class:`UnsupportedDeploymentError` instead of silently producing
a different experiment than the one configured.

The engine port is published on ``127.0.0.1`` only: the load generator runs on the GPU
host itself (no WAN latency in measurements, no public model endpoint).
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field

from inferencelab.domain.deployment import BackendKind, DeploymentSpec, EngineArgValue
from inferencelab.domain.model import ModelSpec

__all__ = [
    "CONTAINER_BIND_ALL",
    "CONTAINER_HF_CACHE",
    "ENGINE_VERSIONS",
    "LaunchPlan",
    "UnsupportedDeploymentError",
    "build_launch_plan",
    "engine_arg_flags",
]

#: Versions the launch builders were verified against (2026-11-13).
ENGINE_VERSIONS: Mapping[BackendKind, str] = {
    BackendKind.VLLM: "0.33.0",
    BackendKind.SGLANG: "0.5.23",
    BackendKind.TENSORRT_LLM: "1.2.1",
}

CONTAINER_HF_CACHE = "/root/.cache/huggingface"
#: Engines bind all interfaces *inside* the container; the host publishes the port on
#: 127.0.0.1 only (see LaunchPlan.docker_run), so nothing is exposed externally.
CONTAINER_BIND_ALL = "0.0.0.0"  # noqa: S104  # nosec B104
CONTAINER_CONFIG_DIR = "/inferencelab"


class UnsupportedDeploymentError(ValueError):
    """The requested deployment cannot be expressed for this engine."""


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    backend: BackendKind
    deployment: str
    image: str
    port: int
    command: tuple[str, ...]  # appended after the image
    health_path: str
    extra_docker_args: tuple[str, ...] = ()
    config_files: Mapping[str, str] = field(default_factory=dict)  # name -> content
    prepare: tuple[tuple[str, ...], ...] = ()  # host commands to run before start
    notes: tuple[str, ...] = ()

    @property
    def container_name(self) -> str:
        return f"inferencelab-{self.deployment}"

    def docker_run(
        self, *, hf_cache: str, env_file: str | None, config_dir: str | None
    ) -> list[str]:
        argv = [
            "docker", "run", "-d", "--name", self.container_name,
            "--gpus", "all", "--ipc=host",
            "-p", f"127.0.0.1:{self.port}:{self.port}",
            "-v", f"{hf_cache}:{CONTAINER_HF_CACHE}",
        ]  # fmt: skip
        if env_file:
            argv += ["--env-file", env_file]
        if self.config_files:
            if config_dir is None:
                raise ValueError("this launch plan needs a config_dir for its config files")
            argv += ["-v", f"{config_dir}:{CONTAINER_CONFIG_DIR}:ro"]
        argv += list(self.extra_docker_args)
        argv.append(self.image)
        argv += list(self.command)
        return argv

    def describe(self, *, hf_cache: str = "~/.cache/huggingface") -> str:
        return shlex.join(
            self.docker_run(hf_cache=hf_cache, env_file="~/.inferencelab/env", config_dir="./cfg")
        )


def engine_arg_flags(args: Mapping[str, EngineArgValue], *, dash: str = "-") -> list[str]:
    """Render verbatim engine args. ``True`` -> ``--flag``; ``False`` -> ``--no-flag``."""
    out: list[str] = []
    for key, value in args.items():
        flag = "--" + key.lstrip("-").replace("_", dash) if dash == "-" else "--" + key.lstrip("-")
        if value is True:
            out.append(flag)
        elif value is False:
            out.append("--no-" + flag[2:])
        else:
            out += [flag, str(value)]
    return out


def require_image(spec: DeploymentSpec) -> str:
    if spec.image is None:
        raise UnsupportedDeploymentError(f"deployment {spec.name!r} needs a pinned image")
    return spec.image


def build_launch_plan(spec: DeploymentSpec, model: ModelSpec) -> LaunchPlan:
    if spec.model != model.name:
        raise ValueError(
            f"deployment {spec.name!r} references model {spec.model!r}, got {model.name!r}"
        )
    if spec.backend is BackendKind.VLLM:
        from inferencelab.backends.vllm.launch import build_vllm_plan

        return build_vllm_plan(spec, model)
    if spec.backend is BackendKind.SGLANG:
        from inferencelab.backends.sglang.launch import build_sglang_plan

        return build_sglang_plan(spec, model)
    if spec.backend is BackendKind.TENSORRT_LLM:
        from inferencelab.backends.tensorrt_llm.launch import build_trtllm_plan

        return build_trtllm_plan(spec, model)
    raise UnsupportedDeploymentError(
        f"{spec.backend.value} deployments are not launched as containers"
    )


def quantization_flag_needed(spec: DeploymentSpec, model: ModelSpec) -> str | None:
    """Decide whether an explicit quantization flag is needed.

    Pre-quantized checkpoints (AWQ/GPTQ/FP8) are auto-detected by vLLM and SGLang from
    ``quantization_config``; passing a flag can force slower kernels. Only *online*
    quantization of a BF16/FP16 checkpoint (currently FP8) needs a flag.
    """
    dense = {"bf16", "fp16"}
    if spec.quantization.value in dense:
        if model.quantization.value not in dense:
            raise UnsupportedDeploymentError(
                f"cannot serve {model.quantization.value} checkpoint {model.checkpoint} as "
                f"{spec.quantization.value}; register the dense checkpoint instead"
            )
        return None
    if spec.quantization == model.quantization:
        return None  # pre-quantized checkpoint: auto-detected
    if spec.quantization.value == "fp8" and model.quantization.value in dense:
        return "fp8"  # online FP8 quantization at load time
    raise UnsupportedDeploymentError(
        f"{spec.quantization.value} serving needs a pre-quantized {spec.quantization.value} "
        f"checkpoint, but model {model.name!r} is {model.quantization.value}"
    )
