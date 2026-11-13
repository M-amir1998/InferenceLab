"""GPU host doctor: driver, CUDA, GPUs/VRAM, Docker and the NVIDIA container runtime.

Runs through any :class:`CommandExecutor`, so the same checks serve
``inferencelab gpu doctor`` (local) and ``inferencelab remote doctor HOST`` (SSH).
On the CPU-only development laptop, missing GPU tooling is reported as *skip*, not
failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from inferencelab.remote.executor import CommandExecutor

__all__ = ["Check", "GpuInfo", "gpu_doctor", "parse_cuda_version", "parse_gpu_csv"]

Status = Literal["ok", "warn", "fail", "skip"]


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True, slots=True)
class GpuInfo:
    name: str
    memory_mib: float
    driver: str


def parse_gpu_csv(text: str) -> list[GpuInfo]:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(GpuInfo(parts[0], float(parts[1]), parts[2]))
        except ValueError:
            continue
    return gpus


def parse_cuda_version(nvidia_smi_output: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([\d.]+)", nvidia_smi_output)
    return match.group(1) if match else None


async def gpu_doctor(
    executor: CommandExecutor,
    *,
    expect_gpu: bool = True,
    container_test_image: str | None = None,
    min_free_disk_gb: float = 100.0,
) -> list[Check]:
    checks: list[Check] = []
    missing: Status = "fail" if expect_gpu else "skip"
    probe = await executor.run(["sh", "-c", "command -v nvidia-smi"], check=False)
    if not probe.ok:
        checks.append(Check("nvidia-smi", missing, "not found (no NVIDIA driver on this host)"))
    else:
        query = await executor.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
        )
        gpus = parse_gpu_csv(query.stdout) if query.ok else []
        if gpus:
            desc = "; ".join(f"{g.name} ({g.memory_mib / 1024:.0f} GiB)" for g in gpus)
            checks.append(Check("gpus", "ok", f"{len(gpus)}× {desc}"))
            checks.append(Check("driver", "ok", gpus[0].driver))
        else:
            checks.append(Check("gpus", "fail", "nvidia-smi present but no GPU reported"))
        full = await executor.run(["nvidia-smi"], check=False)
        cuda = parse_cuda_version(full.stdout)
        checks.append(Check("cuda", "ok" if cuda else "warn", cuda or "CUDA version not reported"))
    docker = await executor.run(
        ["docker", "version", "--format", "{{.Server.Version}}"], check=False
    )
    if docker.ok and docker.stdout.strip():
        checks.append(Check("docker", "ok", f"server {docker.stdout.strip()}"))
        runtimes = await executor.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"], check=False
        )
        has_nvidia = "nvidia" in runtimes.stdout
        checks.append(
            Check(
                "nvidia-container-runtime",
                "ok" if has_nvidia else ("warn" if expect_gpu else "skip"),
                "nvidia runtime registered"
                if has_nvidia
                else "nvidia runtime not listed (CDI-only setups may still work)",
            )
        )
    else:
        checks.append(
            Check("docker", "fail" if expect_gpu else "warn", "docker daemon not reachable")
        )
    uv = await executor.run(["sh", "-c", "command -v uv && uv --version"], check=False)
    checks.append(
        Check(
            "uv",
            "ok" if uv.ok else "warn",
            uv.stdout.strip().splitlines()[-1]
            if uv.ok and uv.stdout.strip()
            else "uv not installed (run scripts/remote/bootstrap.sh)",
        )
    )
    df = await executor.run(["df", "-Pk", "."], check=False)
    free_gb = _free_gb(df.stdout) if df.ok else None
    if free_gb is None:
        checks.append(Check("disk", "warn", "could not read free disk space"))
    else:
        status: Status = "ok" if free_gb >= min_free_disk_gb else "warn"
        checks.append(Check("disk", status, f"{free_gb:.0f} GB free in the working directory"))
    if container_test_image and docker.ok:
        test = await executor.run(
            ["docker", "run", "--rm", "--gpus", "all", container_test_image, "nvidia-smi", "-L"],
            check=False,
            timeout_s=600,
        )
        checks.append(
            Check(
                "container-gpu",
                "ok" if test.ok else "fail",
                test.stdout.strip() or test.stderr.strip()[:200],
            )
        )
    return checks


def _free_gb(df_output: str) -> float | None:
    lines = df_output.strip().splitlines()
    if len(lines) < 2:
        return None
    parts = lines[-1].split()
    try:
        return int(parts[3]) * 1024 / 1e9
    except (IndexError, ValueError):
        return None
