"""Host facts recorded with every run — without hostname, user name or local paths."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess  # fixed argv, no shell  # nosec B404
import sys
from pathlib import Path
from typing import Any

import psutil

from inferencelab import __version__

__all__ = ["collect_system_info", "git_state", "nvidia_smi_query"]

_PACKAGES = ("fastapi", "uvicorn", "httpx", "pydantic", "numpy", "pyarrow", "prometheus-client")


def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_smi_query(timeout_s: float = 10.0) -> list[dict[str, str]] | None:
    """GPU name/memory/driver via nvidia-smi, or ``None`` when no NVIDIA GPU is present."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell  # nosec B603
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "memory_total_mib": parts[1], "driver": parts[2]})
    return gpus


def git_state(repo: Path | None = None, timeout_s: float = 5.0) -> dict[str, Any]:
    """Commit and dirty flag of the InferenceLab checkout, if run from one."""
    exe = shutil.which("git")
    if exe is None:
        return {}
    cwd = repo or Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(  # noqa: S603 - fixed argv, no shell  # nosec B603
            [exe, "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603 - fixed argv, no shell  # nosec B603
            [exe, "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"commit": commit, "dirty": bool(dirty)}


def collect_system_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_logical": os.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(vm.total / 1e9, 1),
        "inferencelab": __version__,
        "packages": {p: _version(p) for p in _PACKAGES},
        "gpus": nvidia_smi_query(),
    }
