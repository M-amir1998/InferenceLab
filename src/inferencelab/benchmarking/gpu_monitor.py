"""GPU memory/utilisation sampling during real GPU runs (non-blocking, via nvidia-smi).

Started automatically by the runner for ``execution: gpu`` when ``nvidia-smi`` exists.
Peak memory is recorded so that every remote run documents how close it came to the
VRAM limit (runs must never be configured to OOM on purpose).
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from types import TracebackType
from typing import Any

__all__ = ["GpuMemorySampler", "parse_gpu_samples"]

_QUERY = "--query-gpu=index,memory.used,memory.total,utilization.gpu"


def parse_gpu_samples(text: str) -> list[tuple[int, float, float, float]]:
    """``index, used MiB, total MiB, util %`` rows -> tuples (MiB values)."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    return rows


class GpuMemorySampler:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.exe = shutil.which("nvidia-smi")
        self.samples: list[tuple[int, float, float, float]] = []
        self._task: asyncio.Task[None] | None = None

    @property
    def available(self) -> bool:
        return self.exe is not None

    async def _sample_once(self) -> None:
        if self.exe is None:
            return
        proc = await asyncio.create_subprocess_exec(
            self.exe,
            _QUERY,
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        self.samples.extend(parse_gpu_samples(out.decode(errors="replace")))

    async def _loop(self) -> None:
        while True:
            with contextlib.suppress(OSError):
                await self._sample_once()
            await asyncio.sleep(self.interval_s)

    async def __aenter__(self) -> GpuMemorySampler:
        if self.available:
            self._task = asyncio.get_running_loop().create_task(self._loop())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def summary(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        gpus = sorted({s[0] for s in self.samples})
        peak_per_gpu = {g: max(s[1] for s in self.samples if s[0] == g) for g in gpus}
        total_per_gpu = {g: max(s[2] for s in self.samples if s[0] == g) for g in gpus}
        return {
            "gpus": len(gpus),
            "peak_gb": round(max(peak_per_gpu.values()) / 1024, 2),
            "peak_gb_per_gpu": {str(g): round(v / 1024, 2) for g, v in peak_per_gpu.items()},
            "total_gb_per_gpu": {str(g): round(v / 1024, 2) for g, v in total_per_gpu.items()},
            "mean_utilization_pct": round(sum(s[3] for s in self.samples) / len(self.samples), 1),
            "samples": len(self.samples),
            "source": "nvidia-smi (sampled ~1 Hz on the GPU host)",
        }
