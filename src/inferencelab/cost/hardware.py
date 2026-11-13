"""Hardware profile registry (specifications only — never prices)."""

from __future__ import annotations

from pathlib import Path

from inferencelab.config.loader import load_model_file
from inferencelab.domain.hardware import HardwareProfile
from inferencelab.domain.model import ModelSpec, Quantization

__all__ = ["HardwareRegistry", "WeightFit", "weight_fit"]


class HardwareRegistry:
    def __init__(self, profiles: dict[str, HardwareProfile]) -> None:
        self._profiles = profiles

    @classmethod
    def from_dir(cls, directory: Path) -> HardwareRegistry:
        profiles: dict[str, HardwareProfile] = {}
        for path in sorted(directory.glob("*.yaml")):
            profile = load_model_file(path, HardwareProfile)
            profiles[profile.name] = profile
        return cls(profiles)

    def get(self, name: str) -> HardwareProfile:
        try:
            return self._profiles[name]
        except KeyError:
            raise KeyError(
                f"unknown hardware profile {name!r}; known: {sorted(self._profiles)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._profiles)

    def all(self) -> list[HardwareProfile]:
        return [self._profiles[n] for n in self.names()]


class WeightFit:
    def __init__(self, weights_gb: float | None, capacity_gb: float, ratio: float | None) -> None:
        self.weights_gb = weights_gb
        self.capacity_gb = capacity_gb
        self.ratio = ratio

    @property
    def fits(self) -> bool | None:
        return None if self.ratio is None else self.ratio < 0.9

    def describe(self) -> str:
        if self.weights_gb is None:
            return "unknown (model parameter count not recorded)"
        verdict = "fits" if self.fits else "DOES NOT FIT"
        return (
            f"weights ~{self.weights_gb:.1f} GB of {self.capacity_gb:.0f} GB "
            f"({(self.ratio or 0) * 100:.0f}%) - {verdict}; KV cache and activations need the rest"
        )


def weight_fit(
    model: ModelSpec, hardware: HardwareProfile, gpu_count: int, quant: Quantization
) -> WeightFit:
    """Pre-flight memory sanity check (weights only). Never a throughput estimate."""
    weights = model.weight_memory_gb(quant)
    capacity = hardware.vram_gb * max(1, gpu_count)
    return WeightFit(weights, capacity, None if weights is None else weights / capacity)
