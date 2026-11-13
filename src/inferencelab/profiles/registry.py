"""File-backed registry of performance profiles (one small JSON file per profile)."""

from __future__ import annotations

import re
from pathlib import Path

from inferencelab.profiles.model import PerformanceProfile

__all__ = ["ProfileRegistry"]

_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


class ProfileRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, profile_id: str) -> Path:
        if not _SAFE.match(profile_id):
            raise ValueError(f"invalid profile id {profile_id!r}")
        return self.root / f"{profile_id}.json"

    def save(self, profile: PerformanceProfile) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(profile.profile_id)
        path.write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def load(self, ref: str | Path) -> PerformanceProfile:
        path = Path(ref)
        if not path.suffix:
            path = self._path(str(ref))
        return PerformanceProfile.model_validate_json(path.read_text(encoding="utf-8"))

    def all(self) -> list[PerformanceProfile]:
        if not self.root.exists():
            return []
        return [self.load(p) for p in sorted(self.root.glob("*.json"))]

    def find(
        self, *, model: str | None = None, backend: str | None = None, gpu: str | None = None
    ) -> list[PerformanceProfile]:
        out = []
        for profile in self.all():
            if model is not None and profile.model.get("name") != model:
                continue
            if backend is not None and profile.backend.get("kind") != backend:
                continue
            if gpu is not None and profile.hardware.get("gpu") != gpu:
                continue
            out.append(profile)
        return out
