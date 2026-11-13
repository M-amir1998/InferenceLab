"""YAML configuration loading with file references.

Any mapping value of the form ``{"$ref": "path/to/file.yaml"}`` or a string ending in
``.yaml``/``.yml`` under a *reference key* (``workload``, ``slo``, ``deployment``,
``model``, ``gateway``, ``backend``, ``hardware``) is replaced by that file's parsed
content, resolved relative to the referencing file. This lets an experiment reuse the
same workload and SLO definitions across many runs.

Secrets never belong in YAML: configs reference environment variable *names*
(``api_key_env``), never values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from inferencelab.backends.config import BackendConfig

__all__ = ["REF_KEYS", "load_backend_config", "load_model_file", "load_yaml", "resolve_refs"]

M = TypeVar("M", bound=BaseModel)
REF_KEYS = frozenset({"workload", "slo", "deployment", "model", "gateway", "backend", "hardware"})
_MAX_DEPTH = 8


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _is_ref(value: Any) -> bool:
    return isinstance(value, str) and value.endswith((".yaml", ".yml"))


def resolve_refs(data: Any, base_dir: Path, *, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        raise ValueError("config references nested too deeply (cycle?)")
    if isinstance(data, dict):
        if set(data) == {"$ref"}:
            path = (base_dir / str(data["$ref"])).resolve()
            return resolve_refs(load_yaml(path), path.parent, depth=depth + 1)
        out: dict[str, Any] = {}
        for key, value in data.items():
            if key in REF_KEYS and _is_ref(value):
                path = (base_dir / value).resolve()
                out[key] = resolve_refs(load_yaml(path), path.parent, depth=depth + 1)
            else:
                out[key] = resolve_refs(value, base_dir, depth=depth)
        return out
    if isinstance(data, list):
        return [resolve_refs(v, base_dir, depth=depth) for v in data]
    return data


def load_model_file(path: Path, model: type[M]) -> M:
    """Load ``path`` (YAML), resolve references and validate it as ``model``."""
    raw = load_yaml(path)
    if raw is None:
        raise ValueError(f"{path} is empty")
    return model.model_validate(resolve_refs(raw, path.parent))


def load_backend_config(path: Path) -> BackendConfig:
    """Load a backend config YAML (discriminated by ``type``)."""
    from pydantic import TypeAdapter

    raw = load_yaml(path)
    if raw is None:
        raise ValueError(f"{path} is empty")
    adapter: TypeAdapter[BackendConfig] = TypeAdapter(BackendConfig)
    return adapter.validate_python(resolve_refs(raw, path.parent))
