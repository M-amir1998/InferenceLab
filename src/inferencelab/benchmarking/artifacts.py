"""Run directory layout and (de)serialisation.

``results/<run-id>/``::

    config.yaml      resolved benchmark configuration (env var *names* only, no secrets)
    metadata.json    provenance: versions, labels, target, privacy settings
    requests.parquet one row per request: timings, tokens, status (no prompt text by default)
    summary.json     RunSummary
    profile.json     PerformanceProfile (single load point)
    system.json      client host facts (no hostname, user or paths)
    report.md        human-readable report
    plots/           PNG plots (if matplotlib is installed)
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from inferencelab.domain.response import TokenCountSource
from inferencelab.domain.result import Phase, RequestRecord, RequestStatus
from inferencelab.metrics.summary import RunSummary
from inferencelab.profiles.model import PerformanceProfile

__all__ = [
    "LoadedRun",
    "load_run",
    "read_records",
    "resolve_run_dir",
    "write_json",
    "write_records",
    "write_yaml",
]

_FIELDS = [f.name for f in dataclasses.fields(RequestRecord)]


def write_records(path: Path, records: Sequence[RequestRecord]) -> None:
    columns: dict[str, list[Any]] = {name: [] for name in _FIELDS}
    for record in records:
        for name in _FIELDS:
            value = getattr(record, name)
            if name == "chunk_times":
                value = list(value)
            elif isinstance(value, Phase | RequestStatus | TokenCountSource):
                value = value.value
            columns[name].append(value)
    schema_overrides = {"chunk_times": pa.list_(pa.float64())}
    arrays: dict[str, Any] = {}
    for name, values in columns.items():
        if name in schema_overrides:
            arrays[name] = pa.array(values, type=schema_overrides[name])
        elif all(v is None for v in values):
            arrays[name] = pa.array(values, type=pa.null())
        else:
            arrays[name] = pa.array(values)
    table = pa.table(arrays)
    pq.write_table(table, path, compression="zstd")


def read_records(path: Path) -> list[RequestRecord]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    out = []
    for row in rows:
        row["phase"] = Phase(row["phase"])
        row["status"] = RequestStatus(row["status"])
        row["token_source"] = TokenCountSource(row["token_source"])
        row["chunk_times"] = tuple(row.get("chunk_times") or ())
        kwargs: dict[str, Any] = {k: row[k] for k in _FIELDS if k in row}
        out.append(RequestRecord(**kwargs))
    return out


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8"
    )


def write_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class LoadedRun:
    run_dir: Path
    summary: RunSummary
    metadata: dict[str, Any]
    config: dict[str, Any]
    profile: PerformanceProfile | None

    def records(self) -> list[RequestRecord]:
        return read_records(self.run_dir / "requests.parquet")


def resolve_run_dir(ref: str, results_root: Path) -> Path:
    """A run reference is either a path to a run directory or a run id under results/."""
    candidate = Path(ref)
    if (candidate / "summary.json").exists():
        return candidate
    by_id = results_root / ref
    if (by_id / "summary.json").exists():
        return by_id
    matches = sorted(p for p in results_root.glob(f"*{ref}*") if (p / "summary.json").exists())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no run matching {ref!r} under {results_root}")
    raise FileNotFoundError(f"{ref!r} is ambiguous: {[m.name for m in matches]}")


def load_run(run_dir: Path) -> LoadedRun:
    summary = RunSummary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    profile_path = run_dir / "profile.json"
    profile = (
        PerformanceProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
        if profile_path.exists()
        else None
    )
    return LoadedRun(run_dir, summary, metadata, config or {}, profile)
