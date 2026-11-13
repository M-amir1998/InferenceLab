"""Trace-driven workloads from JSONL *shape* traces.

Each line describes one request's shape, never its content::

    {"input_tokens": 812, "output_tokens": 128, "timestamp_s": 0.42,
     "conversation_id": "c-17", "prefix_key": "rag-doc-3"}

Only the fields below are read. Lines containing prompt/response text are rejected so
that a private production log cannot be ingested (and later committed) by accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["TraceEntry", "load_trace"]

_TEXT_FIELDS = {"prompt", "messages", "text", "content", "completion", "response", "output"}


@dataclass(frozen=True, slots=True)
class TraceEntry:
    input_tokens: int
    output_tokens: int
    timestamp_s: float | None = None
    conversation_id: str | None = None
    prefix_key: str | None = None


def load_trace(path: Path, *, max_entries: int | None = None) -> list[TraceEntry]:
    entries: list[TraceEntry] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
            leaked = _TEXT_FIELDS & row.keys()
            if leaked:
                raise ValueError(
                    f"{path}:{lineno}: trace contains text fields {sorted(leaked)}; "
                    "InferenceLab traces must only describe request shapes"
                )
            try:
                entry = TraceEntry(
                    input_tokens=int(row["input_tokens"]),
                    output_tokens=int(row["output_tokens"]),
                    timestamp_s=None
                    if row.get("timestamp_s") is None
                    else float(row["timestamp_s"]),
                    conversation_id=None
                    if row.get("conversation_id") is None
                    else str(row["conversation_id"]),
                    prefix_key=None if row.get("prefix_key") is None else str(row["prefix_key"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid trace entry: {exc}") from exc
            if entry.input_tokens < 1 or entry.output_tokens < 1:
                raise ValueError(f"{path}:{lineno}: token counts must be >= 1")
            entries.append(entry)
            if max_entries is not None and len(entries) >= max_entries:
                break
    timestamps = [e.timestamp_s for e in entries if e.timestamp_s is not None]
    if timestamps and timestamps != sorted(timestamps):
        raise ValueError(f"{path}: timestamp_s values must be non-decreasing")
    return entries
