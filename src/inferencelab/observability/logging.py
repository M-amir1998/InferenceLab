"""Structured logging. Log records carry ids, backends and timings — never prompt text."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

__all__ = ["JSONFormatter", "configure_logging"]

_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {"message"}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        body: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        body.update(
            {k: v for k, v in vars(record).items() if k not in _STANDARD and not k.startswith("_")}
        )
        if record.exc_info:
            body["exc"] = self.formatException(record.exc_info)
        return json.dumps(body, default=str)


def configure_logging(level: str = "info", *, json_logs: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    if json_logs:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger("inferencelab")
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root.propagate = False
