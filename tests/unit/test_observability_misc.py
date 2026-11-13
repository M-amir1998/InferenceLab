from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from inferencelab.observability.logging import JSONFormatter, configure_logging
from inferencelab.observability.prometheus import GatewayMetrics
from inferencelab.observability.tracing import TracingConfig, configure_tracing, get_tracer
from inferencelab.remote.executor import CommandError, LocalExecutor


def test_json_log_formatter_includes_extras_but_no_message_args() -> None:
    record = logging.LogRecord(
        "inferencelab.x", logging.INFO, __file__, 1, "routed %s", ("r1",), None
    )
    record.backend = "vllm-a"
    data = json.loads(JSONFormatter().format(record))
    assert data["msg"] == "routed r1" and data["backend"] == "vllm-a" and data["level"] == "info"


def test_configure_logging_installs_single_handler() -> None:
    configure_logging("debug", json_logs=True)
    configure_logging("info")
    logger = logging.getLogger("inferencelab")
    assert len(logger.handlers) == 1 and logger.level == logging.INFO and not logger.propagate


def test_tracing_is_a_noop_unless_enabled() -> None:
    assert configure_tracing(TracingConfig()) is False
    with get_tracer().start_as_current_span("x") as span:
        span.set_attribute("inferencelab.request_id", "r")


def test_console_tracing_can_be_enabled() -> None:
    assert (
        configure_tracing(TracingConfig(enabled=True, exporter="console", sample_ratio=0.0)) is True
    )


def test_metrics_registry_is_isolated_per_instance() -> None:
    a, b = GatewayMetrics(), GatewayMetrics()
    a.requests.labels("m", "x", "ok").inc()
    assert b"inferencelab_requests_total" in a.render()
    assert (
        b.registry.get_sample_value(
            "inferencelab_requests_total", {"model": "m", "backend": "x", "status": "ok"}
        )
        is None
    )


async def test_local_executor(tmp_path: Path) -> None:
    ex = LocalExecutor()
    ok = await ex.run(["python", "-c", "print('hi')"])
    assert ok.ok and ok.stdout.strip() == "hi"
    missing = await ex.run(["definitely-not-a-binary-xyz"], check=False)
    assert missing.returncode == 127
    with pytest.raises(CommandError):
        await ex.run(["python", "-c", "import sys; sys.exit(3)"])
    timed = await ex.run(["python", "-c", "import time; time.sleep(5)"], timeout_s=0.2, check=False)
    assert timed.returncode == 124
    src = tmp_path / "src.txt"
    src.write_text("x", encoding="utf-8")
    await ex.put(src, str(tmp_path / "dest"))
    assert (tmp_path / "dest" / "src.txt").read_text(encoding="utf-8") == "x"
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a").write_text("a", encoding="utf-8")
    await ex.get(str(folder), tmp_path / "copy")
    assert (tmp_path / "copy" / "folder" / "a").exists()
