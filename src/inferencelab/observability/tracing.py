"""Optional OpenTelemetry tracing.

The OpenTelemetry *API* is a core dependency: without a configured SDK every span is a
zero-cost no-op, so normal operation needs no collector. Installing the ``tracing``
extra and enabling ``observability.tracing`` exports spans (console or OTLP/HTTP).

Spans: ``gateway.request`` → ``gateway.admission`` → ``gateway.route`` →
``backend.attempt`` (until first token) → ``backend.stream``. Attributes carry ids,
backend names and token counts — never prompt or completion text.
"""

from __future__ import annotations

import logging
from typing import Literal

from opentelemetry import trace
from pydantic import Field

from inferencelab import __version__
from inferencelab.schema import StrictModel

__all__ = ["TracingConfig", "configure_tracing", "get_tracer"]

log = logging.getLogger(__name__)


class TracingConfig(StrictModel):
    enabled: bool = False
    exporter: Literal["console", "otlp"] = "otlp"
    endpoint: str | None = Field(default=None, description="OTLP/HTTP endpoint (default: env)")
    service_name: str = "inferencelab-gateway"
    sample_ratio: float = Field(default=1.0, ge=0, le=1)


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("inferencelab", __version__)


def configure_tracing(config: TracingConfig) -> bool:
    """Install an SDK tracer provider if enabled and available. Returns True if active."""
    if not config.enabled:
        return False
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SpanExporter,
        )
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    except ImportError:
        log.warning(
            "tracing enabled but opentelemetry-sdk is missing: pip install 'inferencelab[tracing]'"
        )
        return False
    exporter: SpanExporter
    if config.exporter == "console":
        exporter = ConsoleSpanExporter()
    else:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            log.warning("OTLP exporter missing: pip install 'inferencelab[tracing]'")
            return False
        exporter = (
            OTLPSpanExporter(endpoint=config.endpoint) if config.endpoint else OTLPSpanExporter()
        )
    provider = TracerProvider(
        resource=Resource.create({"service.name": config.service_name}),
        sampler=TraceIdRatioBased(config.sample_ratio),
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return True
