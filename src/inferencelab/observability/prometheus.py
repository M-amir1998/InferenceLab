"""Gateway Prometheus metrics (``/metrics``).

Each gateway instance owns its own :class:`CollectorRegistry`, so tests and multiple
in-process gateways never collide on the global registry. Label values are bounded
(backend names, models and status/reason enums) — never request ids or tenant ids, which
would explode cardinality.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST

__all__ = ["CONTENT_TYPE_LATEST", "GatewayMetrics"]

TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 30, 60)
TPOT_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 1)
E2E_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600)
QUEUE_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class GatewayMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        r = registry or CollectorRegistry()
        self.registry = r
        self.requests = Counter(
            "inferencelab_requests_total",
            "Finished gateway requests by outcome",
            ["model", "backend", "status"],
            registry=r,
        )
        self.ttft = Histogram(
            "inferencelab_ttft_seconds",
            "Time to first token observed at the gateway",
            ["backend"],
            buckets=TTFT_BUCKETS,
            registry=r,
        )
        self.tpot = Histogram(
            "inferencelab_tpot_seconds",
            "Mean time per output token after the first, per request",
            ["backend"],
            buckets=TPOT_BUCKETS,
            registry=r,
        )
        self.e2e = Histogram(
            "inferencelab_e2e_seconds",
            "End-to-end latency including queueing",
            ["backend"],
            buckets=E2E_BUCKETS,
            registry=r,
        )
        self.queue_wait = Histogram(
            "inferencelab_queue_wait_seconds",
            "Time spent in the admission queue",
            buckets=QUEUE_BUCKETS,
            registry=r,
        )
        self.queue_depth = Gauge(
            "inferencelab_queue_depth", "Requests waiting for admission", registry=r
        )
        self.in_flight = Gauge(
            "inferencelab_in_flight_requests", "Requests admitted and executing", registry=r
        )
        self.backend_healthy = Gauge(
            "inferencelab_backend_healthy",
            "1 if the backend passed health checks",
            ["backend"],
            registry=r,
        )
        self.backend_outstanding = Gauge(
            "inferencelab_backend_outstanding_requests",
            "In-flight requests per backend",
            ["backend"],
            registry=r,
        )
        self.circuit_state = Gauge(
            "inferencelab_circuit_state",
            "Circuit breaker state: 0=closed, 1=half_open, 2=open",
            ["backend"],
            registry=r,
        )
        self.circuit_transitions = Counter(
            "inferencelab_circuit_transitions_total",
            "Circuit breaker state transitions",
            ["backend", "to_state"],
            registry=r,
        )
        self.retries = Counter(
            "inferencelab_retries_total", "Retried attempts", ["backend", "reason"], registry=r
        )
        self.errors = Counter(
            "inferencelab_errors_total", "Failed attempts by kind", ["backend", "kind"], registry=r
        )
        self.rejections = Counter(
            "inferencelab_rejections_total",
            "Requests rejected before reaching a backend",
            ["reason"],
            registry=r,
        )
        self.tokens = Counter(
            "inferencelab_tokens_total",
            "Tokens processed (source=reported|estimated)",
            ["backend", "direction", "source"],
            registry=r,
        )
        self.router_decisions = Counter(
            "inferencelab_router_decisions_total",
            "Primary routing choices",
            ["policy", "backend"],
            registry=r,
        )
        self.slo_at_risk = Counter(
            "inferencelab_routing_slo_at_risk_total",
            "Routing decisions where no backend was predicted to meet the TTFT budget",
            registry=r,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
