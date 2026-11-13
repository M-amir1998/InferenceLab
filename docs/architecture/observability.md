# Observability

## Prometheus (`GET /metrics`)

Each gateway instance owns its registry. Label values are bounded (backend, model, enum
status/kind/reason) — never request ids or tenant ids.

| Metric | Type | Labels |
|---|---|---|
| `inferencelab_requests_total` | counter | model, backend, status (`ok`/`error`/`timeout`/`rejected`/`cancelled`) |
| `inferencelab_ttft_seconds` | histogram | backend — gateway receipt → first token (includes queueing, retries) |
| `inferencelab_tpot_seconds` | histogram | backend — per request, after the first token |
| `inferencelab_e2e_seconds` | histogram | backend |
| `inferencelab_queue_wait_seconds` | histogram | — |
| `inferencelab_queue_depth`, `inferencelab_in_flight_requests` | gauge | — |
| `inferencelab_backend_healthy`, `inferencelab_backend_outstanding_requests` | gauge | backend |
| `inferencelab_circuit_state` | gauge | backend (0 closed, 1 half-open, 2 open) |
| `inferencelab_circuit_transitions_total` | counter | backend, to_state |
| `inferencelab_retries_total` | counter | backend, reason |
| `inferencelab_errors_total` | counter | backend, kind (every failed attempt, including retried ones) |
| `inferencelab_rejections_total` | counter | reason (`rate_limited`, `queue_full`, `load_shed`, `retry_budget`, …) |
| `inferencelab_tokens_total` | counter | backend, direction (`input`/`output`), source (`reported`/`estimated`) |
| `inferencelab_router_decisions_total` | counter | policy, backend |
| `inferencelab_routing_slo_at_risk_total` | counter | — |

The serving app for local backends exports `inferencelab_engine_*` (running/waiting
requests, KV usage, prefix-cache queries/hits, preemptions, token counters). Real engines
are scraped directly (vLLM `vllm:*`, SGLang `sglang:*` with `--enable-metrics`,
TensorRT-LLM `trtllm_*` at `/prometheus/metrics`).

## Grafana

`docker compose up` provisions Prometheus and a Grafana dashboard
(`deployment/grafana/dashboards/inferencelab.json`): requests/s by outcome, token
throughput, TTFT p50/p95/p99, TPOT p50/p95, E2E, queue depth and in-flight, errors by kind,
rejections by reason, backend health, circuit state, retries, router decisions, outstanding
per backend, engine KV usage and prefix-cache hit rate. Neither is required for tests.

## Tracing (optional)

The OpenTelemetry **API** is a core dependency; without an SDK every span is a no-op.
`pip install 'inferencelab[tracing]'` and `observability.tracing.enabled: true` export spans
(console or OTLP/HTTP): `gateway.request` → `gateway.admission` → `gateway.route` →
`backend.attempt` (until first token). Attributes carry ids, policy, ranking, backend,
attempt and status — never prompt or completion text.

## Logs

`observability.log_json: true` emits one JSON object per line. Logs never contain prompts
or completions.
