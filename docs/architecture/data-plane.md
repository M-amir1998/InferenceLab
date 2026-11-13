# Data plane

The data plane is `gateway/service.py:InferenceService` plus the components it composes.
It is framework-independent: FastAPI only translates HTTP to `InferenceRequest` objects
(`gateway/translate.py`) and responses back to OpenAI JSON/SSE.

## Request lifecycle

```text
received ─► tenant lease (token buckets: req/s, tokens/min; concurrency)
         ─► admission permit (bounded priority queue; may shed)
         ─► route: candidates serving the model ─► tiered ranking by the policy
         ─► attempt loop
               pick next candidate whose circuit admits a call
               dispatch: backend.stream(request)
               await first chunk  (first-token timeout ∧ overall deadline)
                 └─ failure before output: breaker/stats/metrics, backoff, next candidate
         ─► stream remaining chunks (idle timeout ∧ overall deadline)
         ─► finalize: circuit verdict, EWMA stats, Prometheus, tenant refund, release slot
```

Every request goes to the backend as a *stream*, even non-streaming client requests. That
gives uniform TTFT/TPOT metrics and lets the first-token timeout apply everywhere.

## Retries: safe vs unsafe

| Situation | Retry? | Why |
|---|---|---|
| Failure before the first token (connect error, 5xx, 429/503 from engine, first-token timeout, malformed response) | yes, bounded, preferably another backend | nothing reached the client |
| Streaming response after output started | **never** | a second generation would duplicate or contradict user-visible text; the stream ends with an error event |
| Non-streaming response, failure mid-generation | yes (configurable) | output is still buffered in the gateway |
| 4xx from the engine, cancellation, rate limit, overall deadline | no | retrying cannot help |

Retries are bounded three ways: `max_attempts`, the request's overall deadline, and a
**retry budget** (retries ≤ `budget_ratio` × requests, Finagle/Envoy style) so a sick fleet
is not hit by a retry storm.

## Timeouts

| Timeout | Scope | Enforced by |
|---|---|---|
| connect | TCP/TLS connect to the engine | httpx (`connect_timeout_s` on HTTP backends) |
| first token | dispatch → first content token, per attempt | `next_with_timeout` on the injected clock |
| idle | gap between consecutive chunks | same |
| overall | gateway receipt → end, across queueing and all attempts (deadline) | same |

A timeout cancels the backend call, which closes the HTTP connection — the engine's
signal to abort the request and free its KV cache.

## Circuit breakers

One per backend: `CLOSED → OPEN` after N consecutive failures or a failure rate over a
sliding window; `OPEN → HALF_OPEN` after the cooldown; limited concurrent probes; success
threshold to close. Permits are stamped with a *generation* so a slow call admitted in an
earlier state cannot flip the current state. Only backend-health failures count (5xx,
timeouts, malformed responses, disconnects) — client errors and cancellations do not.

## Backpressure and load shedding

* `queue.max_concurrency` requests execute; up to `queue.max_pending_requests` wait in a
  priority queue (HIGH > NORMAL > LOW, FIFO within a priority). Nothing queues unbounded.
* Shedding policies: `reject_newest` (503 `queue_full`), `reject_lowest_priority` (evict a
  lower-priority waiter), `slo_aware` (reject on arrival if the Little's-law predicted wait
  exceeds `slo_queue_budget_s`), plus `max_queue_wait_s`.
* Tenant quotas (`limits/tenant.py`) answer 429 with `Retry-After`; token quotas reserve
  `input estimate + max_tokens` and refund the unused part on completion.

## Streaming and cancellation

`openai_api/streaming.py` formats SSE (`role` chunk, deltas, finish chunk, optional usage
chunk, `[DONE]`; mid-stream errors become an `error` event). Starlette abandons a response
iterator when the client disconnects; `ClosingStreamingResponse` always closes it, shielded
from anyio cancellation, so backend connections and engine slots are released immediately.
For non-streaming requests the handler races the work against `http.disconnect` and cancels
the backend call when the client goes away.

## Status codes

| Code | Kinds |
|---|---|
| 400 | `invalid_request`, engine 4xx (`backend_rejected`) |
| 401 | `unauthorized` |
| 413 | body larger than `limits.max_request_bytes` |
| 429 | `rate_limited` (with `Retry-After`) |
| 502 | `backend_error`, `backend_unavailable`, `malformed_response`, `stream_interrupted` |
| 503 | `queue_full`, `load_shed`, `no_healthy_backend`, `circuit_open`, `backend_overloaded` |
| 504 | `timeout_*` |

The error body follows OpenAI's envelope; `error.code` carries InferenceLab's error kind so
clients (and the benchmark client) can classify failures precisely.
