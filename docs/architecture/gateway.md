# Gateway

`inferencelab gateway serve CONFIG` runs an OpenAI-compatible gateway (FastAPI + Uvicorn).

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness (process up) |
| `GET /ready` | readiness: 200 if at least one backend is healthy with a closed/half-open circuit, else 503 |
| `GET /v1/models` | logical models served by configured backends |
| `POST /v1/chat/completions` | chat, streaming and non-streaming |
| `POST /v1/completions` | text completion, streaming and non-streaming |
| `GET /metrics` | Prometheus exposition (see observability.md) |
| `GET /inferencelab/backends` | operator view: health, circuit state, outstanding, EWMA TTFT/TPOT, capabilities |

OpenAPI/Swagger UIs are disabled by default (no unauthenticated schema exposure).

## OpenAI compatibility matrix

| Feature | Status |
|---|---|
| `model`, `messages` (string or text parts), `prompt` (string or single-element list) | supported |
| `max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `stop`, `seed`, `user` | supported |
| `stream`, `stream_options.include_usage` | supported (final usage chunk with `choices: []`) |
| `n` | only `n=1` |
| `echo` | not supported (400) |
| Multimodal content parts | not supported (text parts only) |
| `response_format`, `tools`, `tool_choice`, `logprobs`, `top_logprobs`, `logit_bias`, penalties, `top_k`, `min_p`, `min_tokens`, `ignore_eos` | passed through verbatim to the engine; behaviour is the engine's |
| Unknown fields | ignored |
| Embeddings, images, audio, files, batches, Assistants/Responses APIs | not implemented |

`developer` messages are mapped to `system`. Usage comes from the engine when reported;
otherwise it is estimated and the response header `x-inferencelab-token-source: estimated`
says so.

## InferenceLab headers

| Request header | Effect |
|---|---|
| `x-request-id` | propagated (validated `[A-Za-z0-9._:-]{1,128}`) |
| `x-tenant-id` | tenant for quotas (ignored when auth is enabled: tenant comes from the API key) |
| `x-conversation-id`, `x-prefix-key` | routing affinity keys (identifiers only; prompts are never inspected) |
| `x-priority` | `low` / `normal` / `high` for admission |

| Response header | Meaning |
|---|---|
| `x-inferencelab-backend`, `x-inferencelab-deployment` | where the request ran |
| `x-inferencelab-attempts` | attempts including retries |
| `x-inferencelab-queue-ms` | time spent in the admission queue |
| `x-inferencelab-route-policy` | routing policy (streaming responses) |

## Security controls

Optional API-key auth (`auth.enabled`, keys only from an environment variable, constant-time
comparison), request body limit enforced before JSON parsing (413), message-count and
`max_tokens` caps, tenant quotas, an SSRF guard (`security.allowed_backend_hosts`) and
backend URLs validated at load time (http/https only, no embedded credentials). Backend
URLs always come from operator configuration, never from requests. See `SECURITY.md`.
