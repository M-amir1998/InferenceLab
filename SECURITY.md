# Security

InferenceLab sits in front of model servers and drives remote GPU machines, so it is
built to be safe by default. Please report vulnerabilities privately through GitHub's
"Report a vulnerability" (Security tab) rather than in public issues.

## Threats and controls

| Concern | Control |
|---|---|
| **Untrusted model endpoints** | Backends are operator-configured only; responses are parsed defensively (malformed SSE/JSON → classified errors, never guessed); error excerpts are truncated; engine output is never executed. |
| **Prompt confidentiality** | Prompts/completions are not stored by default (ADR-007); records keep a salted HMAC fingerprint whose salt is never persisted; logs, metric labels and trace spans never contain text; traces with text fields are rejected; `system.json` has no hostname, user or paths. |
| **API keys** | Gateway keys come only from an environment variable (`auth.api_keys_env`), are ≥ 16 characters, and are compared in constant time against every key. Backend keys are referenced by env-var *name* (`api_key_env`), never stored in configs. When auth is enabled the tenant is derived from the key, so `x-tenant-id` cannot be spoofed. |
| **TLS** | HTTP backends support `https` with certificate verification on by default (`verify_tls`). Terminate client TLS at a reverse proxy/ingress in front of the gateway; the gateway itself serves plain HTTP. |
| **Backend authentication** | Bearer token per backend from an env var; vLLM's `--api-key` guards `/v1/*` only (`/health`, `/metrics` stay open), SGLang exempts `/health*`, `/metrics*` — keep engine ports private. |
| **SSRF** | Backend URLs never come from requests. Configs are validated (http/https only, no embedded credentials); `security.allowed_backend_hosts` restricts backends to an allowlist so a tampered config cannot point the gateway at internal endpoints such as cloud metadata (`169.254.169.254`). |
| **Request-size limits** | Body size enforced before JSON parsing (413, also for chunked uploads); message-count and `max_tokens` caps; header ids validated (`[A-Za-z0-9._:-]{1,128}`). |
| **Model endpoint exposure** | Engine containers publish ports on `127.0.0.1` only; the load generator runs on the GPU host; Compose binds every port to localhost; OpenAPI/docs UIs are disabled on the gateway. |
| **Rate limiting / abuse** | Per-tenant token buckets (req/s, tokens/min) and concurrency caps; bounded admission queue with shedding; retry budget prevents retry storms; tenant state is bounded against id-flooding. |
| **Multi-tenant concerns** | Quotas are isolated per tenant; metrics never use tenant ids as labels; no response caching across tenants. Engine-level prefix caches are shared by design of the engines — tenants sharing an engine can in principle infer cache hits from timing; isolate sensitive tenants on dedicated deployments (vLLM supports `cache_salt`). |
| **Remote execution** | SSH with `BatchMode` (no interactive prompts) and `StrictHostKeyChecking=accept-new`; hostnames/users validated; `ProxyCommand`/`LocalCommand` options rejected; every remote command is `shlex`-quoted; only a key *path* is configured; secrets live in an env file on the host; machines are never created or deleted. |
| **Supply chain** | `uv.lock` pinned and checked in CI (`uv lock --check`); engine images pinned by tag + digest; Docker image runs as non-root (uid 10001), read-only in Compose with `no-new-privileges`. |
| **Secrets in git** | `.gitignore` covers `.env`, keys, weights, caches and results; `detect-secrets` (baseline) and optional `gitleaks` run in CI and pre-commit; `.env.example` holds placeholders only. |

## Out of scope (v0.1)

Enterprise identity (OIDC/SAML), per-tenant encryption, audit logging, mTLS between gateway
and engines (use a service mesh or private network), and multi-replica shared quota state.
