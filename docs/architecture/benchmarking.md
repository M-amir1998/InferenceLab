# Benchmarking

```text
BenchmarkConfig ─► target (HTTP endpoint | in-process backend | in-process gateway)
               ─► WorkloadPlan (deterministic sessions from the seed)
               ─► engine metrics snapshot (before)
               ─► LoadGenerator (timestamps on the injected clock)
               ─► engine metrics snapshot (after) ─► counter deltas, phase means
               ─► RunSummary (metrics, goodput, SLO, errors, tokens, cost)
               ─► results/<run-id>/ artifacts + report.md + plots
```

## Load generator

* **Open loop** (constant, Poisson, burst, step, trace): sessions start on schedule
  regardless of server speed. In-flight sessions are bounded by `max_in_flight`; when the
  bound is reached a session is **dropped and recorded**, never delayed — delaying would
  silently turn an open-loop test into a closed-loop one (coordinated omission). Every
  record keeps its intended send time, so client lag is measured and reported.
* **Closed loop**: exactly `concurrency` workers.
* Multi-turn sessions resend the full conversation including generated text, with think
  time between turns; a failed turn ends the session.
* Client-side cancellation (`cancel_probability`, `cancel_after_tokens`) exercises the
  cancellation path end to end.
* Warmup/cooldown by request count or by time are marked per record and excluded from
  metrics, but kept in `requests.parquet`.

## Targets

| Target | Use |
|---|---|
| `http` | any OpenAI-compatible endpoint: the gateway, or an engine directly (the GPU path; run on the GPU host so WAN latency does not pollute TTFT) |
| `backend` | an in-process backend (CPU simulator studies, fast) |
| `gateway` | the gateway data plane in-process (routing/admission without HTTP) |

`virtual_time: true` runs in-process targets on the discrete-event clock: a 60-second
simulated run completes in about a second and is bit-for-bit reproducible.

## Artifacts

```text
results/<run-id>/
├── config.yaml        resolved config (env-var names only, never secret values)
├── metadata.json      versions, git commit, label, target, engine version, privacy settings
├── requests.parquet   one row per request (timings, tokens, status, ids; no text by default)
├── summary.json       RunSummary
├── profile.json       single-point PerformanceProfile
├── system.json        client host facts (no hostname, user or paths)
├── report.md          human-readable report
└── plots/             TTFT/E2E CDFs, in-flight & queue depth timeline
```

`inferencelab report RUN` regenerates `report.md`/plots from the stored artifacts;
`inferencelab benchmark compare RUN...` builds comparison tables from measured values only.

## Privacy

Prompts and completions are **not** stored by default. Each record carries a 16-hex
HMAC-SHA256 fingerprint of the prompt keyed with a per-run random salt that is never
persisted — useful to spot duplicates within a run, useless for dictionary attacks.
`capture_text: true` opts in to storing text (ADR-007). Trace files describe request shapes
only; a trace line containing text fields is rejected.

## Experiments and safety

`experiments/` expands matrices explicitly and refuses to run more than `max_runs`.
Before any remote run it shows the number of runs, known/upper-bound duration, hardware and
an upper-bound cost when a price is configured. Remote (paid) execution requires
`--execute`; local mock/simulated experiments run directly.
