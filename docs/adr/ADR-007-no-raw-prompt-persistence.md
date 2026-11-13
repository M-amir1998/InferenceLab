# ADR-007 — Raw prompts are not persisted by default

**Status:** accepted · **Date:** 2026-11-13

## Context

Benchmark artifacts are shared, archived by CI and sometimes committed as reports. Prompts
and completions can contain confidential data, and production traces certainly do.

## Decision

* Per-request records store token counts, timings, status, ids and a 16-hex HMAC-SHA256
  prompt fingerprint keyed with a **per-run random salt that is never persisted** — enough
  to detect duplicates within a run, useless for dictionary attacks.
* Prompt and completion text are stored only with explicit `capture_text: true`.
* Trace files describe request shapes only; lines with text fields are rejected.
* Logs, metric labels and trace spans never contain prompt text.
* `system.json` records no hostname, user name or local paths.

## Consequences

* Results can be shared safely by default.
* Output-quality analysis (e.g. valid-JSON rate) requires opting into text capture for that
  run.
