# ADR-002 — OpenAI-compatible gateway

**Status:** accepted · **Date:** 2026-11-13

## Context

Clients, SDKs and the engines themselves already speak the OpenAI Chat Completions /
Completions API. A custom API would force clients to change and make the gateway harder to
place in front of existing services.

## Decision

The gateway implements the useful **serving subset** of the OpenAI API (chat, completions,
streaming with usage, models) and documents a compatibility matrix
(docs/architecture/gateway.md). Engine extensions (`response_format`, `tools`, penalties,
`ignore_eos`, …) are passed through verbatim from an allowlist; unknown fields are ignored.
InferenceLab metadata travels in `x-inferencelab-*` / `x-*` headers so bodies stay
standard. Errors use OpenAI's envelope with InferenceLab's error kind in `error.code`.

## Consequences

* Any OpenAI client works unchanged; the benchmark client can target the gateway or an
  engine directly with the same code; one wire module serves every hop.
* Not every OpenAI feature is supported (`n > 1`, `echo`, multimodal content,
  Assistants/Responses APIs). Unsupported features fail explicitly with 400 rather than
  being emulated.
