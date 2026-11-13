# ADR-001 — HTTP backend abstraction

**Status:** accepted · **Date:** 2026-11-13

## Context

InferenceLab must drive vLLM, SGLang and TensorRT-LLM, whose Python APIs change every
release, require CUDA, and pin conflicting dependency versions (torch, CUDA runtime).
Development and CI happen on CPU-only machines.

## Decision

Integrate with every engine **over its OpenAI-compatible HTTP server**. The core package
never imports engine libraries. Engine-specific knowledge is limited to (a) launch builders
that render pinned `docker run` commands and (b) metric-name maps for each engine's
Prometheus endpoint. One generic adapter (`OpenAICompatibleBackend`) does the I/O.

## Consequences

* Version independence: an engine upgrade means a new image tag and re-verified flags, not
  a dependency change.
* Failure isolation: an engine crash cannot crash the gateway.
* Deployment flexibility: engines can live on another host, behind a tunnel, in a container.
* The whole core installs and tests on a laptop.
* Cost: an HTTP hop and SSE parsing add a small overhead (measurable by comparing a
  direct-to-engine run with a through-gateway run).
* Cost: features not exposed over HTTP (internal scheduler knobs, per-request engine
  timings) are available only through the engines' metrics endpoints or not at all — and
  are then reported as unavailable, never estimated.
* Direct Python adapters may be added later behind the same `InferenceBackend` protocol.
