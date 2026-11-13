# ADR-004 — No GPU dependency in core development

**Status:** accepted · **Date:** 2026-11-13

## Context

The development laptop is CPU-only (16 GB RAM); cloud GPUs are rented temporarily for real
benchmarks. Correctness of the gateway, routing, admission, metrics, reporting and planning
code must not depend on GPU availability.

## Decision

Everything except real-engine performance is developed and tested on CPU:

* a deterministic **MockBackend** with fault injection;
* a **SimulatedInferenceBackend** that models continuous batching, a paged KV cache, prefix
  caching, chunked prefill and preemption (a systems model, not GPU physics);
* a **VirtualClock** so timing logic is deterministic and tests never sleep;
* GPU-only steps (launch, audit, campaign) are tested through a dry-run executor that
  records and checks the exact commands.

Simulated results are always labelled SIMULATED BACKEND RESULTS, and no GPU number is
written anywhere until it is measured.

## Consequences

* Fast, deterministic, offline CI; the full stack demos on a laptop.
* Simulator numbers say nothing about real engines; real-engine behaviour (kernels, memory
  fragmentation, CUDA graphs) is only learned on GPU runs.
