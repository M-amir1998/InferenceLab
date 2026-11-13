# ADR-005 — Routing architecture

**Status:** accepted · **Date:** 2026-11-13

## Context

Routing must support simple baselines and load-, latency-, cost- and SLO-aware policies,
session and prefix affinity, and preferred → secondary → fallback tiers, while staying
testable and explainable.

## Decision

* Routers are **pure rankers** over immutable `BackendView` snapshots and return every
  candidate exactly once. The data plane owns I/O and walks the ranking for retries, so the
  fallback order and the retry order are the same decision.
* Tiers are applied outside the policy (`rank_tiered`): unhealthy and circuit-open backends
  are excluded, each tier is ranked, tiers are concatenated.
* Affinity is a wrapper (`PrefixAffinityRouter`) using rendezvous hashing with bounded loads
  over opaque ids (`prefix_key`, `conversation_id`, `tenant_id`); prompts are never inspected.
* Runtime signals: outstanding requests, EWMA TTFT/TPOT, peak-EWMA latency, circuit state,
  health. Costs and capacities are configured (from measurements).
* State is process-local. Gateway replicas share no routing or quota state in v0.1; a shared
  store (e.g. Redis) would be optional future work.

## Consequences

* Each policy is a few lines and is property-tested (permutation, tier order, determinism);
  new policies plug in without touching the data plane.
* Several gateway replicas each learn latencies independently, and tenant quotas are
  per replica.
