# Routing

A router **ranks** candidate backends; the data plane walks the ranking for the first
attempt and for retries. Routers are pure functions of immutable `BackendView` snapshots
(health, circuit state, outstanding requests, EWMA TTFT/TPOT, peak-EWMA latency, cost,
capacity) and of request metadata — never of prompt text.

## Policies

| Policy | Rule | Good for |
|---|---|---|
| `round_robin` | rotate through candidates | homogeneous fleets, baselines |
| `random` | seeded uniform shuffle | baselines, reproducible tests |
| `least_outstanding` | fewest in-flight requests; ties rotate | heterogeneous fleets, simple and robust |
| `latency_aware` | lowest peak-EWMA TTFT × (outstanding + 1); unmeasured backends tried first | heterogeneous speed; reacts fast to slowdowns |
| `cost_aware` | cheapest backend with spare capacity (`max_outstanding`), then spill | mixed cheap/expensive pools |
| `slo_aware` | cheapest backend whose *predicted* TTFT (EWMA × (1 + utilisation)) fits the remaining TTFT budget; otherwise the fastest, flagged `slo_at_risk` | SLO-bound traffic with mixed pools |

*Peak EWMA* (as in Finagle/Linkerd) jumps up immediately on a slow observation and decays
over time, so a degrading backend loses traffic quickly and regains it gradually.

## Prefix / session affinity

`PrefixAffinityRouter` wraps any policy. When a request carries an affinity key
(`prefix_key`, `conversation_id` or `tenant_id`, in configured order) it prefers the backend
chosen by **rendezvous (highest-random-weight) hashing**, so the engine that already holds
the prefix/conversation in its KV cache gets the request. Adding or removing a backend only
remaps keys that lived on it.

Hot keys are contained with **bounded loads** (Mirrokni et al.): the preferred backend is
skipped if its load would exceed `load_factor` × the mean load; the request then falls back
to the inner policy. InferenceLab never sees the cache — it only routes by identifiers;
the KV cache stays inside the engine. (SGLang's own model gateway implements a server-side
cache-aware policy based on an approximate radix tree of request text; InferenceLab's
approach needs no prompt inspection and works across engines.)

The CPU demo shows the effect on multi-turn chat: with affinity, more prompt tokens are
served from the simulated engines' prefix cache and TTFT drops (`reports/local-simulated/`).

## Tiered fallback

Each backend has `routing.tier` (0 = preferred, 1 = secondary, 2+ = fallback).
`rank_tiered` excludes unhealthy and circuit-open backends, ranks each tier with the
policy, and concatenates tiers in order. A secondary tier is therefore used only when every
routable preferred backend has been tried or none is routable:

```text
preferred (tier 0, policy order) → secondary (tier 1) → fallback (tier 2)
```

## Extending

A policy implements `rank(ctx, candidates) -> RoutingDecision` returning every candidate
exactly once (a property test checks this for all policies). Register it in
`routing/factory.py:build_router`.
