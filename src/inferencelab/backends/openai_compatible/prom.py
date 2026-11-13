"""Prometheus text parsing and engine-metric mapping.

Engines rename metrics between releases (e.g. vLLM's ``gpu_cache_usage_perc`` became
``kv_cache_usage_perc``), so every mapped field lists *candidate* sample names; the first
one present wins. Unmapped or absent metrics stay ``None`` — never guessed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from prometheus_client.parser import text_string_to_metric_families

from inferencelab.backends.base import BackendMetrics

__all__ = [
    "INFERENCELAB_SERVER_METRICS",
    "MetricMap",
    "MetricSpec",
    "build_metrics",
    "engine_phase_means",
    "parse_prometheus_text",
]


def engine_phase_means(
    phase_metrics: Mapping[str, str],
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, float]:
    """Mean engine-reported phase times over a window, from histogram ``_sum``/``_count``.

    ``phase_metrics`` maps a phase (``queue``, ``prefill``…) to a histogram base name.
    Only phases whose sum and count exist in both snapshots and whose count grew are
    returned, so a missing or idle metric is omitted rather than reported as zero.
    """
    out: dict[str, float] = {}
    for phase, base in phase_metrics.items():
        keys = (f"{base}_sum", f"{base}_count")
        if not all(k in before and k in after for k in keys):
            continue
        d_count = after[keys[1]] - before[keys[1]]
        if d_count <= 0:
            continue
        out[phase] = (after[keys[0]] - before[keys[0]]) / d_count
    return out


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Flatten exposition text to ``{sample_name: value}``, summing over label sets.

    Summing is correct for single-model servers whose series differ only by labels such
    as ``model_name`` or ``engine``; ``_created`` timestamps are dropped.
    """
    samples: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name.endswith("_created"):
                continue
            if sample.name.endswith("_bucket"):
                continue  # buckets are large and not needed for summaries
            samples[sample.name] = samples.get(sample.name, 0.0) + float(sample.value)
    return samples


@dataclass(frozen=True, slots=True)
class MetricSpec:
    candidates: tuple[str, ...]
    scale: float = 1.0  # e.g. 0.01 to turn a percentage into a fraction


@dataclass(frozen=True, slots=True)
class MetricMap:
    running_requests: MetricSpec | None = None
    waiting_requests: MetricSpec | None = None
    kv_cache_usage: MetricSpec | None = None
    prefix_cache_queries: MetricSpec | None = None
    prefix_cache_hits: MetricSpec | None = None
    prefix_cache_hit_rate: MetricSpec | None = None
    preemptions_total: MetricSpec | None = None
    prompt_tokens_total: MetricSpec | None = None
    generation_tokens_total: MetricSpec | None = None
    spec_decode_accepted_tokens: MetricSpec | None = None
    spec_decode_draft_tokens: MetricSpec | None = None
    spec_decode_acceptance_rate: MetricSpec | None = None
    #: Extra samples copied verbatim into ``BackendMetrics.raw`` when present.
    raw_keep_prefixes: tuple[str, ...] = field(default_factory=tuple)


def _lookup(samples: Mapping[str, float], spec: MetricSpec | None) -> float | None:
    if spec is None:
        return None
    for name in spec.candidates:
        if name in samples:
            return samples[name] * spec.scale
    return None


def build_metrics(samples: Mapping[str, float], mapping: MetricMap, source: str) -> BackendMetrics:
    raw = {
        k: v
        for k, v in samples.items()
        if mapping.raw_keep_prefixes and k.startswith(mapping.raw_keep_prefixes)
    }
    return BackendMetrics(
        source=source,
        running_requests=_lookup(samples, mapping.running_requests),
        waiting_requests=_lookup(samples, mapping.waiting_requests),
        kv_cache_usage=_lookup(samples, mapping.kv_cache_usage),
        prefix_cache_queries=_lookup(samples, mapping.prefix_cache_queries),
        prefix_cache_hits=_lookup(samples, mapping.prefix_cache_hits),
        prefix_cache_hit_rate=_lookup(samples, mapping.prefix_cache_hit_rate),
        preemptions_total=_lookup(samples, mapping.preemptions_total),
        prompt_tokens_total=_lookup(samples, mapping.prompt_tokens_total),
        generation_tokens_total=_lookup(samples, mapping.generation_tokens_total),
        spec_decode_accepted_tokens=_lookup(samples, mapping.spec_decode_accepted_tokens),
        spec_decode_draft_tokens=_lookup(samples, mapping.spec_decode_draft_tokens),
        spec_decode_acceptance_rate_reported=_lookup(samples, mapping.spec_decode_acceptance_rate),
        raw=raw,
    )


#: Metrics exported by InferenceLab's own serving app (``inferencelab backend serve``).
INFERENCELAB_SERVER_METRICS = MetricMap(
    running_requests=MetricSpec(("inferencelab_engine_running_requests",)),
    waiting_requests=MetricSpec(("inferencelab_engine_waiting_requests",)),
    kv_cache_usage=MetricSpec(("inferencelab_engine_kv_cache_usage_ratio",)),
    prefix_cache_queries=MetricSpec(("inferencelab_engine_prefix_cache_queries_total",)),
    prefix_cache_hits=MetricSpec(("inferencelab_engine_prefix_cache_hits_total",)),
    preemptions_total=MetricSpec(("inferencelab_engine_preemptions_total",)),
    prompt_tokens_total=MetricSpec(("inferencelab_engine_prompt_tokens_total",)),
    generation_tokens_total=MetricSpec(("inferencelab_engine_generation_tokens_total",)),
)
