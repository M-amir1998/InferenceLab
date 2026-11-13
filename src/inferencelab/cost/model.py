"""Inference cost modelling from *recorded* prices.

``cost = hourly price × hours`` for the hardware a run occupied. Prices are never
hard-coded: they come from configuration together with their source and observation
date, and every derived figure carries that provenance.

Per-token costs are **all-in**: the whole run cost divided by input (or output, or
total) tokens. Input and output processing share the same GPUs, so the two are not
separable; each figure answers "what did a million of these tokens cost in this run".
"""

from __future__ import annotations

from datetime import date

from inferencelab.domain.hardware import PricingSpec
from inferencelab.schema import MutableModel

__all__ = ["CostReport", "compute_cost", "estimate_cost_upper_bound"]


class CostReport(MutableModel):
    hourly_cost_usd: float
    unit: str
    gpu_count: int
    effective_hourly_cost_usd: float
    price_source: str
    price_recorded_at: date | None
    duration_s: float
    total_cost_usd: float
    cost_per_1m_input_tokens_usd: float | None
    cost_per_1m_output_tokens_usd: float | None
    cost_per_1m_total_tokens_usd: float | None
    cost_per_successful_request_usd: float | None
    cost_per_slo_compliant_request_usd: float | None
    note: str = "per-token figures are all-in (total cost / tokens of that type)"


def _per(total: float, n: int | float | None, scale: float = 1.0) -> float | None:
    if not n:
        return None
    return total / n * scale


def compute_cost(
    pricing: PricingSpec,
    *,
    gpu_count: int,
    duration_s: float,
    input_tokens: int,
    output_tokens: int,
    successful_requests: int,
    slo_compliant_requests: int | None,
) -> CostReport:
    if duration_s < 0:
        raise ValueError("duration must be non-negative")
    hourly = pricing.hourly_cost_for(gpu_count)
    total = hourly * duration_s / 3600.0
    return CostReport(
        hourly_cost_usd=pricing.hourly_cost_usd,
        unit=pricing.unit,
        gpu_count=gpu_count,
        effective_hourly_cost_usd=hourly,
        price_source=pricing.price_source,
        price_recorded_at=pricing.recorded_at,
        duration_s=duration_s,
        total_cost_usd=total,
        cost_per_1m_input_tokens_usd=_per(total, input_tokens, 1e6),
        cost_per_1m_output_tokens_usd=_per(total, output_tokens, 1e6),
        cost_per_1m_total_tokens_usd=_per(total, input_tokens + output_tokens, 1e6),
        cost_per_successful_request_usd=_per(total, successful_requests),
        cost_per_slo_compliant_request_usd=_per(total, slo_compliant_requests),
    )


def estimate_cost_upper_bound(pricing: PricingSpec, *, gpu_count: int, hours: float) -> float:
    """Upper-bound spend for a planned campaign (used by the experiment cost guard)."""
    return pricing.hourly_cost_for(gpu_count) * max(0.0, hours)
