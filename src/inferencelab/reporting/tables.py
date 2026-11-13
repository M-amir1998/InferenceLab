"""Markdown tables built only from measured summaries ("n/a" when not measured)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inferencelab.metrics.summary import RunSummary

__all__ = ["RunRow", "comparison_table", "fmt", "latency_table", "prefix_cache_table"]


def fmt(value: float | int | None, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}{unit}"
    if abs(value) >= 1000:
        return f"{value:,.0f}{unit}"
    return f"{value:.{digits}f}{unit}"


@dataclass(frozen=True, slots=True)
class RunRow:
    """One run as it appears in comparison tables."""

    backend: str
    config: str
    summary: RunSummary
    gpu_memory_gb: float | None = None
    prefix_caching: bool | None = None


def _md(rows: Sequence[Sequence[str]], header: Sequence[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def comparison_table(rows: Sequence[RunRow]) -> str:
    """| Backend | Config | TTFT p95 | TPOT p95 | Req/s | Tok/s | Goodput | GPU Mem | Cost |"""
    out = []
    for r in rows:
        s = r.summary
        cost = None
        if s.cost and isinstance(s.cost.get("cost_per_1m_output_tokens_usd"), float):
            cost = f"${s.cost['cost_per_1m_output_tokens_usd']:.3f}/1M out"
        out.append(
            [
                r.backend,
                r.config,
                fmt(s.latency["ttft_ms"].p95, 1, " ms"),
                fmt(s.latency["tpot_ms"].p95, 2, " ms"),
                fmt(s.throughput.request_throughput, 2),
                fmt(s.throughput.output_token_throughput, 0),
                fmt(s.throughput.goodput_rps, 2),
                fmt(r.gpu_memory_gb, 1, " GB"),
                cost or "n/a",
            ]
        )
    return _md(
        out,
        [
            "Backend",
            "Config",
            "TTFT p95",
            "TPOT p95",
            "Req/s",
            "Tok/s",
            "Goodput",
            "GPU Mem",
            "Cost",
        ],
    )


def prefix_cache_table(rows: Sequence[RunRow]) -> str:
    """| Backend | Cache | TTFT | Throughput | Hit Ratio | (hit ratio as reported by the engine)."""
    out = []
    for r in rows:
        s = r.summary
        hit = s.engine.get("prefix_cache_hit_rate")
        if hit is None:
            hit = s.tokens.cached_input_ratio
        cache = "n/a" if r.prefix_caching is None else ("on" if r.prefix_caching else "off")
        out.append(
            [
                r.backend,
                cache,
                f"p50 {fmt(s.latency['ttft_ms'].p50, 1)} / p95 {fmt(s.latency['ttft_ms'].p95, 1)} ms",
                f"{fmt(s.throughput.output_token_throughput, 0)} tok/s",
                "n/a" if hit is None else f"{hit * 100:.1f}%",
            ]
        )
    return _md(out, ["Backend", "Cache", "TTFT", "Throughput", "Hit Ratio"])


def latency_table(summary: RunSummary) -> str:
    rows = []
    names = {
        "ttft_ms": "TTFT",
        "tpot_ms": "TPOT",
        "itl_ms": "ITL",
        "e2e_ms": "E2E",
        "e2e_from_schedule_ms": "E2E from schedule",
        "gateway_queue_ms": "Gateway queue",
        "engine_queue_ms": "Engine queue",
        "engine_prefill_ms": "Engine prefill",
        "engine_decode_ms": "Engine decode",
        "schedule_lag_ms": "Client schedule lag",
    }
    for key, label in names.items():
        d = summary.latency.get(key)
        if d is None or d.count == 0:
            continue
        rows.append(
            [
                label,
                f"{d.count:,}",
                fmt(d.mean, 2),
                fmt(d.p50, 2),
                fmt(d.p90, 2),
                fmt(d.p95, 2),
                fmt(d.p99, 2),
                fmt(d.max, 2),
            ]
        )
    return _md(rows, ["Metric (ms)", "n", "mean", "p50", "p90", "p95", "p99", "max"])
