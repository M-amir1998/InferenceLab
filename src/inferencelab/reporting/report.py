"""Per-run ``report.md``: everything a reviewer needs to judge one benchmark run.

Sections that were not measured say so explicitly ("not measured") instead of being
omitted or filled with guesses. Simulated runs carry a banner on the first line.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from inferencelab.metrics.summary import SIMULATED_LABEL, RunSummary
from inferencelab.reporting.tables import fmt, latency_table

__all__ = ["render_run_report"]

# SLO status labels (not credentials)
_STATUS = {"pass": "PASS", "fail": "FAIL", "insufficient_data": "INSUFFICIENT DATA"}  # nosec B105


def _kv(rows: Sequence[tuple[str, str]]) -> str:
    lines = ["| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def _banner(summary: RunSummary) -> str:
    if summary.label == SIMULATED_LABEL:
        return (
            "> **SIMULATED BACKEND RESULTS** — produced on CPU by InferenceLab's mock/simulated "
            "backends. They exercise the serving stack (routing, admission, streaming, metrics); "
            "they are **not** GPU inference measurements and say nothing about vLLM, SGLang or "
            "TensorRT-LLM performance."
        )
    if summary.simulated is False:
        return (
            "> **GPU INFERENCE RESULTS** — measured against a real serving engine (see Hardware)."
        )
    return "> **UNVERIFIED BACKEND TYPE** — the target did not identify itself; label results accordingly."


def _section_dict(title: str, data: Mapping[str, Any] | None, empty: str) -> str:
    if not data:
        return f"## {title}\n\n{empty}\n"
    rows = [(str(k), "n/a" if v is None else str(v)) for k, v in data.items()]
    return f"## {title}\n\n" + _kv(rows) + "\n"


def render_run_report(
    summary: RunSummary,
    metadata: Mapping[str, Any],
    *,
    plots: Sequence[Path] = (),
    run_dir: Path | None = None,
) -> str:
    s = summary
    lat = s.latency
    parts: list[str] = [
        f"# {metadata.get('name', s.run_id)} — benchmark report",
        "",
        _banner(s),
        "",
        f"Run `{s.run_id}` · {metadata.get('created_at', '')} · "
        f"InferenceLab {metadata.get('inferencelab_version', '?')}"
        + (
            f" · git `{metadata['git']['commit'][:12]}`"
            if metadata.get("git", {}).get("commit")
            else ""
        ),
        "",
        "## Summary",
        "",
    ]
    slo_status = _STATUS[s.slo.status] if s.slo else "no SLO configured"
    parts.append(
        _kv(
            [
                ("Requests (measured / total)", f"{s.counts.measured:,} / {s.counts.total:,}"),
                ("Successful", f"{s.counts.ok:,} of {s.errors.attempted:,} attempted"),
                ("Measurement duration", fmt(s.throughput.duration_s, 2, " s")),
                ("Request throughput", fmt(s.throughput.request_throughput, 2, " req/s")),
                ("Output token throughput", fmt(s.throughput.output_token_throughput, 1, " tok/s")),
                (
                    "Goodput",
                    fmt(s.throughput.goodput_rps, 2, " req/s") if s.slo else "no SLO configured",
                ),
                (
                    "TTFT p50 / p95",
                    f"{fmt(lat['ttft_ms'].p50, 1)} / {fmt(lat['ttft_ms'].p95, 1)} ms",
                ),
                (
                    "TPOT p50 / p95",
                    f"{fmt(lat['tpot_ms'].p50, 2)} / {fmt(lat['tpot_ms'].p95, 2)} ms",
                ),
                ("SLO", slo_status),
                (
                    "Token counts",
                    "engine-reported" if not s.tokens.any_estimated else "partly ESTIMATED",
                ),
            ]
        )
    )
    parts += [
        "",
        _section_dict(
            "Deployment", metadata.get("deployment"), "No deployment metadata (local target)."
        ),
        _section_dict(
            "Hardware", metadata.get("hardware"), "Local CPU run; no GPU hardware involved."
        ),
        _section_dict("Workload", metadata.get("workload"), "No workload metadata."),
    ]
    # SLO
    parts.append("## SLO\n")
    if s.slo is None:
        parts.append("No SLO configured for this run; goodput is not computed.\n")
    else:
        rows = [
            "| Objective | Measured | Samples (needed) | Status |",
            "|---|---|---|---|",
        ]
        rows.extend(
            f"| {o.label} | {fmt(o.measured_ms, 1, ' ms')} | {o.samples:,} ({o.required_samples:,}) "
            f"| {_STATUS[o.status]} |"
            for o in s.slo.objectives
        )
        if s.slo.availability_target is not None:
            measured = s.slo.availability_measured
            rows.append(
                f"| availability >= {s.slo.availability_target:g} | "
                f"{'n/a' if measured is None else f'{measured:.4f}'} | {s.errors.attempted:,} | "
                f"{_STATUS[s.slo.availability_status or 'insufficient_data']} |"
            )
        parts.append("\n".join(rows))
        attainment = s.slo.attainment
        parts.append(
            f"\nPer-request thresholds {s.slo.thresholds}; SLO attainment "
            f"{'n/a' if attainment is None else f'{attainment * 100:.1f}%'} of eligible requests. "
            "These are measurements of this run, not guarantees.\n"
        )
    parts += ["## Latency\n", latency_table(s), ""]
    parts += [
        "## Throughput & goodput\n",
        _kv(
            [
                ("Duration (first send → last token)", fmt(s.throughput.duration_s, 3, " s")),
                ("Requests/s (successful)", fmt(s.throughput.request_throughput, 3)),
                ("Input tokens/s", fmt(s.throughput.input_token_throughput, 1)),
                ("Output tokens/s", fmt(s.throughput.output_token_throughput, 1)),
                ("Total tokens/s", fmt(s.throughput.total_token_throughput, 1)),
                ("Goodput (SLO-compliant req/s)", fmt(s.throughput.goodput_rps, 3)),
                ("Goodput output tokens/s", fmt(s.throughput.goodput_output_token_throughput, 1)),
            ]
        ),
        "",
    ]
    e = s.errors
    parts += [
        "## Errors\n",
        _kv(
            [
                (
                    "Status counts",
                    f"ok {s.counts.ok}, error {s.counts.error}, timeout {s.counts.timeout}, "
                    f"rejected {s.counts.rejected}, cancelled {s.counts.cancelled}, dropped {s.counts.dropped}",
                ),
                ("Error rate", f"{e.error_rate * 100:.2f}%"),
                ("Timeout rate", f"{e.timeout_rate * 100:.2f}%"),
                ("Backend error rate", f"{e.backend_error_rate * 100:.2f}%"),
                ("Invalid response rate", f"{e.invalid_response_rate * 100:.2f}%"),
                ("Queue rejection rate", f"{e.queue_rejection_rate * 100:.2f}%"),
                ("Rate-limit rejection rate", f"{e.rate_limit_rate * 100:.2f}%"),
                ("Circuit-breaker rejection rate", f"{e.circuit_rejection_rate * 100:.2f}%"),
                ("Cancelled (by workload)", f"{e.cancelled_rate * 100:.2f}%"),
                ("By kind", ", ".join(f"{k}: {v}" for k, v in e.by_kind.items()) or "none"),
                (
                    "By HTTP status",
                    ", ".join(f"{k}: {v}" for k, v in e.by_http_status.items()) or "none",
                ),
            ]
        ),
        "",
    ]
    t = s.tokens
    parts += [
        "## Tokens\n",
        _kv(
            [
                (
                    "Input / output / total",
                    f"{t.input_tokens:,} / {t.output_tokens:,} / {t.total_tokens:,}",
                ),
                (
                    "Mean input / output per request",
                    f"{fmt(t.mean_input_tokens, 1)} / {fmt(t.mean_output_tokens, 1)}",
                ),
                (
                    "Source",
                    f"{t.reported_requests} engine-reported, {t.estimated_requests} ESTIMATED",
                ),
                (
                    "Prefix-cache hits reported per request",
                    "not reported"
                    if t.cached_input_ratio is None
                    else f"{t.cached_input_ratio * 100:.1f}% of input tokens",
                ),
            ]
        ),
        "",
    ]
    if s.per_backend:
        rows = [
            "| Backend | Requests | Share | OK | TTFT p50 (ms) | TTFT p95 (ms) | Output tokens |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, b in s.per_backend.items():
            rows.append(
                f"| {name} | {b.requests:,} | {b.share * 100:.1f}% | {b.ok:,} | {fmt(b.ttft_ms.p50, 1)} "
                f"| {fmt(b.ttft_ms.p95, 1)} | {b.output_tokens:,} |"
            )
        parts += ["## Per-backend breakdown\n", "\n".join(rows), ""]
    parts.append(
        _section_dict(
            "Engine-reported metrics",
            {k: round(v, 6) for k, v in s.engine.items()} if s.engine else None,
            "The target did not expose engine metrics for this run.",
        )
    )
    parts.append(_section_dict("Cost", s.cost, "No hardware price configured; cost not computed."))
    parts.append(
        _section_dict(
            "Memory", metadata.get("gpu_memory"), "GPU memory not measured (no GPU in this run)."
        )
    )
    parts.append(
        _section_dict(
            "Quality validation",
            metadata.get("quality"),
            "Not run for this benchmark (see `inferencelab quality compare`).",
        )
    )
    if s.warnings:
        parts += ["## Warnings\n", "\n".join(f"- {w}" for w in s.warnings), ""]
    if plots:
        parts.append("## Plots\n")
        for p in plots:
            rel = p.relative_to(run_dir) if run_dir is not None and p.is_relative_to(run_dir) else p
            parts.append(f"![{p.stem}]({rel.as_posix()})\n")
    parts += [
        "## Methodology\n",
        "TTFT = first content chunk − send; E2E = last content chunk − send; TPOT = (E2E − TTFT)/(output "
        "tokens − 1); ITL = gaps between content chunks; throughput counts successful requests over the "
        "makespan of the measurement window; percentiles are withheld when fewer than 5 samples lie "
        "beyond them. See docs/methodology/.",
        "",
    ]
    return "\n".join(parts)
