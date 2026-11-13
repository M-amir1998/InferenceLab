"""Distribution statistics with sample-size-aware percentiles.

A percentile is only reported when enough samples lie *beyond* it to make it
meaningful: p\\ :sub:`q` requires ``n · (1 − q/100) >= min_tail_samples`` (default 5).
With the default this means p50 needs 10 samples, p90 50, p95 100 and p99 500.
Otherwise the value is ``None`` and reports show "insufficient samples" — a p99 from 40
requests is the max of a tiny sample, not a tail estimate.

Percentiles use linear interpolation between order statistics (NumPy's default
``linear`` method, Hyndman–Fan type 7).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DEFAULT_PERCENTILES",
    "MIN_TAIL_SAMPLES",
    "Distribution",
    "min_samples_for",
    "percentile",
]

DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)
MIN_TAIL_SAMPLES = 5


def min_samples_for(q: float, min_tail_samples: int = MIN_TAIL_SAMPLES) -> int:
    if not 0 < q < 100:
        raise ValueError("percentile must be in (0, 100)")
    return math.ceil(min_tail_samples / (1.0 - q / 100.0) - 1e-9)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0 <= q <= 100:
        raise ValueError("q must be within [0, 100]")
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def _label(q: float) -> str:
    return f"p{q:g}".replace(".", "_")


@dataclass(frozen=True, slots=True)
class Distribution:
    count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    percentiles: dict[str, float | None] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        values: Iterable[float],
        *,
        scale: float = 1.0,
        quantiles: Sequence[float] = DEFAULT_PERCENTILES,
        min_tail_samples: int = MIN_TAIL_SAMPLES,
    ) -> Distribution:
        clean = [float(v) * scale for v in values if math.isfinite(v)]
        n = len(clean)
        if n == 0:
            return cls(count=0, percentiles={_label(q): None for q in quantiles})
        arr = np.asarray(clean, dtype=np.float64)
        pct: dict[str, float | None] = {}
        for q in quantiles:
            if n >= min_samples_for(q, min_tail_samples):
                pct[_label(q)] = float(np.percentile(arr, q, method="linear"))
            else:
                pct[_label(q)] = None
        return cls(
            count=n,
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=1)) if n > 1 else 0.0,
            min=min(clean),
            max=max(clean),
            percentiles=pct,
        )

    def p(self, q: float) -> float | None:
        return self.percentiles.get(_label(q))

    def to_dict(self) -> dict[str, float | int | None]:
        out: dict[str, float | int | None] = {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }
        out.update(self.percentiles)
        return out
