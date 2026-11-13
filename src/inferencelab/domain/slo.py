"""Service Level Objectives.

Two complementary views:

* **Objectives** are distribution-level targets measured over a run, e.g.
  ``TTFT p95 <= 1000 ms`` or ``availability >= 0.999``. A run either meets them, misses
  them, or has too few samples to say (never "guaranteed").
* **Per-request thresholds** classify each request as SLO-compliant or not. Goodput is
  the rate of successful, compliant requests.

Compact YAML form::

    slo:
      ttft_p95_ms: 1000
      e2e_p95_ms: 5000
      availability: 0.999
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, model_validator

from inferencelab.schema import StrictModel

__all__ = ["RequestThresholds", "SLOMetric", "SLOObjective", "SLOSpec", "is_request_compliant"]

SLOMetric = Literal["ttft", "tpot", "itl", "e2e"]
_COMPACT = re.compile(r"^(ttft|tpot|itl|e2e)_p(\d{1,2}(?:\.\d+)?)_ms$")


class SLOObjective(StrictModel):
    metric: SLOMetric
    percentile: float = Field(gt=0, lt=100)
    threshold_ms: float = Field(gt=0)

    @property
    def label(self) -> str:
        pct = f"{self.percentile:g}"
        return f"{self.metric.upper()} p{pct} <= {self.threshold_ms:g} ms"


class RequestThresholds(StrictModel):
    ttft_ms: float | None = Field(default=None, gt=0)
    tpot_ms: float | None = Field(default=None, gt=0)
    e2e_ms: float | None = Field(default=None, gt=0)

    @property
    def is_empty(self) -> bool:
        return self.ttft_ms is None and self.tpot_ms is None and self.e2e_ms is None


class SLOSpec(StrictModel):
    name: str = "slo"
    objectives: list[SLOObjective] = Field(default_factory=list)
    availability: float | None = Field(default=None, gt=0, le=1)
    request_thresholds: RequestThresholds | None = None

    @model_validator(mode="before")
    @classmethod
    def _expand_compact(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        objectives = list(data.get("objectives", []))
        compact = [(k, m) for k in data if isinstance(k, str) and (m := _COMPACT.match(k))]
        for key, match in compact:
            objectives.append(
                {
                    "metric": match.group(1),
                    "percentile": float(match.group(2)),
                    "threshold_ms": data.pop(key),
                }
            )
        data["objectives"] = objectives
        return data

    def effective_thresholds(self) -> RequestThresholds:
        """Per-request thresholds: explicit, else the tail objective of each metric.

        For every metric, the objective with the highest percentile defines the
        per-request bound (``ttft_p95_ms: 1000`` ⇒ each request needs TTFT <= 1000 ms).
        ITL objectives are distribution-level only and do not classify requests.
        """
        if self.request_thresholds is not None:
            return self.request_thresholds
        tail: dict[str, SLOObjective] = {}
        for obj in self.objectives:
            current = tail.get(obj.metric)
            if current is None or obj.percentile > current.percentile:
                tail[obj.metric] = obj
        return RequestThresholds(
            ttft_ms=tail["ttft"].threshold_ms if "ttft" in tail else None,
            tpot_ms=tail["tpot"].threshold_ms if "tpot" in tail else None,
            e2e_ms=tail["e2e"].threshold_ms if "e2e" in tail else None,
        )


def is_request_compliant(
    thresholds: RequestThresholds,
    *,
    ttft_s: float | None,
    tpot_s: float | None,
    e2e_s: float | None,
) -> bool:
    """Classify one *successful* request against per-request thresholds.

    A threshold on a metric that is undefined for the request (e.g. TPOT of a
    single-token response) is not violated. A required TTFT/E2E that is missing is a
    violation: a request whose latency was not observed cannot be claimed compliant.
    """
    if thresholds.ttft_ms is not None and (ttft_s is None or ttft_s * 1000 > thresholds.ttft_ms):
        return False
    if thresholds.e2e_ms is not None and (e2e_s is None or e2e_s * 1000 > thresholds.e2e_ms):
        return False
    return not (
        thresholds.tpot_ms is not None and tpot_s is not None and tpot_s * 1000 > thresholds.tpot_ms
    )
