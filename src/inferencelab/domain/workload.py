"""Typed benchmark workload definitions.

A workload fixes everything about the *traffic*: prompt/output length distributions,
arrival process, shared prefixes, multi-turn sessions, tenants and priorities. Two
backends are only comparable under the same workload (see docs/methodology).
"""

from __future__ import annotations

import random
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from inferencelab.schema import StrictModel

__all__ = [
    "ArrivalPattern",
    "BucketLength",
    "BurstArrival",
    "ClosedLoopArrival",
    "ConstantRateArrival",
    "FixedLength",
    "LengthBucket",
    "LengthDistribution",
    "PoissonArrival",
    "PriorityShare",
    "SessionSpec",
    "SharedPrefixSpec",
    "StepArrival",
    "StepStage",
    "TenantShare",
    "TraceArrival",
    "TraceSpec",
    "UniformLength",
    "WarmupSpec",
    "WorkloadSpec",
]


# ---------------------------------------------------------------- length distributions
class FixedLength(StrictModel):
    kind: Literal["fixed"] = "fixed"
    value: int = Field(ge=1)

    def sample(self, rng: random.Random) -> int:
        return self.value

    @property
    def mean(self) -> float:
        return float(self.value)


class UniformLength(StrictModel):
    kind: Literal["uniform"] = "uniform"
    min: int = Field(ge=1)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> UniformLength:
        if self.min > self.max:
            raise ValueError("uniform: min must be <= max")
        return self

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.min, self.max)

    @property
    def mean(self) -> float:
        return (self.min + self.max) / 2.0


class LengthBucket(StrictModel):
    min: int = Field(ge=1)
    max: int = Field(ge=1)
    weight: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> LengthBucket:
        if self.min > self.max:
            raise ValueError("bucket: min must be <= max")
        return self


class BucketLength(StrictModel):
    """Weighted buckets, e.g. 70% short chat prompts and 30% long RAG prompts."""

    kind: Literal["buckets"] = "buckets"
    buckets: list[LengthBucket] = Field(min_length=1)

    def sample(self, rng: random.Random) -> int:
        bucket = rng.choices(self.buckets, weights=[b.weight for b in self.buckets], k=1)[0]
        return rng.randint(bucket.min, bucket.max)

    @property
    def mean(self) -> float:
        total = sum(b.weight for b in self.buckets)
        return sum(b.weight * (b.min + b.max) / 2.0 for b in self.buckets) / total


LengthDistribution = Annotated[
    FixedLength | UniformLength | BucketLength, Field(discriminator="kind")
]


# ---------------------------------------------------------------- arrival processes
class ClosedLoopArrival(StrictModel):
    """N concurrent users; each sends its next request when the previous one ends."""

    kind: Literal["closed_loop"] = "closed_loop"
    concurrency: int = Field(ge=1)


class ConstantRateArrival(StrictModel):
    kind: Literal["constant"] = "constant"
    rate: float = Field(gt=0, description="requests (sessions) per second")


class PoissonArrival(StrictModel):
    kind: Literal["poisson"] = "poisson"
    rate: float = Field(gt=0, description="mean requests (sessions) per second")


class BurstArrival(StrictModel):
    kind: Literal["burst"] = "burst"
    burst_size: int = Field(ge=1)
    interval_s: float = Field(gt=0)


class StepStage(StrictModel):
    rate: float = Field(gt=0)
    duration_s: float = Field(gt=0)


class StepArrival(StrictModel):
    """Piecewise-constant load (step test), optionally Poisson within each stage."""

    kind: Literal["step"] = "step"
    stages: list[StepStage] = Field(min_length=1)
    poisson: bool = False


class TraceArrival(StrictModel):
    """Replay arrival timestamps recorded in the workload trace."""

    kind: Literal["trace"] = "trace"


ArrivalPattern = Annotated[
    ClosedLoopArrival
    | ConstantRateArrival
    | PoissonArrival
    | BurstArrival
    | StepArrival
    | TraceArrival,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------- workload features
class SharedPrefixSpec(StrictModel):
    """Requests share one of ``num_prefixes`` long common prefixes (system prompt/RAG context)."""

    num_prefixes: int = Field(ge=1)
    prefix_tokens: int = Field(ge=1)
    popularity: Literal["uniform", "zipf"] = "uniform"
    zipf_s: float = Field(default=1.1, gt=0)


class SessionSpec(StrictModel):
    """Multi-turn chat: each session carries its history forward turn by turn."""

    turns: LengthDistribution
    think_time_s: float = Field(default=0.0, ge=0)
    system_prompt_tokens: int = Field(default=0, ge=0)


class TraceSpec(StrictModel):
    """JSONL trace of request *shapes* (lengths, optional timestamps/keys), never prompts."""

    path: str
    time_scale: float = Field(default=1.0, gt=0)


class WarmupSpec(StrictModel):
    requests: int = Field(default=0, ge=0)
    seconds: float = Field(default=0.0, ge=0)


class TenantShare(StrictModel):
    tenant_id: str
    weight: float = Field(gt=0)


class PriorityShare(StrictModel):
    priority: Literal["low", "normal", "high"]
    weight: float = Field(gt=0)


class WorkloadSpec(StrictModel):
    name: str
    description: str = ""
    model: str = Field(default="default", description="Model name sent in each request")
    api: Literal["chat", "completions"] = "chat"
    num_requests: int | None = Field(default=None, ge=1)
    duration_s: float | None = Field(default=None, gt=0)
    input_tokens: LengthDistribution = Field(default_factory=lambda: FixedLength(value=256))
    output_tokens: LengthDistribution = Field(default_factory=lambda: FixedLength(value=128))
    arrival: ArrivalPattern = Field(default_factory=lambda: ClosedLoopArrival(concurrency=1))
    shared_prefix: SharedPrefixSpec | None = None
    sessions: SessionSpec | None = None
    trace: TraceSpec | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    ignore_eos: bool = Field(
        default=True, description="Ask the engine to generate exactly max_tokens (if supported)"
    )
    tenants: list[TenantShare] = Field(default_factory=list)
    priorities: list[PriorityShare] = Field(default_factory=list)
    warmup: WarmupSpec = Field(default_factory=WarmupSpec)
    cooldown: WarmupSpec = Field(default_factory=WarmupSpec)
    seed: int = 0
    max_in_flight: int = Field(default=1024, ge=1, description="Client-side safety bound")
    request_timeout_s: float = Field(default=600.0, gt=0)
    cancel_probability: float = Field(default=0.0, ge=0, le=1)
    cancel_after_tokens: int = Field(default=1, ge=1)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _bounded(self) -> WorkloadSpec:
        if self.num_requests is None and self.duration_s is None and self.trace is None:
            raise ValueError(
                "workload needs num_requests, duration_s or a trace (runs must be bounded)"
            )
        if isinstance(self.arrival, TraceArrival) and self.trace is None:
            raise ValueError("arrival kind 'trace' requires a trace")
        if self.sessions is not None and self.shared_prefix is not None:
            raise ValueError("sessions and shared_prefix are mutually exclusive in v0.1")
        return self

    @property
    def is_open_loop(self) -> bool:
        return not isinstance(self.arrival, ClosedLoopArrival)
