"""Response contract: what InferenceLab records about a completed request.

Values that an engine does not expose stay ``None``. Nothing here is ever fabricated;
estimated token counts are explicitly labelled with :class:`TokenCountSource.ESTIMATED`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "EngineTimings",
    "FinishReason",
    "GatewayTimings",
    "InferenceResponse",
    "TokenCountSource",
    "TokenUsage",
]


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    CANCELLED = "cancelled"

    @classmethod
    def parse(cls, value: str | None) -> FinishReason | None:
        if value is None:
            return None
        try:
            return cls(value)
        except ValueError:
            # Engines occasionally use non-standard reasons ("abort", "eos").
            if value in {"abort", "aborted"}:
                return cls.CANCELLED
            return cls.STOP


class TokenCountSource(StrEnum):
    REPORTED = "reported"  # returned by the serving engine (usage block)
    ESTIMATED = "estimated"  # computed locally by a heuristic; never presented as exact


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    source: TokenCountSource
    #: Prompt tokens served from the engine's prefix cache, when the engine reports it
    #: (OpenAI ``usage.prompt_tokens_details.cached_tokens``).
    cached_input_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class EngineTimings:
    """Per-request phase timings reported *by the engine itself* (seconds)."""

    queue_s: float | None = None
    prefill_s: float | None = None
    decode_s: float | None = None


@dataclass(frozen=True, slots=True)
class GatewayTimings:
    """Latencies observed at the gateway (seconds)."""

    queue_s: float
    ttft_s: float | None
    e2e_s: float


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    request_id: str
    model: str
    backend: str
    deployment: str | None
    text: str
    finish_reason: FinishReason | None
    usage: TokenUsage
    timings: GatewayTimings
    attempts: int
    engine: EngineTimings | None = None
