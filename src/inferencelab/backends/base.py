"""Backend abstraction.

An :class:`InferenceBackend` is anything that can generate tokens for a
:class:`BackendRequest`: an in-process mock, the CPU engine simulator, or an HTTP client
for an externally running vLLM / SGLang / TensorRT-LLM server. InferenceLab never
imports those engines; it talks to them over their OpenAI-compatible HTTP APIs
(see docs/adr/ADR-001-http-backend-abstraction.md).

Contract for ``stream()``:

* yields only *content-bearing* chunks plus one final chunk carrying ``finish_reason``
  and usage (if the engine reports it); role-only/empty deltas are swallowed so the
  first yielded chunk marks the first token;
* raises :class:`~inferencelab.domain.errors.InferenceError` subclasses for failures;
* must release all resources (HTTP connection, engine slot) when the consumer stops
  early — closing the generator *is* the cancellation signal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from inferencelab.domain.request import ChatMessage, InferenceRequest, RequestKind, SamplingParams
from inferencelab.domain.response import EngineTimings, FinishReason, TokenCountSource, TokenUsage
from inferencelab.tokens import HeuristicTokenCounter, TokenCounter, estimate_chat_tokens

__all__ = [
    "BackendCapabilities",
    "BackendChunk",
    "BackendHealth",
    "BackendMetrics",
    "BackendRequest",
    "BackendResponse",
    "HealthStatus",
    "InferenceBackend",
    "collect_stream",
]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """What a backend can do. The router and experiment planner never assume more."""

    streaming: bool = True
    chat: bool = True
    completions: bool = True
    usage_in_stream: bool = False
    structured_output: bool = False
    tool_calling: bool = False
    lora: bool = False
    multi_lora: bool = False
    prefix_caching: bool = False
    chunked_prefill: bool = False
    speculative_decoding: bool = False
    quantization: frozenset[str] = frozenset()
    embeddings: bool = False
    distributed_inference: bool = False
    disaggregated_serving: bool = False
    cancellation: bool = False
    engine_metrics: bool = False

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            out[name] = sorted(value) if isinstance(value, frozenset) else value
        return out

    def missing(self, required: set[str]) -> set[str]:
        """Names of required boolean features this backend lacks."""
        return {name for name in required if not getattr(self, name, False)}


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class BackendHealth:
    status: HealthStatus
    latency_s: float | None = None
    detail: str | None = None

    @property
    def healthy(self) -> bool:
        return self.status is not HealthStatus.UNHEALTHY


@dataclass(frozen=True, slots=True)
class BackendMetrics:
    """Engine-reported metrics. Fields the engine does not expose stay ``None``."""

    source: str
    running_requests: float | None = None
    waiting_requests: float | None = None
    kv_cache_usage: float | None = None  # fraction in [0, 1]
    prefix_cache_queries: float | None = None  # cumulative tokens queried
    prefix_cache_hits: float | None = None  # cumulative tokens hit
    prefix_cache_hit_rate: float | None = None  # engine-provided ratio, if it exposes one
    preemptions_total: float | None = None
    prompt_tokens_total: float | None = None
    generation_tokens_total: float | None = None
    spec_decode_accepted_tokens: float | None = None
    spec_decode_draft_tokens: float | None = None
    spec_decode_acceptance_rate_reported: float | None = None  # engine gauge, if any
    raw: Mapping[str, float] = field(default_factory=dict)

    @property
    def derived_prefix_hit_rate(self) -> float | None:
        if self.prefix_cache_hit_rate is not None:
            return self.prefix_cache_hit_rate
        if self.prefix_cache_queries and self.prefix_cache_hits is not None:
            return self.prefix_cache_hits / self.prefix_cache_queries
        return None

    @property
    def spec_decode_acceptance_rate(self) -> float | None:
        if self.spec_decode_draft_tokens and self.spec_decode_accepted_tokens is not None:
            return self.spec_decode_accepted_tokens / self.spec_decode_draft_tokens
        return self.spec_decode_acceptance_rate_reported


@dataclass(frozen=True, slots=True)
class BackendRequest:
    """What a backend needs to serve one generation (no tenant/priority concerns)."""

    request_id: str
    model: str  # model name as served by this backend
    kind: RequestKind
    sampling: SamplingParams
    messages: tuple[ChatMessage, ...] = ()
    prompt: str | None = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    prefix_key: str | None = None

    @classmethod
    def from_inference(cls, request: InferenceRequest, served_model: str) -> BackendRequest:
        return cls(
            request_id=request.request_id,
            model=served_model,
            kind=request.kind,
            sampling=request.sampling,
            messages=request.messages,
            prompt=request.prompt,
            extra_body=request.extra_body,
            prefix_key=request.hints.prefix_key,
        )

    def estimate_input_tokens(self, counter: TokenCounter | None = None) -> int:
        counter = counter or HeuristicTokenCounter()
        if self.kind is RequestKind.CHAT:
            return estimate_chat_tokens(self.messages, counter)
        return counter.count(self.prompt or "")


@dataclass(frozen=True, slots=True)
class BackendChunk:
    text: str
    token_count: int | None = None  # tokens in this chunk, if known
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None
    engine: EngineTimings | None = None

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None or self.usage is not None


@dataclass(frozen=True, slots=True)
class BackendResponse:
    text: str
    finish_reason: FinishReason | None
    usage: TokenUsage
    chunks: int
    engine: EngineTimings | None = None


@runtime_checkable
class InferenceBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    async def health(self) -> BackendHealth: ...

    async def infer(self, request: BackendRequest) -> BackendResponse: ...

    def stream(self, request: BackendRequest) -> AsyncIterator[BackendChunk]: ...

    async def metrics(self) -> BackendMetrics: ...

    async def aclose(self) -> None: ...


async def collect_stream(
    chunks: AsyncIterator[BackendChunk], request: BackendRequest
) -> BackendResponse:
    """Drain a stream into a :class:`BackendResponse`.

    If the engine did not report usage, token counts are *estimated* and labelled so.
    """
    parts: list[str] = []
    finish: FinishReason | None = None
    usage: TokenUsage | None = None
    engine: EngineTimings | None = None
    counted = 0
    n_chunks = 0
    known_counts = True
    async for chunk in chunks:
        n_chunks += 1
        if chunk.text:
            parts.append(chunk.text)
        if chunk.token_count is None:
            known_counts = known_counts and not chunk.text
        else:
            counted += chunk.token_count
        finish = chunk.finish_reason or finish
        usage = chunk.usage or usage
        engine = chunk.engine or engine
    text = "".join(parts)
    if usage is None:
        output = counted if known_counts else HeuristicTokenCounter().count(text)
        usage = TokenUsage(request.estimate_input_tokens(), output, TokenCountSource.ESTIMATED)
    return BackendResponse(
        text=text, finish_reason=finish, usage=usage, chunks=n_chunks, engine=engine
    )
