"""Internal request contract.

These objects are deliberately independent of the OpenAI wire format and of FastAPI:
the gateway translates API payloads into :class:`InferenceRequest`, and backends receive
a narrower :class:`~inferencelab.backends.base.BackendRequest`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Literal

from inferencelab.domain.errors import RequestValidationError

__all__ = [
    "ChatMessage",
    "InferenceRequest",
    "Priority",
    "RequestKind",
    "RoutingHints",
    "SamplingParams",
]

Role = Literal["system", "user", "assistant", "tool"]


class RequestKind(StrEnum):
    CHAT = "chat"
    COMPLETION = "completion"


class Priority(IntEnum):
    """Request priority used by admission control and load shedding."""

    LOW = 0
    NORMAL = 1
    HIGH = 2

    @classmethod
    def parse(cls, value: str | int | None) -> Priority:
        if value is None or value == "":
            return cls.NORMAL
        if isinstance(value, int):
            return cls(max(cls.LOW, min(cls.HIGH, value)))
        try:
            return cls[value.strip().upper()]
        except KeyError:
            raise RequestValidationError(f"unknown priority {value!r}") from None


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class RoutingHints:
    """Optional affinity metadata. Only identifiers are used; prompt text never is."""

    tenant_id: str | None = None
    conversation_id: str | None = None
    prefix_key: str | None = None


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    stop: tuple[str, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise RequestValidationError("max_tokens must be >= 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise RequestValidationError("temperature must be within [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise RequestValidationError("top_p must be within (0, 1]")


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    request_id: str
    model: str
    kind: RequestKind
    sampling: SamplingParams
    messages: tuple[ChatMessage, ...] = ()
    prompt: str | None = None
    stream: bool = False
    include_usage: bool = False
    priority: Priority = Priority.NORMAL
    hints: RoutingHints = field(default_factory=RoutingHints)
    #: Small string metadata (labels); never prompt content.
    metadata: Mapping[str, str] = field(default_factory=dict)
    #: Engine-specific passthrough parameters (e.g. ``ignore_eos``, ``response_format``).
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise RequestValidationError("request_id must be non-empty")
        if not self.model:
            raise RequestValidationError("model must be non-empty")
        if self.kind is RequestKind.CHAT:
            if not self.messages:
                raise RequestValidationError("chat requests need at least one message")
            if self.prompt is not None:
                raise RequestValidationError("chat requests must not set prompt")
        elif self.prompt is None:
            raise RequestValidationError("completion requests need a prompt")

    def text_for_estimation(self) -> str:
        """Concatenated input text, used only for local token *estimates*."""
        if self.kind is RequestKind.CHAT:
            return "\n".join(m.content for m in self.messages)
        return self.prompt or ""
