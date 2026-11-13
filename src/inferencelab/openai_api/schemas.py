"""OpenAI-compatible request schemas and response builders.

Only the useful serving subset is implemented (see docs/architecture/gateway.md for the
compatibility matrix). Unknown request fields are ignored; a fixed allowlist of engine
sampling extensions is passed through to backends untouched.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from inferencelab.domain.response import FinishReason, TokenUsage

__all__ = [
    "PASSTHROUGH_FIELDS",
    "ChatCompletionRequest",
    "ChatMessageIn",
    "CompletionRequest",
    "StreamOptions",
    "chat_chunk",
    "chat_completion",
    "completion_chunk",
    "completion_response",
    "error_body",
    "models_list",
    "usage_dict",
]

#: Sampling/format extensions understood by vLLM, SGLang and TensorRT-LLM's OpenAI
#: servers. They are forwarded verbatim; InferenceLab does not interpret them.
PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "response_format",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "top_k",
    "min_p",
    "min_tokens",
    "ignore_eos",
)


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TextPart(_Lenient):
    type: Literal["text"]
    text: str


class ChatMessageIn(_Lenient):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[TextPart] | None = None
    name: str | None = None

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content)


class StreamOptions(_Lenient):
    include_usage: bool = False


class _SamplingFields(_Lenient):
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int = Field(default=1, ge=1)
    user: str | None = None
    # passthrough extensions (typed loosely on purpose)
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    logprobs: bool | int | None = None
    top_logprobs: int | None = None
    logit_bias: dict[str, float] | None = None
    top_k: int | None = None
    min_p: float | None = None
    min_tokens: int | None = None
    ignore_eos: bool | None = None

    def stop_sequences(self) -> tuple[str, ...]:
        if self.stop is None:
            return ()
        return (self.stop,) if isinstance(self.stop, str) else tuple(self.stop)

    def passthrough(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in PASSTHROUGH_FIELDS if getattr(self, f) is not None}


class ChatCompletionRequest(_SamplingFields):
    model: str = Field(min_length=1)
    messages: list[ChatMessageIn] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)

    def requested_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class CompletionRequest(_SamplingFields):
    model: str = Field(min_length=1)
    prompt: str | list[str]
    max_tokens: int = Field(default=16, ge=1)  # OpenAI's documented default
    echo: bool = False


# ------------------------------------------------------------------------ responses
def usage_dict(usage: TokenUsage) -> dict[str, Any]:
    body: dict[str, Any] = {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }
    if usage.cached_input_tokens is not None:
        body["prompt_tokens_details"] = {"cached_tokens": usage.cached_input_tokens}
    return body


def _finish(reason: FinishReason | None) -> str | None:
    if reason is None:
        return None
    if reason in (FinishReason.CANCELLED, FinishReason.ERROR):
        return "stop"
    return reason.value


def chat_completion(
    *, response_id: str, model: str, text: str, finish_reason: FinishReason | None,
    usage: TokenUsage, created: int | None = None,
) -> dict[str, Any]:  # fmt: skip
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _finish(finish_reason),
                "logprobs": None,
            }
        ],
        "usage": usage_dict(usage),
    }


def chat_chunk(
    *, response_id: str, model: str, created: int, content: str | None = None,
    role: bool = False, finish_reason: FinishReason | None = None,
    usage: TokenUsage | None = None, empty_choices: bool = False,
) -> dict[str, Any]:  # fmt: skip
    body: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    if empty_choices:
        body["choices"] = []
    else:
        delta: dict[str, Any] = {}
        if role:
            delta["role"] = "assistant"
        if content is not None:
            delta["content"] = content
        body["choices"] = [
            {"index": 0, "delta": delta, "finish_reason": _finish(finish_reason), "logprobs": None}
        ]
    if usage is not None:
        body["usage"] = usage_dict(usage)
    return body


def completion_response(
    *, response_id: str, model: str, text: str, finish_reason: FinishReason | None,
    usage: TokenUsage, created: int | None = None,
) -> dict[str, Any]:  # fmt: skip
    return {
        "id": response_id,
        "object": "text_completion",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "text": text, "finish_reason": _finish(finish_reason), "logprobs": None}
        ],
        "usage": usage_dict(usage),
    }


def completion_chunk(
    *, response_id: str, model: str, created: int, text: str = "",
    finish_reason: FinishReason | None = None, usage: TokenUsage | None = None,
    empty_choices: bool = False,
) -> dict[str, Any]:  # fmt: skip
    body: dict[str, Any] = {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": []
        if empty_choices
        else [
            {"index": 0, "text": text, "finish_reason": _finish(finish_reason), "logprobs": None}
        ],
    }
    if usage is not None:
        body["usage"] = usage_dict(usage)
    return body


def error_body(message: str, *, kind: str, status: int) -> dict[str, Any]:
    """OpenAI-style error envelope. ``code`` carries InferenceLab's error kind."""
    error_type = "invalid_request_error" if 400 <= status < 500 else "server_error"
    return {"error": {"message": message, "type": error_type, "code": kind, "param": None}}


def models_list(model_ids: list[str]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "inferencelab"}
            for m in sorted(set(model_ids))
        ],
    }
