"""OpenAI API payloads + headers -> internal :class:`InferenceRequest`."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping

from inferencelab.domain.errors import RequestValidationError
from inferencelab.domain.request import (
    ChatMessage,
    InferenceRequest,
    Priority,
    RequestKind,
    RoutingHints,
    SamplingParams,
)
from inferencelab.gateway.config import RequestLimitsConfig
from inferencelab.openai_api.schemas import ChatCompletionRequest, CompletionRequest

__all__ = ["HEADER_PREFIX", "chat_request", "completion_request", "request_id_from"]

HEADER_PREFIX = "x-inferencelab-"
_ID = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")


def _header_id(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None or value == "":
        return None
    if not _ID.match(value):
        raise RequestValidationError(f"header {name} must match [A-Za-z0-9._:-]{{1,128}}")
    return value


def request_id_from(headers: Mapping[str, str]) -> str:
    supplied = headers.get("x-request-id")
    if supplied and _ID.match(supplied):
        return supplied
    return f"req-{uuid.uuid4().hex}"


def _hints(headers: Mapping[str, str], tenant: str | None) -> RoutingHints:
    return RoutingHints(
        tenant_id=tenant if tenant is not None else _header_id(headers, "x-tenant-id"),
        conversation_id=_header_id(headers, "x-conversation-id"),
        prefix_key=_header_id(headers, "x-prefix-key"),
    )


def _max_tokens(requested: int | None, limits: RequestLimitsConfig) -> int:
    value = requested if requested is not None else limits.default_max_tokens
    if value > limits.max_tokens_cap:
        raise RequestValidationError(
            f"max_tokens {value} exceeds the cap of {limits.max_tokens_cap}"
        )
    return value


def chat_request(
    body: ChatCompletionRequest,
    *,
    headers: Mapping[str, str],
    request_id: str,
    limits: RequestLimitsConfig,
    tenant: str | None = None,
) -> InferenceRequest:
    if body.n != 1:
        raise RequestValidationError("only n=1 is supported")
    if len(body.messages) > limits.max_messages:
        raise RequestValidationError(f"too many messages (max {limits.max_messages})")
    messages = tuple(
        ChatMessage(role="system" if m.role == "developer" else m.role, content=m.text())
        for m in body.messages
    )
    return InferenceRequest(
        request_id=request_id,
        model=body.model,
        kind=RequestKind.CHAT,
        sampling=SamplingParams(
            max_tokens=_max_tokens(body.requested_max_tokens(), limits),
            temperature=1.0 if body.temperature is None else body.temperature,
            top_p=1.0 if body.top_p is None else body.top_p,
            stop=body.stop_sequences(),
            seed=body.seed,
        ),
        messages=messages,
        stream=body.stream,
        include_usage=bool(body.stream_options and body.stream_options.include_usage),
        priority=Priority.parse(headers.get("x-priority")),
        hints=_hints(headers, tenant),
        extra_body=body.passthrough(),
    )


def completion_request(
    body: CompletionRequest,
    *,
    headers: Mapping[str, str],
    request_id: str,
    limits: RequestLimitsConfig,
    tenant: str | None = None,
) -> InferenceRequest:
    if body.n != 1:
        raise RequestValidationError("only n=1 is supported")
    if body.echo:
        raise RequestValidationError("echo is not supported")
    if isinstance(body.prompt, list):
        if len(body.prompt) != 1:
            raise RequestValidationError("batched prompts are not supported; send one prompt")
        prompt = body.prompt[0]
    else:
        prompt = body.prompt
    return InferenceRequest(
        request_id=request_id,
        model=body.model,
        kind=RequestKind.COMPLETION,
        sampling=SamplingParams(
            max_tokens=_max_tokens(body.max_tokens, limits),
            temperature=1.0 if body.temperature is None else body.temperature,
            top_p=1.0 if body.top_p is None else body.top_p,
            stop=body.stop_sequences(),
            seed=body.seed,
        ),
        prompt=prompt,
        stream=body.stream,
        include_usage=bool(body.stream_options and body.stream_options.include_usage),
        priority=Priority.parse(headers.get("x-priority")),
        hints=_hints(headers, tenant),
        extra_body=body.passthrough(),
    )
