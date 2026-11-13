"""Server-Sent Events encoding and robust parsing of OpenAI-style streams.

The parser follows the SSE spec closely enough for LLM servers: ``data:`` lines are
accumulated until a blank line dispatches the event, comment lines (``:``) and other
fields are ignored, and ``data: [DONE]`` terminates the stream. Anything that is not
valid JSON or lacks the expected structure raises
:class:`~inferencelab.domain.errors.MalformedResponseError` instead of being guessed at.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from inferencelab.domain.errors import (
    BackendHTTPError,
    ErrorKind,
    InferenceError,
    MalformedResponseError,
)
from inferencelab.domain.response import FinishReason, TokenCountSource, TokenUsage

__all__ = [
    "DONE",
    "ParsedChunk",
    "encode_done",
    "encode_event",
    "iter_sse_data",
    "parse_stream_payload",
    "parse_usage",
]

DONE = "[DONE]"


def encode_event(data: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n\n"


def encode_done() -> bytes:
    return b"data: [DONE]\n\n"


async def iter_sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield the ``data`` payload of each SSE event from an async line iterator."""
    buffer: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if buffer:
                yield "\n".join(buffer)
                buffer.clear()
            continue
        if line.startswith(":"):
            continue  # comment / keep-alive
        field, sep, value = line.partition(":")
        if not sep:
            continue  # a field with no value; ignored per spec
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            buffer.append(value)
    if buffer:  # stream ended without a trailing blank line
        yield "\n".join(buffer)


@dataclass(frozen=True, slots=True)
class ParsedChunk:
    text: str = ""
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None
    done: bool = False


def parse_usage(raw: Any) -> TokenUsage | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise MalformedResponseError("usage must be an object")
    try:
        prompt = int(raw["prompt_tokens"])
        completion = int(raw.get("completion_tokens") or 0)
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedResponseError(f"invalid usage block: {exc}") from exc
    cached: int | None = None
    details = raw.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        try:
            cached = int(details["cached_tokens"])
        except (TypeError, ValueError):
            cached = None
    return TokenUsage(prompt, completion, TokenCountSource.REPORTED, cached_input_tokens=cached)


def parse_stream_payload(payload: str, *, chat: bool, backend: str | None = None) -> ParsedChunk:
    """Interpret one SSE ``data`` payload of a chat/completions stream."""
    if payload.strip() == DONE:
        return ParsedChunk(done=True)
    try:
        body = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(f"invalid JSON in stream: {exc.msg}", backend=backend) from exc
    if not isinstance(body, dict):
        raise MalformedResponseError("stream chunk is not a JSON object", backend=backend)
    if "error" in body:
        err = body["error"]
        message = err.get("message", "stream error") if isinstance(err, dict) else str(err)
        code = err.get("code") if isinstance(err, dict) else None
        if isinstance(code, str) and code in ErrorKind.__members__.values():
            # An InferenceLab gateway upstream: preserve its classification.
            raise InferenceError(f"error event in stream: {message}", kind=ErrorKind(code))
        status = code if isinstance(code, int) else 500
        raise BackendHTTPError(status, f"error event in stream: {message}", backend=backend)
    usage = parse_usage(body.get("usage"))
    choices = body.get("choices")
    if choices is None:
        if usage is None:
            raise MalformedResponseError(
                "stream chunk has neither choices nor usage", backend=backend
            )
        return ParsedChunk(usage=usage)
    if not isinstance(choices, list):
        raise MalformedResponseError("choices must be a list", backend=backend)
    if not choices:
        return ParsedChunk(usage=usage)
    choice = choices[0]
    if not isinstance(choice, dict):
        raise MalformedResponseError("choice must be an object", backend=backend)
    if chat:
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise MalformedResponseError("delta must be an object", backend=backend)
        text = delta.get("content") or ""
    else:
        text = choice.get("text") or ""
    if not isinstance(text, str):
        raise MalformedResponseError("content must be a string", backend=backend)
    return ParsedChunk(
        text=text, finish_reason=FinishReason.parse(choice.get("finish_reason")), usage=usage
    )
