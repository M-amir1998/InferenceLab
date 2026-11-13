from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pydantic
import pytest

from inferencelab.domain.errors import (
    BackendHTTPError,
    ErrorKind,
    InferenceError,
    MalformedResponseError,
)
from inferencelab.domain.response import FinishReason, TokenCountSource, TokenUsage
from inferencelab.openai_api.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    chat_chunk,
    chat_completion,
    error_body,
    models_list,
)
from inferencelab.openai_api.sse import (
    encode_done,
    encode_event,
    iter_sse_data,
    parse_stream_payload,
)


async def _lines(*lines: str) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def _collect(*lines: str) -> list[str]:
    return [p async for p in iter_sse_data(_lines(*lines))]


async def test_sse_parser_handles_comments_multiline_and_crlf() -> None:
    payloads = await _collect(
        ": keep-alive",
        "event: message",
        "data: first",
        "",
        "data: line1\r\n",
        "data: line2",
        "",
        "data:no-space",
        "",
        "data: [DONE]",
    )
    assert payloads == ["first", "line1\nline2", "no-space", "[DONE]"]


async def test_sse_ignores_fields_without_values_and_empty_events() -> None:
    assert await _collect("retry", "", "", "data: x", "") == ["x"]


def test_parse_chat_delta_finish_and_usage() -> None:
    p = parse_stream_payload(
        json.dumps({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}), chat=True
    )
    assert p.text == "hi" and p.finish_reason is None
    done = parse_stream_payload("[DONE]", chat=True)
    assert done.done
    usage = parse_stream_payload(
        json.dumps(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
            }
        ),
        chat=True,
    )
    assert usage.usage == TokenUsage(5, 3, TokenCountSource.REPORTED, cached_input_tokens=2)


def test_parse_completion_text_and_role_only_delta() -> None:
    assert (
        parse_stream_payload(json.dumps({"choices": [{"text": "abc"}]}), chat=False).text == "abc"
    )
    role = parse_stream_payload(
        json.dumps({"choices": [{"delta": {"role": "assistant"}}]}), chat=True
    )
    assert role.text == "" and role.finish_reason is None
    fin = parse_stream_payload(
        json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}), chat=True
    )
    assert fin.finish_reason is FinishReason.LENGTH


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        "[1, 2]",
        json.dumps({"id": "x"}),
        json.dumps({"choices": "nope"}),
        json.dumps({"choices": [{"delta": "x"}]}),
        json.dumps({"choices": [{"delta": {"content": 5}}]}),
        json.dumps({"usage": {"completion_tokens": 3}}),
    ],
)
def test_malformed_payloads_raise(payload: str) -> None:
    with pytest.raises(MalformedResponseError):
        parse_stream_payload(payload, chat=True)


def test_error_events_are_classified() -> None:
    with pytest.raises(BackendHTTPError) as exc:
        parse_stream_payload(json.dumps({"error": {"message": "boom", "code": 500}}), chat=True)
    assert exc.value.backend_status == 500
    with pytest.raises(InferenceError) as gw:
        parse_stream_payload(
            json.dumps({"error": {"message": "full", "code": "queue_full"}}), chat=True
        )
    assert gw.value.kind is ErrorKind.QUEUE_FULL


def test_encoders() -> None:
    assert encode_event({"a": 1}) == b'data: {"a":1}\n\n'
    assert encode_done() == b"data: [DONE]\n\n"


def test_chat_request_schema_passthrough_and_text_parts() -> None:
    body = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                }
            ],
            "max_completion_tokens": 7,
            "ignore_eos": True,
            "response_format": {"type": "json_object"},
            "unknown_field": "ignored",
        }
    )
    assert body.messages[0].text() == "ab"
    assert body.requested_max_tokens() == 7
    assert body.passthrough() == {"response_format": {"type": "json_object"}, "ignore_eos": True}
    with pytest.raises(pydantic.ValidationError):
        ChatCompletionRequest.model_validate({"model": "m", "messages": []})


def test_completion_request_defaults() -> None:
    body = CompletionRequest.model_validate({"model": "m", "prompt": "x", "stop": "\n"})
    assert body.max_tokens == 16 and body.stop_sequences() == ("\n",)


def test_response_builders() -> None:
    usage = TokenUsage(3, 2, TokenCountSource.REPORTED)
    body = chat_completion(
        response_id="id", model="m", text="hi", finish_reason=FinishReason.CANCELLED, usage=usage
    )
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    chunk = chat_chunk(response_id="id", model="m", created=1, usage=usage, empty_choices=True)
    assert chunk["choices"] == [] and chunk["usage"]["total_tokens"] == 5
    assert error_body("x", kind="queue_full", status=503)["error"]["type"] == "server_error"
    assert [m["id"] for m in models_list(["b", "a", "a"])["data"]] == ["a", "b"]
