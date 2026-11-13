"""Turn a stream of :class:`BackendChunk` into OpenAI-compatible SSE bytes.

Also provides :class:`ClosingStreamingResponse`: Starlette abandons the body iterator
when the client disconnects (it stops iterating but never calls ``aclose()``), leaving
cleanup to garbage collection. This subclass always closes the iterator — shielded from
anyio's level-triggered cancellation — so backend connections and engine slots are
released immediately.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Literal

import anyio
from starlette.responses import StreamingResponse
from starlette.types import Send

from inferencelab.backends.base import BackendChunk
from inferencelab.domain.errors import InferenceError
from inferencelab.domain.response import FinishReason, TokenUsage
from inferencelab.openai_api.schemas import chat_chunk, completion_chunk, error_body
from inferencelab.openai_api.sse import encode_done, encode_event

__all__ = ["ClosingStreamingResponse", "openai_sse"]


class ClosingStreamingResponse(StreamingResponse):
    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                with anyio.CancelScope(shield=True):
                    await aclose()


async def openai_sse(
    chunks: AsyncIterator[BackendChunk],
    *,
    chat: bool,
    response_id: str,
    model: str,
    include_usage: bool,
    fallback_usage: Callable[[], TokenUsage] | None = None,
    errors: Literal["event", "raise"] = "event",
) -> AsyncGenerator[bytes, None]:
    """Format chunks as SSE. Mid-stream failures become an error event + ``[DONE]``
    (``errors="event"``) because headers have already been sent; no retry is possible."""
    created = int(time.time())
    usage: TokenUsage | None = None
    finish: FinishReason | None = None
    try:
        if chat:
            yield encode_event(
                chat_chunk(
                    response_id=response_id, model=model, created=created, content="", role=True
                )
            )
        async for chunk in chunks:
            usage = chunk.usage or usage
            finish = chunk.finish_reason or finish
            if not chunk.text:
                continue
            if chat:
                body = chat_chunk(
                    response_id=response_id, model=model, created=created, content=chunk.text
                )
            else:
                body = completion_chunk(
                    response_id=response_id, model=model, created=created, text=chunk.text
                )
            yield encode_event(body)
        final_reason = finish or FinishReason.STOP
        if chat:
            yield encode_event(
                chat_chunk(
                    response_id=response_id,
                    model=model,
                    created=created,
                    finish_reason=final_reason,
                )
            )
        else:
            yield encode_event(
                completion_chunk(
                    response_id=response_id,
                    model=model,
                    created=created,
                    finish_reason=final_reason,
                )
            )
        if include_usage:
            final_usage = usage or (fallback_usage() if fallback_usage is not None else None)
            if final_usage is not None:
                maker = chat_chunk if chat else completion_chunk
                yield encode_event(
                    maker(
                        response_id=response_id,
                        model=model,
                        created=created,
                        usage=final_usage,
                        empty_choices=True,
                    )
                )
        yield encode_done()
    except InferenceError as err:
        if errors == "raise":
            raise
        yield encode_event(error_body(err.message, kind=err.kind.value, status=err.http_status))
        yield encode_done()
