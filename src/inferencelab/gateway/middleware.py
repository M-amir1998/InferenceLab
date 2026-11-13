"""Pure-ASGI middleware: request body size limit (before any JSON parsing)."""

from __future__ import annotations

import json
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = ["BodySizeLimitMiddleware"]


class BodySizeLimitMiddleware:
    """Reject bodies larger than ``max_bytes`` with 413, honouring chunked uploads too."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await self._reject(send)
                        return
                except ValueError:
                    await self._reject(send, status=400, message="invalid content-length")
                    return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()  # later calls observe client disconnects

        await self.app(scope, replay, send)

    async def _reject(self, send: Send, *, status: int = 413, message: str | None = None) -> None:
        text = message or f"request body exceeds {self.max_bytes} bytes"
        payload: dict[str, Any] = {
            "error": {"message": text, "type": "invalid_request_error", "code": "invalid_request"}
        }
        data = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(data)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": data})
