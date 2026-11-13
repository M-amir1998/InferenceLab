"""Run ASGI apps on loopback ports inside the current event loop.

Used by the CPU demo and the end-to-end tests to stand up real HTTP servers (engine
stand-ins and gateways) without subprocesses. Sockets are bound to 127.0.0.1 on an
ephemeral port before the server starts, so there are no port races.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import Iterator
from types import TracebackType

import uvicorn
from starlette.types import ASGIApp

__all__ = ["LoopbackServer"]


class _EmbeddedServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        # Embedded servers must not hijack the host process's SIGINT/SIGTERM handlers.
        yield


class LoopbackServer:
    def __init__(self, app: ASGIApp, *, log_level: str = "warning") -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self.port: int = self._sock.getsockname()[1]
        config = uvicorn.Config(
            app,
            log_level=log_level,
            access_log=False,
            lifespan="on",
            timeout_graceful_shutdown=2,
        )
        self._server = _EmbeddedServer(config)
        self._task: asyncio.Task[None] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(
            self._server.serve(sockets=[self._sock])
        )
        for _ in range(500):
            if self._server.started:
                return
            if self._task.done():
                self._task.result()  # surface startup errors
            await asyncio.sleep(0.01)
        raise RuntimeError("loopback server did not start within 5 s")

    async def stop(self) -> None:
        self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._sock.close()

    async def __aenter__(self) -> LoopbackServer:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()
