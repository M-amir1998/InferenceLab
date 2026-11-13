"""Command execution on local or remote hosts.

* :class:`SSHExecutor`     runs ``argv`` on the remote host. The command is quoted with
                           ``shlex.join`` so arguments can never be re-interpreted by the
                           remote shell. ``BatchMode`` forbids interactive prompts.
* :class:`LocalExecutor`   runs on this machine (``inferencelab gpu doctor``).
* :class:`DryRunExecutor`  records commands without running them (default mode for
                           campaigns, and the test double).
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from inferencelab.remote.host import RemoteHost

__all__ = [
    "CommandError",
    "CommandExecutor",
    "CommandResult",
    "DryRunExecutor",
    "LocalExecutor",
    "SSHExecutor",
]


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CommandError(RuntimeError):
    def __init__(self, result: CommandResult) -> None:
        tail = (result.stderr or result.stdout).strip()[-500:]
        super().__init__(f"command failed ({result.returncode}): {shlex.join(result.argv)}\n{tail}")
        self.result = result


class CommandExecutor(Protocol):
    description: str

    async def run(
        self, argv: Sequence[str], *, timeout_s: float = 600.0, check: bool = True
    ) -> CommandResult: ...

    async def put(self, local: Path, remote: str) -> None: ...

    async def get(self, remote: str, local: Path) -> None: ...


async def _exec(argv: Sequence[str], timeout_s: float) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timed out after {timeout_s}s", time.perf_counter() - start
    return (
        proc.returncode if proc.returncode is not None else -1,
        out.decode(errors="replace"),
        err.decode(errors="replace"),
        time.perf_counter() - start,
    )


class SSHExecutor:
    def __init__(self, host: RemoteHost, *, ssh: str = "ssh", scp: str = "scp") -> None:
        self.host = host
        self.description = f"ssh:{host.target}:{host.port}"
        self._ssh = ssh
        self._scp = scp

    def _common(self, port_flag: str) -> list[str]:
        args = [
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=30",
            port_flag, str(self.host.port),
        ]  # fmt: skip
        if self.host.identity_file:
            args += ["-i", self.host.identity_file]
        for opt in self.host.ssh_options:
            args += ["-o", opt]
        return args

    def ssh_argv(self, argv: Sequence[str]) -> list[str]:
        return [self._ssh, *self._common("-p"), self.host.target, "--", shlex.join(argv)]

    async def run(
        self, argv: Sequence[str], *, timeout_s: float = 600.0, check: bool = True
    ) -> CommandResult:
        code, out, err, dur = await _exec(self.ssh_argv(argv), timeout_s)
        result = CommandResult(tuple(argv), code, out, err, dur)
        if check and not result.ok:
            raise CommandError(result)
        return result

    async def put(self, local: Path, remote: str) -> None:
        argv = [self._scp, *self._common("-P"), "-r", str(local), f"{self.host.target}:{remote}"]
        code, out, err, dur = await _exec(argv, 1800.0)
        if code != 0:
            raise CommandError(CommandResult(tuple(argv), code, out, err, dur))

    async def get(self, remote: str, local: Path) -> None:
        await asyncio.to_thread(local.mkdir, parents=True, exist_ok=True)
        argv = [self._scp, *self._common("-P"), "-r", f"{self.host.target}:{remote}", str(local)]
        code, out, err, dur = await _exec(argv, 1800.0)
        if code != 0:
            raise CommandError(CommandResult(tuple(argv), code, out, err, dur))


def _copy(local: Path, dest: Path) -> None:
    if local.is_dir():
        shutil.copytree(local, dest / local.name, dirs_exist_ok=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, dest)


class LocalExecutor:
    description = "local"

    async def run(
        self, argv: Sequence[str], *, timeout_s: float = 600.0, check: bool = True
    ) -> CommandResult:
        if not argv or shutil.which(argv[0]) is None:
            result = CommandResult(
                tuple(argv), 127, "", f"{argv[0] if argv else ''}: not found", 0.0
            )
        else:
            code, out, err, dur = await _exec(argv, timeout_s)
            result = CommandResult(tuple(argv), code, out, err, dur)
        if check and not result.ok:
            raise CommandError(result)
        return result

    async def put(self, local: Path, remote: str) -> None:
        await asyncio.to_thread(_copy, local, Path(remote))

    async def get(self, remote: str, local: Path) -> None:
        await self.put(Path(remote), str(local))


Responder = Callable[[tuple[str, ...]], CommandResult | None]


@dataclass
class DryRunExecutor:
    """Records commands; ``responders`` can script outputs for tests."""

    description: str = "dry-run"
    responders: list[Responder] = field(default_factory=list)
    commands: list[tuple[str, ...]] = field(default_factory=list)
    transfers: list[tuple[str, str, str]] = field(default_factory=list)

    async def run(
        self, argv: Sequence[str], *, timeout_s: float = 600.0, check: bool = True
    ) -> CommandResult:
        key = tuple(argv)
        self.commands.append(key)
        for responder in self.responders:
            result = responder(key)
            if result is not None:
                if check and not result.ok:
                    raise CommandError(result)
                return result
        return CommandResult(key, 0, "", "", 0.0)

    async def put(self, local: Path, remote: str) -> None:
        self.transfers.append(("put", str(local), remote))

    async def get(self, remote: str, local: Path) -> None:
        self.transfers.append(("get", remote, str(local)))
