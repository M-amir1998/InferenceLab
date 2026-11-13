"""Cloud-neutral description of a GPU host reachable over SSH.

No provider SDKs, no credentials: authentication is whatever the user's SSH setup
provides (agent, ``~/.ssh/config``, or an ``identity_file`` *path*). Key material is never
stored in configs. Secrets the engines need (``HF_TOKEN``) live in an env file *on the
host* that the user creates; InferenceLab only references its path.
"""

from __future__ import annotations

import re

from pydantic import Field, field_validator

from inferencelab.schema import StrictModel

__all__ = ["RemoteHost", "expand_remote_path"]

_HOST = re.compile(r"^[A-Za-z0-9._:\-\[\]]+$")
_USER = re.compile(r"^[a-z_][a-z0-9_.-]{0,31}$")


class RemoteHost(StrictModel):
    name: str
    hostname: str
    user: str
    port: int = Field(default=22, ge=1, le=65535)
    identity_file: str | None = Field(
        default=None, description="Path to a private key file (never its content)"
    )
    ssh_options: list[str] = Field(
        default_factory=list, description="Extra -o options, e.g. ServerAliveInterval=30"
    )
    gpu_profile: str | None = None
    workdir: str = "~/inferencelab"
    hf_cache: str = "~/.cache/huggingface"
    env_file: str = "~/.inferencelab/env"
    startup_timeout_s: float = Field(default=1800.0, gt=0)

    @field_validator("hostname")
    @classmethod
    def _hostname(cls, value: str) -> str:
        if not _HOST.match(value):
            raise ValueError("hostname contains unsupported characters")
        return value

    @field_validator("user")
    @classmethod
    def _user(cls, value: str) -> str:
        if not _USER.match(value):
            raise ValueError("user must be a plain POSIX user name")
        return value

    @field_validator("ssh_options")
    @classmethod
    def _options(cls, value: list[str]) -> list[str]:
        for opt in value:
            key = opt.split("=", 1)[0].strip().lower()
            if key in {"proxycommand", "localcommand", "permitlocalcommand"}:
                raise ValueError(f"ssh option {key!r} is not allowed (command execution)")
        return value

    @property
    def target(self) -> str:
        return f"{self.user}@{self.hostname}"


def expand_remote_path(path: str, home: str) -> str:
    """Expand a leading ``~`` using the remote ``$HOME`` (arguments are shell-quoted)."""
    if path == "~":
        return home
    if path.startswith("~/"):
        return home.rstrip("/") + path[1:]
    return path
