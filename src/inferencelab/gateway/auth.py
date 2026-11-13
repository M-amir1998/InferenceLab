"""Optional API-key authentication.

Keys come only from an environment variable (``tenant:key`` pairs, comma separated) and
are compared in constant time against every configured key. When auth is enabled the
tenant identity comes from the key, so clients cannot spoof ``x-tenant-id``.
This is deliberately minimal — not an identity-management system.
"""

from __future__ import annotations

import hmac
import os

from inferencelab.domain.errors import AuthenticationError

__all__ = ["ApiKeyAuth", "parse_api_keys"]


def parse_api_keys(raw: str) -> dict[str, str]:
    """``"teamA:key1,teamB:key2"`` -> ``{key: tenant}``. A bare key maps to 'default'."""
    keys: dict[str, str] = {}
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        tenant, sep, key = item.partition(":")
        if not sep:
            tenant, key = "default", item
        if len(key) < 16:
            raise ValueError("API keys must be at least 16 characters")
        keys[key] = tenant
    return keys


class ApiKeyAuth:
    def __init__(self, keys: dict[str, str]) -> None:
        if not keys:
            raise ValueError("auth is enabled but no API keys are configured")
        self._keys = [(k.encode(), t) for k, t in keys.items()]

    @classmethod
    def from_env(cls, env_var: str) -> ApiKeyAuth:
        raw = os.environ.get(env_var, "")
        return cls(parse_api_keys(raw))

    def authenticate(self, authorization: str | None, api_key_header: str | None) -> str:
        """Return the tenant id for a valid key or raise :class:`AuthenticationError`."""
        presented: str | None = None
        if authorization and authorization.lower().startswith("bearer "):
            presented = authorization[7:].strip()
        elif api_key_header:
            presented = api_key_header.strip()
        if not presented:
            raise AuthenticationError("missing API key")
        candidate = presented.encode()
        tenant: str | None = None
        for key, owner in self._keys:  # no early exit: constant work per request
            if hmac.compare_digest(candidate, key):
                tenant = owner
        if tenant is None:
            raise AuthenticationError("invalid API key")
        return tenant
