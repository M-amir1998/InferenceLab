"""Shared Pydantic base classes for declarative configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["MutableModel", "StrictModel"]


class StrictModel(BaseModel):
    """Immutable config object that rejects unknown keys.

    Rejecting unknown keys turns YAML typos (``prefix_cachng: true``) into validation
    errors instead of silently ignored settings — essential for reproducible runs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class MutableModel(BaseModel):
    """Like :class:`StrictModel` but mutable (used for result documents being built)."""

    model_config = ConfigDict(extra="forbid", validate_default=True)
