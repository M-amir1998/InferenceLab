"""Shared test configuration and helpers.

Timing-dependent logic is tested under :class:`~inferencelab.clock.VirtualClock`, so no
test sleeps on wall-clock time to observe behaviour.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

from inferencelab.backends.base import BackendRequest
from inferencelab.clock import VirtualClock
from inferencelab.domain.request import (
    ChatMessage,
    InferenceRequest,
    Priority,
    RequestKind,
    RoutingHints,
    SamplingParams,
)

settings.register_profile("default", max_examples=60, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def vclock() -> VirtualClock:
    return VirtualClock()


def chat_request(
    request_id: str = "r1",
    *,
    model: str = "demo-model",
    text: str = "hello world",
    max_tokens: int = 8,
    stream: bool = False,
    priority: Priority = Priority.NORMAL,
    hints: RoutingHints | None = None,
    messages: Sequence[ChatMessage] | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        model=model,
        kind=RequestKind.CHAT,
        sampling=SamplingParams(max_tokens=max_tokens, temperature=0.0),
        messages=tuple(messages) if messages is not None else (ChatMessage("user", text),),
        stream=stream,
        include_usage=True,
        priority=priority,
        hints=hints or RoutingHints(),
    )


def backend_request(
    request_id: str = "r1",
    *,
    words: int = 10,
    max_tokens: int = 8,
    model: str = "demo-model",
    messages: Sequence[ChatMessage] | None = None,
) -> BackendRequest:
    return BackendRequest(
        request_id=request_id,
        model=model,
        kind=RequestKind.CHAT,
        sampling=SamplingParams(max_tokens=max_tokens, temperature=0.0),
        messages=tuple(messages)
        if messages is not None
        else (ChatMessage("user", " ".join(["data"] * words)),),
    )
