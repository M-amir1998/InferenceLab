"""Construct backends from declarative configuration."""

from __future__ import annotations

import httpx

from inferencelab.backends.base import InferenceBackend
from inferencelab.backends.config import (
    BackendConfig,
    MockBackendConfig,
    OpenAIBackendConfig,
    SGLangBackendConfig,
    SimulatedBackendConfig,
    TensorRTLLMBackendConfig,
    VLLMBackendConfig,
)
from inferencelab.clock import Clock, SystemClock

__all__ = ["create_backend"]


def create_backend(
    config: BackendConfig,
    *,
    clock: Clock | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> InferenceBackend:
    """Instantiate the backend described by ``config``.

    GPU engines are reached over HTTP only; nothing here imports vLLM, SGLang or
    TensorRT-LLM (ADR-001).
    """
    clock = clock or SystemClock()
    if isinstance(config, MockBackendConfig):
        from inferencelab.backends.mock.backend import MockBackend

        return MockBackend(config, clock=clock)
    if isinstance(config, SimulatedBackendConfig):
        from inferencelab.backends.simulated.backend import SimulatedInferenceBackend

        return SimulatedInferenceBackend(config, clock=clock)
    if isinstance(config, VLLMBackendConfig):
        from inferencelab.backends.vllm.backend import VLLMBackend

        return VLLMBackend(config, client=http_client)
    if isinstance(config, SGLangBackendConfig):
        from inferencelab.backends.sglang.backend import SGLangBackend

        return SGLangBackend(config, client=http_client)
    if isinstance(config, TensorRTLLMBackendConfig):
        from inferencelab.backends.tensorrt_llm.backend import TensorRTLLMBackend

        return TensorRTLLMBackend(config, client=http_client)
    if isinstance(config, OpenAIBackendConfig):
        from inferencelab.backends.openai_compatible.backend import OpenAICompatibleBackend

        return OpenAICompatibleBackend(config, client=http_client)
    raise TypeError(f"unsupported backend config: {type(config).__name__}")  # pragma: no cover
