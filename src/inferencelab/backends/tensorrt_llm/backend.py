"""TensorRT-LLM adapter (trtllm-serve, PyTorch backend, v1.2.1).

``trtllm-serve``'s ``/metrics`` returns JSON per-iteration stats that are consumed once
per poll, so this adapter scrapes ``/prometheus/metrics`` instead (Prometheus text, only
present with ``return_perf_metrics: true``). Only metric names verified in the docs are
mapped; everything with the ``trtllm_`` prefix is preserved in ``raw`` for provenance.
"""

from __future__ import annotations

from typing import ClassVar

from inferencelab.backends.base import BackendCapabilities
from inferencelab.backends.openai_compatible.backend import OpenAICompatibleBackend
from inferencelab.backends.openai_compatible.prom import MetricMap, MetricSpec

__all__ = ["TRTLLM_METRICS", "TensorRTLLMBackend"]

TRTLLM_METRICS = MetricMap(
    prefix_cache_hit_rate=MetricSpec(("trtllm_kv_cache_hit_rate",)),
    raw_keep_prefixes=("trtllm_",),
)


class TensorRTLLMBackend(OpenAICompatibleBackend):
    kind: ClassVar[str] = "tensorrt_llm"
    metric_map: ClassVar[MetricMap] = TRTLLM_METRICS
    phase_metrics: ClassVar[dict[str, str]] = {"ttft": "trtllm_time_to_first_token_seconds"}
    version_paths: ClassVar[tuple[str, ...]] = ("/version",)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            usage_in_stream=self.config.request_usage_in_stream,
            structured_output=True,  # requires guided_decoding_backend in extra options
            tool_calling=True,
            lora=True,
            multi_lora=True,
            prefix_caching=True,
            chunked_prefill=True,
            speculative_decoding=True,
            quantization=frozenset({"bf16", "fp16", "fp8", "nvfp4", "awq", "gptq"}),
            distributed_inference=True,
            disaggregated_serving=True,
            cancellation=True,
            engine_metrics=self.config.metrics_path is not None,
        )
