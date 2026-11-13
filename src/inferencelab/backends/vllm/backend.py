"""vLLM adapter: OpenAI-compatible HTTP + vLLM's Prometheus metric names.

Metric names follow vLLM's V1 engine (v0.33.0). Pre-v0.11 names are kept as fallback
candidates (``gpu_cache_usage_perc``, ``gpu_prefix_cache_*``) so older servers still map.
"""

from __future__ import annotations

from typing import ClassVar

from inferencelab.backends.base import BackendCapabilities
from inferencelab.backends.openai_compatible.backend import OpenAICompatibleBackend
from inferencelab.backends.openai_compatible.prom import MetricMap, MetricSpec

__all__ = ["VLLM_METRICS", "VLLMBackend"]

VLLM_METRICS = MetricMap(
    running_requests=MetricSpec(("vllm:num_requests_running",)),
    waiting_requests=MetricSpec(("vllm:num_requests_waiting",)),
    kv_cache_usage=MetricSpec(("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")),
    prefix_cache_queries=MetricSpec(
        ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total")
    ),
    prefix_cache_hits=MetricSpec(
        ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total")
    ),
    preemptions_total=MetricSpec(("vllm:num_preemptions_total",)),
    prompt_tokens_total=MetricSpec(("vllm:prompt_tokens_total",)),
    generation_tokens_total=MetricSpec(("vllm:generation_tokens_total",)),
    spec_decode_accepted_tokens=MetricSpec(("vllm:spec_decode_num_accepted_tokens_total",)),
    spec_decode_draft_tokens=MetricSpec(("vllm:spec_decode_num_draft_tokens_total",)),
    raw_keep_prefixes=(
        "vllm:time_to_first_token_seconds",
        "vllm:inter_token_latency_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:request_queue_time_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_decode_time_seconds",
    ),
)


class VLLMBackend(OpenAICompatibleBackend):
    kind: ClassVar[str] = "vllm"
    metric_map: ClassVar[MetricMap] = VLLM_METRICS
    phase_metrics: ClassVar[dict[str, str]] = {
        "queue": "vllm:request_queue_time_seconds",
        "prefill": "vllm:request_prefill_time_seconds",
        "decode": "vllm:request_decode_time_seconds",
        "ttft": "vllm:time_to_first_token_seconds",
        "itl": "vllm:inter_token_latency_seconds",
        "e2e": "vllm:e2e_request_latency_seconds",
    }
    version_paths: ClassVar[tuple[str, ...]] = ("/version",)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            usage_in_stream=self.config.request_usage_in_stream,
            structured_output=True,
            tool_calling=True,
            lora=True,
            multi_lora=True,
            prefix_caching=True,
            chunked_prefill=True,
            speculative_decoding=True,
            quantization=frozenset({"bf16", "fp16", "fp8", "awq", "gptq", "int8"}),
            embeddings=True,
            distributed_inference=True,
            disaggregated_serving=True,
            cancellation=True,
            engine_metrics=self.config.metrics_path is not None,
        )
