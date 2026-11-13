"""SGLang adapter: OpenAI-compatible HTTP + SGLang's Prometheus metric names (v0.5.23).

Metrics require the server flag ``--enable-metrics``; the launch builder always sets it.
"""

from __future__ import annotations

from typing import ClassVar

from inferencelab.backends.base import BackendCapabilities
from inferencelab.backends.openai_compatible.backend import OpenAICompatibleBackend
from inferencelab.backends.openai_compatible.prom import MetricMap, MetricSpec

__all__ = ["SGLANG_METRICS", "SGLangBackend"]

SGLANG_METRICS = MetricMap(
    running_requests=MetricSpec(("sglang:num_running_reqs",)),
    waiting_requests=MetricSpec(("sglang:num_queue_reqs",)),
    kv_cache_usage=MetricSpec(("sglang:token_usage",)),
    prefix_cache_hits=MetricSpec(("sglang:cached_tokens_total",)),
    prefix_cache_queries=MetricSpec(("sglang:prompt_tokens_total",)),
    prefix_cache_hit_rate=MetricSpec(("sglang:cache_hit_rate",)),
    prompt_tokens_total=MetricSpec(("sglang:prompt_tokens_total",)),
    generation_tokens_total=MetricSpec(("sglang:generation_tokens_total",)),
    spec_decode_acceptance_rate=MetricSpec(("sglang:spec_accept_rate",)),
    raw_keep_prefixes=(
        "sglang:time_to_first_token_seconds",
        "sglang:inter_token_latency_seconds",
        "sglang:e2e_request_latency_seconds",
        "sglang:queue_time_seconds",
        "sglang:spec_accept_length",
        "sglang:gen_throughput",
    ),
)


class SGLangBackend(OpenAICompatibleBackend):
    kind: ClassVar[str] = "sglang"
    metric_map: ClassVar[MetricMap] = SGLANG_METRICS
    phase_metrics: ClassVar[dict[str, str]] = {
        "queue": "sglang:queue_time_seconds",
        "ttft": "sglang:time_to_first_token_seconds",
        "itl": "sglang:inter_token_latency_seconds",
        "e2e": "sglang:e2e_request_latency_seconds",
    }
    # /server_info is current; /get_server_info is the deprecated alias.
    version_paths: ClassVar[tuple[str, ...]] = ("/server_info", "/get_server_info")

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
