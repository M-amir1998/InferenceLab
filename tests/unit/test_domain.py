from __future__ import annotations

import pydantic
import pytest

from inferencelab.domain.deployment import BackendKind, DeploymentSpec, SpeculativeSpec
from inferencelab.domain.errors import (
    BackendHTTPError,
    BackendTimeoutError,
    ErrorKind,
    QueueFullError,
    RequestValidationError,
    TimeoutPhase,
)
from inferencelab.domain.hardware import HardwareRef
from inferencelab.domain.model import ModelSpec, Quantization
from inferencelab.domain.request import (
    ChatMessage,
    InferenceRequest,
    Priority,
    RequestKind,
    SamplingParams,
)
from inferencelab.domain.response import FinishReason
from inferencelab.domain.result import Phase, RequestRecord, RequestStatus
from inferencelab.domain.slo import RequestThresholds, SLOSpec, is_request_compliant
from inferencelab.domain.workload import WorkloadSpec


# ------------------------------------------------------------------ requests
def test_sampling_params_validation() -> None:
    with pytest.raises(RequestValidationError):
        SamplingParams(max_tokens=0)
    with pytest.raises(RequestValidationError):
        SamplingParams(max_tokens=1, temperature=3.0)
    with pytest.raises(RequestValidationError):
        SamplingParams(max_tokens=1, top_p=0.0)


def test_chat_request_requires_messages_and_no_prompt() -> None:
    sp = SamplingParams(max_tokens=4)
    with pytest.raises(RequestValidationError):
        InferenceRequest(request_id="a", model="m", kind=RequestKind.CHAT, sampling=sp)
    with pytest.raises(RequestValidationError):
        InferenceRequest(
            request_id="a",
            model="m",
            kind=RequestKind.CHAT,
            sampling=sp,
            messages=(ChatMessage("user", "x"),),
            prompt="also",
        )
    with pytest.raises(RequestValidationError):
        InferenceRequest(request_id="a", model="m", kind=RequestKind.COMPLETION, sampling=sp)


def test_priority_parse() -> None:
    assert Priority.parse(None) is Priority.NORMAL
    assert Priority.parse("high") is Priority.HIGH
    assert Priority.parse(" Low ") is Priority.LOW
    assert Priority.parse(7) is Priority.HIGH
    with pytest.raises(RequestValidationError):
        Priority.parse("urgent")


def test_finish_reason_parse_tolerates_engine_variants() -> None:
    assert FinishReason.parse("length") is FinishReason.LENGTH
    assert FinishReason.parse("abort") is FinishReason.CANCELLED
    assert FinishReason.parse("eos") is FinishReason.STOP
    assert FinishReason.parse(None) is None


# ------------------------------------------------------------------ errors
def test_backend_http_error_classification() -> None:
    assert BackendHTTPError(500, "x").kind is ErrorKind.BACKEND_ERROR
    assert BackendHTTPError(503, "x").kind is ErrorKind.BACKEND_OVERLOADED
    assert BackendHTTPError(429, "x").kind is ErrorKind.BACKEND_OVERLOADED
    rejected = BackendHTTPError(400, "bad")
    assert rejected.kind is ErrorKind.BACKEND_REJECTED
    assert rejected.http_status == 400
    assert not rejected.retryable
    assert BackendHTTPError(502, "x").http_status == 502
    assert BackendHTTPError(500, "x").retryable


def test_timeout_error_phases_and_retryability() -> None:
    first = BackendTimeoutError(TimeoutPhase.FIRST_TOKEN, "t")
    assert first.kind is ErrorKind.TIMEOUT_FIRST_TOKEN and first.retryable
    overall = BackendTimeoutError(TimeoutPhase.OVERALL, "t")
    assert not overall.retryable and overall.http_status == 504
    assert QueueFullError("q").http_status == 503
    assert not QueueFullError("q").counts_against_backend


# ------------------------------------------------------------------ models / deployments
def _model(**kw: object) -> ModelSpec:
    base = {"name": "m", "checkpoint": "org/m", "parameter_count_b": 7.0}
    base.update(kw)
    return ModelSpec.model_validate(base)


def test_model_revision_pinning_and_memory() -> None:
    assert not _model().revision_pinned
    assert _model(revision="a" * 40).revision_pinned
    assert _model().weight_memory_gb() == pytest.approx(14.0)
    assert _model().weight_memory_gb(Quantization.AWQ) == pytest.approx(7.0 * 4.25 / 8)
    with pytest.raises(pydantic.ValidationError):
        _model(name="Bad Name")


def _deployment(**kw: object) -> DeploymentSpec:
    base: dict[str, object] = {
        "name": "d",
        "model": "m",
        "backend": "vllm",
        "image": "vllm/vllm-openai:v0.33.0",
        "hardware": {"gpu": "H100-80GB-SXM", "count": 1},
    }
    base.update(kw)
    return DeploymentSpec.model_validate(base)


def test_deployment_rejects_unpinned_images() -> None:
    assert _deployment().image == "vllm/vllm-openai:v0.33.0"
    assert _deployment(image="repo/img@sha256:" + "a" * 64).image
    for bad in ("vllm/vllm-openai:latest", "vllm/vllm-openai", "localhost:5000/img"):
        with pytest.raises(pydantic.ValidationError):
            _deployment(image=bad)


def test_deployment_rejects_secret_env_and_gpu_mismatch() -> None:
    with pytest.raises(pydantic.ValidationError, match="secret"):
        _deployment(env={"HF_TOKEN": "x"})
    with pytest.raises(pydantic.ValidationError, match="GPUs"):
        _deployment(parallelism={"tensor": 2})
    with pytest.raises(pydantic.ValidationError, match="roles"):
        _deployment(backend="mock", role="prefill")
    assert BackendKind.VLLM.is_gpu_engine and BackendKind.MOCK.is_local_simulation


def test_speculative_requires_draft_for_eagle() -> None:
    with pytest.raises(pydantic.ValidationError):
        SpeculativeSpec(method="eagle3")  # type: ignore[arg-type]
    assert SpeculativeSpec(method="ngram").num_speculative_tokens == 4  # type: ignore[arg-type]


def test_hardware_price_requires_source() -> None:
    with pytest.raises(pydantic.ValidationError, match="price_source"):
        HardwareRef(gpu="L4", hourly_cost_usd=1.0)
    ref = HardwareRef(
        gpu="L4", count=2, hourly_cost_usd=1.0, price_source="manual", price_unit="per_gpu_hour"
    )
    assert ref.pricing is not None and ref.pricing.hourly_cost_for(2) == 2.0


# ------------------------------------------------------------------ workloads / SLO
def test_workload_must_be_bounded() -> None:
    with pytest.raises(pydantic.ValidationError, match="bounded"):
        WorkloadSpec(name="w")
    assert WorkloadSpec(name="w", num_requests=1).is_open_loop is False
    with pytest.raises(pydantic.ValidationError):
        WorkloadSpec.model_validate({"name": "w", "num_requests": 1, "typo_field": 1})


def test_slo_compact_syntax_and_thresholds() -> None:
    spec = SLOSpec.model_validate(
        {"ttft_p95_ms": 1000, "ttft_p50_ms": 200, "e2e_p95_ms": 5000, "availability": 0.999}
    )
    assert {o.label for o in spec.objectives} == {
        "TTFT p95 <= 1000 ms",
        "TTFT p50 <= 200 ms",
        "E2E p95 <= 5000 ms",
    }
    thresholds = spec.effective_thresholds()
    assert thresholds.ttft_ms == 1000  # tail objective defines per-request bound
    assert thresholds.e2e_ms == 5000 and thresholds.tpot_ms is None


def test_request_compliance_rules() -> None:
    th = RequestThresholds(ttft_ms=100, tpot_ms=20, e2e_ms=1000)
    assert is_request_compliant(th, ttft_s=0.05, tpot_s=0.01, e2e_s=0.5)
    assert not is_request_compliant(th, ttft_s=0.2, tpot_s=0.01, e2e_s=0.5)
    assert is_request_compliant(th, ttft_s=0.05, tpot_s=None, e2e_s=0.5)  # 1-token output
    assert not is_request_compliant(th, ttft_s=None, tpot_s=None, e2e_s=0.5)


# ------------------------------------------------------------------ records
def test_request_record_metric_definitions() -> None:
    r = RequestRecord(
        index=0,
        request_id="x",
        phase=Phase.MEASURE,
        status=RequestStatus.OK,
        scheduled_at=0.9,
        sent_at=1.0,
        first_token_at=1.2,
        completed_at=2.05,
        chunk_times=(1.2, 1.4, 1.6, 2.0),
        output_tokens=5,
    )
    assert r.ttft_s == pytest.approx(0.2)
    assert r.e2e_s == pytest.approx(1.0)  # last content chunk, not the usage chunk
    assert r.e2e_from_schedule_s == pytest.approx(1.1)
    assert r.schedule_lag_s == pytest.approx(0.1)
    assert r.tpot_s == pytest.approx((1.0 - 0.2) / 4)
    assert r.itl_s == pytest.approx((0.2, 0.2, 0.4))


def test_failed_record_has_no_e2e_or_tpot() -> None:
    r = RequestRecord(
        index=0,
        request_id="x",
        phase=Phase.MEASURE,
        status=RequestStatus.ERROR,
        scheduled_at=0.0,
        sent_at=0.0,
        first_token_at=None,
        completed_at=1.0,
    )
    assert r.e2e_s is None and r.tpot_s is None and r.ttft_s is None
