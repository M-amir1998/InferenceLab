"""Domain contracts: requests, responses, models, deployments, workloads, SLOs, results."""

from inferencelab.domain.deployment import BackendKind, DeploymentRole, DeploymentSpec, Parallelism
from inferencelab.domain.errors import ErrorKind, InferenceError
from inferencelab.domain.hardware import HardwareProfile, HardwareRef, PricingSpec
from inferencelab.domain.model import ModelSpec, Quantization
from inferencelab.domain.request import (
    ChatMessage,
    InferenceRequest,
    Priority,
    RequestKind,
    RoutingHints,
    SamplingParams,
)
from inferencelab.domain.response import (
    EngineTimings,
    FinishReason,
    InferenceResponse,
    TokenCountSource,
    TokenUsage,
)
from inferencelab.domain.result import Phase, RequestRecord, RequestStatus
from inferencelab.domain.slo import RequestThresholds, SLOObjective, SLOSpec
from inferencelab.domain.workload import WorkloadSpec

__all__ = [
    "BackendKind",
    "ChatMessage",
    "DeploymentRole",
    "DeploymentSpec",
    "EngineTimings",
    "ErrorKind",
    "FinishReason",
    "HardwareProfile",
    "HardwareRef",
    "InferenceError",
    "InferenceRequest",
    "InferenceResponse",
    "ModelSpec",
    "Parallelism",
    "Phase",
    "PricingSpec",
    "Priority",
    "Quantization",
    "RequestKind",
    "RequestRecord",
    "RequestStatus",
    "RequestThresholds",
    "RoutingHints",
    "SLOObjective",
    "SLOSpec",
    "SamplingParams",
    "TokenCountSource",
    "TokenUsage",
    "WorkloadSpec",
]
