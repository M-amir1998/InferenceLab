"""Repository configs stay valid; config loading, auth, translation and body limits."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pydantic
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from inferencelab.autoscaling.simulator import AutoscaleConfig
from inferencelab.benchmarking.config import BenchmarkConfig
from inferencelab.benchmarking.gpu_monitor import parse_gpu_samples
from inferencelab.benchmarking.system_info import collect_system_info, git_state
from inferencelab.capacity.planner import CapacityPlanConfig
from inferencelab.config.loader import load_backend_config, load_model_file, resolve_refs
from inferencelab.domain.deployment import DeploymentSpec
from inferencelab.domain.errors import AuthenticationError, RequestValidationError
from inferencelab.domain.hardware import HardwareProfile
from inferencelab.domain.model import ModelSpec
from inferencelab.domain.request import Priority
from inferencelab.domain.slo import SLOSpec
from inferencelab.domain.workload import WorkloadSpec
from inferencelab.experiments.spec import ExperimentSpec
from inferencelab.gateway.auth import ApiKeyAuth, parse_api_keys
from inferencelab.gateway.config import GatewayConfig, RequestLimitsConfig
from inferencelab.gateway.middleware import BodySizeLimitMiddleware
from inferencelab.gateway.translate import chat_request, completion_request, request_id_from
from inferencelab.openai_api.schemas import ChatCompletionRequest, CompletionRequest
from inferencelab.remote.host import RemoteHost
from tests.conftest import CONFIGS

SCHEMAS: dict[str, type[pydantic.BaseModel]] = {
    "models": ModelSpec,
    "hardware": HardwareProfile,
    "deployments": DeploymentSpec,
    "workloads": WorkloadSpec,
    "slos": SLOSpec,
    "gateway": GatewayConfig,
    "benchmarks": BenchmarkConfig,
    "experiments": ExperimentSpec,
    "capacity": CapacityPlanConfig,
    "autoscaling": AutoscaleConfig,
    "hosts": RemoteHost,
}


@pytest.mark.parametrize("folder", sorted(SCHEMAS))
def test_every_repository_config_validates(folder: str) -> None:
    files = sorted((CONFIGS / folder).rglob("*.yaml"))
    assert files, f"no configs in {folder}"
    for path in files:
        load_model_file(path, SCHEMAS[folder])


def test_every_backend_config_validates() -> None:
    kinds = {load_backend_config(p).type for p in (CONFIGS / "backends").glob("*.yaml")}
    assert kinds == {"mock", "simulated", "openai", "vllm", "sglang", "tensorrt_llm"}


def test_gpu_deployments_pin_images_and_revisions() -> None:
    models = {
        m.name: m
        for m in (load_model_file(p, ModelSpec) for p in (CONFIGS / "models").glob("*.yaml"))
    }
    for path in (CONFIGS / "deployments").glob("*.yaml"):
        dep = load_model_file(path, DeploymentSpec)
        assert dep.image and ":" in dep.image and not dep.image.endswith(":latest")
        assert models[dep.model].revision_pinned, f"{path.name} uses an unpinned model revision"


def test_refs_resolution_and_cycles(tmp_path: Path) -> None:
    (tmp_path / "w.yaml").write_text("name: w\nnum_requests: 1\n", encoding="utf-8")
    data = resolve_refs({"workload": "w.yaml", "other": {"$ref": "w.yaml"}, "list": [1]}, tmp_path)
    assert data["workload"]["name"] == "w" and data["other"]["num_requests"] == 1
    (tmp_path / "loop.yaml").write_text("$ref: loop.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="deeply"):
        resolve_refs({"$ref": "loop.yaml"}, tmp_path)
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_model_file(tmp_path / "empty.yaml", SLOSpec)


def test_gateway_config_validation() -> None:
    with pytest.raises(pydantic.ValidationError, match="duplicate"):
        GatewayConfig.model_validate(
            {"backends": [{"type": "mock", "name": "a"}, {"type": "mock", "name": "a"}]}
        )
    with pytest.raises(pydantic.ValidationError, match="allowed_backend_hosts"):
        GatewayConfig.model_validate(
            {
                "backends": [{"type": "vllm", "name": "a", "base_url": "http://169.254.169.254"}],
                "security": {"allowed_backend_hosts": ["vllm.internal"]},
            }
        )
    with pytest.raises(pydantic.ValidationError):
        BenchmarkConfig.model_validate(
            {
                "name": "x",
                "virtual_time": True,
                "target": {"type": "http", "base_url": "http://x"},
                "workload": {"name": "w", "num_requests": 1},
            }
        )


# ---------------------------------------------------------------- auth
def test_api_key_parsing_and_constant_time_auth() -> None:
    keys = parse_api_keys("team-a:" + "a" * 16 + ", " + "b" * 20)
    assert keys == {"a" * 16: "team-a", "b" * 20: "default"}
    with pytest.raises(ValueError, match="16"):
        parse_api_keys("t:short")
    auth = ApiKeyAuth(keys)
    assert auth.authenticate("Bearer " + "a" * 16, None) == "team-a"
    assert auth.authenticate(None, "b" * 20) == "default"
    for bad in ((None, None), ("Bearer wrong", None), ("Basic " + "a" * 16, None)):
        with pytest.raises(AuthenticationError):
            auth.authenticate(*bad)
    with pytest.raises(ValueError):
        ApiKeyAuth({})


# ---------------------------------------------------------------- translation
LIMITS = RequestLimitsConfig(max_messages=3, max_tokens_cap=100, default_max_tokens=50)


def test_chat_translation_headers_and_limits() -> None:
    body = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "developer", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    headers = {
        "x-tenant-id": "t1",
        "x-conversation-id": "c-1",
        "x-prefix-key": "doc:7",
        "x-priority": "high",
    }
    req = chat_request(body, headers=headers, request_id="rid", limits=LIMITS)
    assert req.messages[0].role == "system" and req.sampling.max_tokens == 50
    assert req.include_usage and req.priority is Priority.HIGH
    assert (req.hints.tenant_id, req.hints.conversation_id, req.hints.prefix_key) == (
        "t1",
        "c-1",
        "doc:7",
    )
    authed = chat_request(body, headers=headers, request_id="rid", limits=LIMITS, tenant="from-key")
    assert authed.hints.tenant_id == "from-key"  # auth identity beats a spoofable header
    with pytest.raises(RequestValidationError, match="header"):
        chat_request(body, headers={"x-tenant-id": "bad id\n"}, request_id="r", limits=LIMITS)
    too_many = body.model_copy(update={"messages": body.messages * 2})
    with pytest.raises(RequestValidationError, match="too many"):
        chat_request(too_many, headers={}, request_id="r", limits=LIMITS)
    with pytest.raises(RequestValidationError, match="cap"):
        chat_request(
            body.model_copy(update={"max_tokens": 1000}), headers={}, request_id="r", limits=LIMITS
        )
    with pytest.raises(RequestValidationError, match="n=1"):
        chat_request(body.model_copy(update={"n": 2}), headers={}, request_id="r", limits=LIMITS)


def test_completion_translation() -> None:
    single = CompletionRequest.model_validate({"model": "m", "prompt": ["only"]})
    assert completion_request(single, headers={}, request_id="r", limits=LIMITS).prompt == "only"
    with pytest.raises(RequestValidationError, match="batched"):
        completion_request(
            CompletionRequest.model_validate({"model": "m", "prompt": ["a", "b"]}),
            headers={},
            request_id="r",
            limits=LIMITS,
        )
    with pytest.raises(RequestValidationError, match="echo"):
        completion_request(
            CompletionRequest.model_validate({"model": "m", "prompt": "a", "echo": True}),
            headers={},
            request_id="r",
            limits=LIMITS,
        )


def test_request_id_header() -> None:
    assert request_id_from({"x-request-id": "abc-1"}) == "abc-1"
    generated = request_id_from({"x-request-id": "bad id"})
    assert generated.startswith("req-") and len(generated) > 10


# ---------------------------------------------------------------- body size limit
async def test_body_size_middleware() -> None:
    inner = FastAPI()

    @inner.post("/echo")
    async def echo(request: Request) -> JSONResponse:
        return JSONResponse({"n": len(await request.body())})

    app = BodySizeLimitMiddleware(inner, max_bytes=1024)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        assert (await client.post("/echo", content=b"x" * 100)).json() == {"n": 100}
        too_big = await client.post("/echo", content=b"x" * 2048)
        assert too_big.status_code == 413 and too_big.json()["error"]["code"] == "invalid_request"

        async def chunks() -> AsyncIterator[bytes]:
            for _ in range(5):
                yield b"y" * 300

        streamed = await client.post("/echo", content=chunks())
        assert streamed.status_code == 413


# ---------------------------------------------------------------- host facts
def test_system_info_has_no_identifying_paths() -> None:
    info = collect_system_info()
    assert {"os", "python", "cpu_logical", "ram_gb", "inferencelab"} <= set(info)
    blob = str(info).lower()
    assert "users" not in blob and "home" not in blob
    assert isinstance(git_state(), dict)


def test_gpu_sample_parser() -> None:
    rows = parse_gpu_samples("0, 1024, 81920, 55\n1, 2048, 81920, 60\ngarbage\n0, x, 1, 2\n")
    assert rows == [(0, 1024.0, 81920.0, 55.0), (1, 2048.0, 81920.0, 60.0)]
