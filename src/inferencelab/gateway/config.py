"""Gateway configuration: composition of every data-plane component's config."""

from __future__ import annotations

from pydantic import Field, model_validator

from inferencelab.admission.controller import QueueConfig
from inferencelab.backends.config import BackendConfig, HTTPBackendConfig
from inferencelab.domain.slo import SLOSpec
from inferencelab.limits.tenant import TenantLimitsConfig
from inferencelab.observability.tracing import TracingConfig
from inferencelab.reliability.circuit_breaker import CircuitBreakerConfig
from inferencelab.reliability.retry import RetryConfig
from inferencelab.reliability.timeouts import TimeoutConfig
from inferencelab.routing.factory import RoutingConfig
from inferencelab.schema import StrictModel

__all__ = [
    "AuthConfig",
    "GatewayConfig",
    "HealthCheckConfig",
    "ObservabilityConfig",
    "RequestLimitsConfig",
    "SecurityConfig",
    "ServerConfig",
]


class ServerConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=0, le=65535)


class RequestLimitsConfig(StrictModel):
    max_request_bytes: int = Field(default=1_048_576, ge=1024)
    max_messages: int = Field(default=256, ge=1)
    max_tokens_cap: int = Field(default=8192, ge=1)
    default_max_tokens: int = Field(default=512, ge=1)


class AuthConfig(StrictModel):
    enabled: bool = False
    api_keys_env: str = Field(
        default="INFERENCELAB_API_KEYS",
        description="Env var with 'tenant:key,tenant2:key2' (keys never live in config files)",
    )


class SecurityConfig(StrictModel):
    allowed_backend_hosts: list[str] | None = Field(
        default=None, description="If set, every HTTP backend host must be listed (SSRF guard)"
    )


class HealthCheckConfig(StrictModel):
    enabled: bool = True
    interval_s: float = Field(default=5.0, gt=0)
    timeout_s: float = Field(default=2.0, gt=0)
    unhealthy_threshold: int = Field(default=2, ge=1)
    healthy_threshold: int = Field(default=1, ge=1)


class ObservabilityConfig(StrictModel):
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    log_level: str = "info"
    log_json: bool = False


class GatewayConfig(StrictModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    backends: list[BackendConfig] = Field(min_length=1)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    tenants: TenantLimitsConfig | None = None
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    health: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    limits: RequestLimitsConfig = Field(default_factory=RequestLimitsConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    slo: SLOSpec | None = None

    @model_validator(mode="after")
    def _validate(self) -> GatewayConfig:
        names = [b.name for b in self.backends]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate backend names: {dupes}")
        allowed = self.security.allowed_backend_hosts
        if allowed is not None:
            for backend in self.backends:
                if isinstance(backend, HTTPBackendConfig) and backend.host not in allowed:
                    raise ValueError(
                        f"backend {backend.name!r} host {backend.host!r} is not in "
                        "security.allowed_backend_hosts"
                    )
        if self.limits.default_max_tokens > self.limits.max_tokens_cap:
            raise ValueError("limits.default_max_tokens exceeds limits.max_tokens_cap")
        return self
