"""Error taxonomy shared by the data plane, backends and the benchmark client.

Every failure is classified into an :class:`ErrorKind`. The kind decides

* the HTTP status the gateway returns,
* whether the failure is safe to retry *before any output reached the client*,
* whether it counts against a backend's circuit breaker,
* which error-rate bucket a benchmark attributes it to.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "AuthenticationError",
    "BackendHTTPError",
    "BackendTimeoutError",
    "BackendUnavailableError",
    "CircuitOpenError",
    "ErrorKind",
    "InferenceError",
    "LoadShedError",
    "MalformedResponseError",
    "NoHealthyBackendError",
    "QueueFullError",
    "RateLimitedError",
    "RequestCancelledError",
    "RequestValidationError",
    "StreamInterruptedError",
    "TimeoutPhase",
]


class ErrorKind(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"  # tenant quota exhausted
    QUEUE_FULL = "queue_full"  # bounded admission queue overflow (backpressure)
    LOAD_SHED = "load_shed"  # shed by policy (priority / predicted SLO miss)
    NO_HEALTHY_BACKEND = "no_healthy_backend"
    CIRCUIT_OPEN = "circuit_open"
    BACKEND_UNAVAILABLE = "backend_unavailable"  # connection refused / reset before response
    BACKEND_OVERLOADED = "backend_overloaded"  # backend answered 429/503
    BACKEND_ERROR = "backend_error"  # backend answered 5xx
    BACKEND_REJECTED = "backend_rejected"  # backend answered 4xx (request problem)
    TIMEOUT_CONNECT = "timeout_connect"
    TIMEOUT_FIRST_TOKEN = "timeout_first_token"  # noqa: S105 - not a credential  # nosec B105
    TIMEOUT_IDLE = "timeout_idle"
    TIMEOUT_OVERALL = "timeout_overall"
    MALFORMED_RESPONSE = "malformed_response"
    STREAM_INTERRUPTED = "stream_interrupted"  # backend disconnected mid-stream
    CANCELLED = "cancelled"
    INTERNAL = "internal"


class TimeoutPhase(StrEnum):
    CONNECT = "connect"
    FIRST_TOKEN = "first_token"  # noqa: S105 - not a credential  # nosec B105
    IDLE = "idle"
    OVERALL = "overall"


_TIMEOUT_KIND = {
    TimeoutPhase.CONNECT: ErrorKind.TIMEOUT_CONNECT,
    TimeoutPhase.FIRST_TOKEN: ErrorKind.TIMEOUT_FIRST_TOKEN,
    TimeoutPhase.IDLE: ErrorKind.TIMEOUT_IDLE,
    TimeoutPhase.OVERALL: ErrorKind.TIMEOUT_OVERALL,
}

HTTP_STATUS: dict[ErrorKind, int] = {
    ErrorKind.INVALID_REQUEST: 400,
    ErrorKind.UNAUTHORIZED: 401,
    ErrorKind.RATE_LIMITED: 429,
    ErrorKind.QUEUE_FULL: 503,
    ErrorKind.LOAD_SHED: 503,
    ErrorKind.NO_HEALTHY_BACKEND: 503,
    ErrorKind.CIRCUIT_OPEN: 503,
    ErrorKind.BACKEND_UNAVAILABLE: 502,
    ErrorKind.BACKEND_OVERLOADED: 503,
    ErrorKind.BACKEND_ERROR: 502,
    ErrorKind.BACKEND_REJECTED: 400,
    ErrorKind.TIMEOUT_CONNECT: 504,
    ErrorKind.TIMEOUT_FIRST_TOKEN: 504,
    ErrorKind.TIMEOUT_IDLE: 504,
    ErrorKind.TIMEOUT_OVERALL: 504,
    ErrorKind.MALFORMED_RESPONSE: 502,
    ErrorKind.STREAM_INTERRUPTED: 502,
    ErrorKind.CANCELLED: 499,
    ErrorKind.INTERNAL: 500,
}

#: Failures that may be retried on another (or the same) backend *as long as no output
#: has been delivered to the client*. Retrying after output started could duplicate
#: user-visible text, so the data plane never does that for streaming responses.
RETRYABLE_BEFORE_OUTPUT: frozenset[ErrorKind] = frozenset(
    {
        ErrorKind.BACKEND_UNAVAILABLE,
        ErrorKind.BACKEND_OVERLOADED,
        ErrorKind.BACKEND_ERROR,
        ErrorKind.TIMEOUT_CONNECT,
        ErrorKind.TIMEOUT_FIRST_TOKEN,
        ErrorKind.MALFORMED_RESPONSE,
        ErrorKind.STREAM_INTERRUPTED,
        ErrorKind.CIRCUIT_OPEN,
    }
)

#: Failures that indicate backend ill-health and count toward its circuit breaker.
#: Client mistakes (4xx), cancellations and gateway-side rejections do not.
BACKEND_HEALTH_FAILURES: frozenset[ErrorKind] = frozenset(
    {
        ErrorKind.BACKEND_UNAVAILABLE,
        ErrorKind.BACKEND_ERROR,
        ErrorKind.TIMEOUT_CONNECT,
        ErrorKind.TIMEOUT_FIRST_TOKEN,
        ErrorKind.TIMEOUT_IDLE,
        ErrorKind.MALFORMED_RESPONSE,
        ErrorKind.STREAM_INTERRUPTED,
    }
)


class InferenceError(Exception):
    """Base class for all classified failures."""

    default_kind: ErrorKind = ErrorKind.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind | None = None,
        backend: str | None = None,
        http_status: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind: ErrorKind = kind if kind is not None else self.default_kind
        self.backend = backend
        self._http_status = http_status
        self.retry_after_s = retry_after_s

    @property
    def http_status(self) -> int:
        return self._http_status if self._http_status is not None else HTTP_STATUS[self.kind]

    @property
    def retryable(self) -> bool:
        """Safe to retry if (and only if) no output has reached the client yet."""
        return self.kind in RETRYABLE_BEFORE_OUTPUT

    @property
    def counts_against_backend(self) -> bool:
        return self.kind in BACKEND_HEALTH_FAILURES

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kind={self.kind.value!r}, message={self.message!r})"


class RequestValidationError(InferenceError):
    default_kind = ErrorKind.INVALID_REQUEST


class AuthenticationError(InferenceError):
    default_kind = ErrorKind.UNAUTHORIZED


class RateLimitedError(InferenceError):
    default_kind = ErrorKind.RATE_LIMITED


class QueueFullError(InferenceError):
    default_kind = ErrorKind.QUEUE_FULL


class LoadShedError(InferenceError):
    default_kind = ErrorKind.LOAD_SHED


class NoHealthyBackendError(InferenceError):
    default_kind = ErrorKind.NO_HEALTHY_BACKEND


class CircuitOpenError(InferenceError):
    default_kind = ErrorKind.CIRCUIT_OPEN


class BackendUnavailableError(InferenceError):
    default_kind = ErrorKind.BACKEND_UNAVAILABLE


class MalformedResponseError(InferenceError):
    default_kind = ErrorKind.MALFORMED_RESPONSE


class StreamInterruptedError(InferenceError):
    default_kind = ErrorKind.STREAM_INTERRUPTED


class RequestCancelledError(InferenceError):
    default_kind = ErrorKind.CANCELLED


class BackendHTTPError(InferenceError):
    """A backend answered with a non-2xx HTTP status."""

    def __init__(self, status: int, message: str, *, backend: str | None = None) -> None:
        if status in (429, 503):
            kind = ErrorKind.BACKEND_OVERLOADED
        elif status >= 500:
            kind = ErrorKind.BACKEND_ERROR
        else:
            kind = ErrorKind.BACKEND_REJECTED
        # Client errors keep the backend's status; server errors become a 502/503.
        http_status = status if 400 <= status < 500 and status != 429 else None
        super().__init__(message, kind=kind, backend=backend, http_status=http_status)
        self.backend_status = status


class BackendTimeoutError(InferenceError):
    def __init__(self, phase: TimeoutPhase, message: str, *, backend: str | None = None) -> None:
        super().__init__(message, kind=_TIMEOUT_KIND[phase], backend=backend)
        self.phase = phase
