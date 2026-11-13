"""Per-request benchmark records and the derived latency metrics.

All timestamps are seconds relative to the run start, taken from a monotonic clock on
the *client* side. Definitions (see docs/methodology/latency.md):

* **TTFT**  first content chunk − send time
* **E2E**   last content chunk − send time (the trailing usage-only chunk and
  ``[DONE]`` are excluded, matching ``vllm bench serve``'s E2EL)
* **TPOT**  (E2E − TTFT) / (output_tokens − 1); undefined for single-token outputs
* **ITL**   gaps between consecutive content-bearing chunks (a chunk carrying several
  tokens counts as one gap, so ITL ≠ TPOT under speculative decoding)
* **E2E from schedule**  last content chunk − *intended* send time; includes client-side
  lag, which exposes coordinated omission when the generator falls behind schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from inferencelab.domain.response import TokenCountSource

__all__ = ["Phase", "RequestRecord", "RequestStatus"]


class RequestStatus(StrEnum):
    OK = "ok"
    ERROR = "error"  # backend/gateway failure (5xx, malformed, interrupted)
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"  # cancelled by the client (by design of the workload)
    REJECTED = "rejected"  # admission control: rate limit, queue full, load shed (429/503)
    DROPPED = "dropped"  # never sent: client hit its max_in_flight safety bound


class Phase(StrEnum):
    WARMUP = "warmup"
    MEASURE = "measure"
    COOLDOWN = "cooldown"


@dataclass(frozen=True, slots=True)
class RequestRecord:
    index: int
    request_id: str
    phase: Phase
    status: RequestStatus
    scheduled_at: float
    sent_at: float | None
    first_token_at: float | None
    completed_at: float | None
    chunk_times: tuple[float, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_source: TokenCountSource = TokenCountSource.ESTIMATED
    cached_input_tokens: int | None = None
    requested_input_tokens: int = 0
    requested_output_tokens: int = 0
    backend: str | None = None
    deployment: str | None = None
    attempts: int | None = None
    gateway_queue_s: float | None = None
    engine_queue_s: float | None = None
    engine_prefill_s: float | None = None
    engine_decode_s: float | None = None
    finish_reason: str | None = None
    error_kind: str | None = None
    http_status: int | None = None
    error_message: str | None = None
    session_id: str | None = None
    turn: int = 0
    prefix_group: int | None = None
    tenant_id: str | None = None
    priority: str = "normal"
    prompt_fingerprint: str | None = None
    prompt_text: str | None = None  # only with explicit text-capture opt-in
    output_text: str | None = None  # only with explicit text-capture opt-in

    @property
    def ok(self) -> bool:
        return self.status is RequestStatus.OK

    @property
    def ttft_s(self) -> float | None:
        if self.sent_at is None or self.first_token_at is None:
            return None
        return self.first_token_at - self.sent_at

    @property
    def last_token_at(self) -> float | None:
        return self.chunk_times[-1] if self.chunk_times else self.completed_at

    @property
    def e2e_s(self) -> float | None:
        end = self.last_token_at
        if not self.ok or self.sent_at is None or end is None:
            return None
        return end - self.sent_at

    @property
    def e2e_from_schedule_s(self) -> float | None:
        end = self.last_token_at
        if not self.ok or end is None:
            return None
        return end - self.scheduled_at

    @property
    def schedule_lag_s(self) -> float | None:
        return None if self.sent_at is None else self.sent_at - self.scheduled_at

    @property
    def tpot_s(self) -> float | None:
        ttft, e2e = self.ttft_s, self.e2e_s
        if ttft is None or e2e is None or self.output_tokens is None or self.output_tokens < 2:
            return None
        return max(0.0, e2e - ttft) / (self.output_tokens - 1)

    @property
    def itl_s(self) -> tuple[float, ...]:
        return tuple(b - a for a, b in pairwise(self.chunk_times))
