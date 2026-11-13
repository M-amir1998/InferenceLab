"""Turn a :class:`WorkloadSpec` into a deterministic stream of sessions.

Every workload is a sequence of *sessions*; single-turn workloads simply have one turn
per session. Generation is lazy (duration-bounded closed-loop runs never need to know
the total upfront) and fully determined by ``spec.seed``: separate RNG streams are
derived for lengths, arrivals, prefixes and assignment, so changing e.g. the arrival
rate does not change the prompts.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from inferencelab.domain.workload import ClosedLoopArrival, TraceArrival, WorkloadSpec
from inferencelab.vocabulary import VOCABULARY
from inferencelab.workloads.arrivals import arrival_offsets
from inferencelab.workloads.traces import TraceEntry, load_trace

__all__ = ["SessionPlan", "TurnPlan", "WorkloadPlan", "synthetic_text"]


def synthetic_text(rng: random.Random, n_tokens: int) -> str:
    """``n_tokens`` words drawn from the shared vocabulary (≈ n_tokens BPE tokens)."""
    return " ".join(rng.choices(VOCABULARY, k=max(1, n_tokens)))


def _stream(seed: int, name: str) -> random.Random:
    digest = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


@dataclass(frozen=True, slots=True)
class TurnPlan:
    user_text: str
    input_tokens: int  # requested (target) input tokens for this turn's new text
    max_tokens: int
    cancel_after_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class SessionPlan:
    index: int
    start_offset_s: float | None  # None for closed-loop sessions
    turns: tuple[TurnPlan, ...]
    system_prompt: str | None = None
    prefix_group: int | None = None
    session_id: str | None = None  # also used as conversation_id for affinity
    prefix_key: str | None = None
    tenant_id: str | None = None
    priority: str = "normal"
    think_time_s: float = 0.0


class WorkloadPlan:
    def __init__(self, spec: WorkloadSpec, *, base_dir: Path | None = None) -> None:
        self.spec = spec
        self._trace: list[TraceEntry] | None = None
        if spec.trace is not None:
            path = Path(spec.trace.path)
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            self._trace = load_trace(path, max_entries=spec.num_requests)

    @property
    def closed_loop_concurrency(self) -> int | None:
        arrival = self.spec.arrival
        return arrival.concurrency if isinstance(arrival, ClosedLoopArrival) else None

    @property
    def max_requests(self) -> int | None:
        if self._trace is not None:
            return len(self._trace)
        return self.spec.num_requests

    def _prefixes(self) -> list[str]:
        sp = self.spec.shared_prefix
        if sp is None:
            return []
        rng = _stream(self.spec.seed, "prefix")
        return [synthetic_text(rng, sp.prefix_tokens) for _ in range(sp.num_prefixes)]

    def _offsets(self) -> Iterator[float | None]:
        spec = self.spec
        if isinstance(spec.arrival, ClosedLoopArrival):
            while True:
                yield None
        elif isinstance(spec.arrival, TraceArrival):
            if self._trace is None:  # WorkloadSpec validation guarantees a trace
                raise ValueError("trace arrival requires a trace file")
            scale = spec.trace.time_scale if spec.trace is not None else 1.0
            for entry in self._trace:
                yield None if entry.timestamp_s is None else entry.timestamp_s * scale
        else:
            yield from arrival_offsets(
                spec.arrival, _stream(spec.seed, "arrival"), duration_s=spec.duration_s
            )

    def sessions(self) -> Iterator[SessionPlan]:
        """Sessions in start order, bounded by ``num_requests`` turns or the trace."""
        spec = self.spec
        lengths = _stream(spec.seed, "lengths")
        text_rng = _stream(spec.seed, "text")
        assign = _stream(spec.seed, "assign")
        prefixes = self._prefixes()
        budget = self.max_requests
        emitted_turns = 0
        trace_iter = iter(self._trace) if self._trace is not None else None
        for index, offset in enumerate(self._offsets()):
            if budget is not None and emitted_turns >= budget:
                return
            entry = next(trace_iter, None) if trace_iter is not None else None
            if trace_iter is not None and entry is None:
                return
            n_turns = 1
            if spec.sessions is not None and entry is None:
                n_turns = max(1, spec.sessions.turns.sample(lengths))
            if budget is not None:
                n_turns = min(n_turns, budget - emitted_turns)
            turns = []
            for _ in range(n_turns):
                if entry is not None:
                    n_in, n_out = entry.input_tokens, entry.output_tokens
                else:
                    n_in, n_out = (
                        spec.input_tokens.sample(lengths),
                        spec.output_tokens.sample(lengths),
                    )
                cancel = None
                if spec.cancel_probability and assign.random() < spec.cancel_probability:
                    cancel = min(spec.cancel_after_tokens, n_out)
                if spec.shared_prefix is not None:
                    n_in = max(1, n_in - spec.shared_prefix.prefix_tokens)
                turns.append(TurnPlan(synthetic_text(text_rng, n_in), n_in, n_out, cancel))
            emitted_turns += n_turns
            group = self._pick_group(assign, len(prefixes))
            session_id = None
            if spec.sessions is not None:
                session_id = f"s{index:06d}"
            elif entry is not None and entry.conversation_id:
                session_id = entry.conversation_id
            system = None
            if spec.sessions is not None and spec.sessions.system_prompt_tokens:
                system = synthetic_text(
                    _stream(spec.seed, "system"), spec.sessions.system_prompt_tokens
                )
            yield SessionPlan(
                index=index,
                start_offset_s=offset,
                turns=tuple(turns),
                system_prompt=system,
                prefix_group=group,
                session_id=session_id,
                prefix_key=(entry.prefix_key if entry is not None and entry.prefix_key else None)
                or (f"prefix-{group}" if group is not None else None),
                tenant_id=self._weighted(assign, [(t.tenant_id, t.weight) for t in spec.tenants]),
                priority=self._weighted(assign, [(p.priority, p.weight) for p in spec.priorities])
                or "normal",
                think_time_s=spec.sessions.think_time_s if spec.sessions is not None else 0.0,
            )

    def prefix_text(self, group: int | None) -> str | None:
        if group is None:
            return None
        return self._prefixes()[group]

    def _pick_group(self, rng: random.Random, n: int) -> int | None:
        sp = self.spec.shared_prefix
        if sp is None or n == 0:
            return None
        if sp.popularity == "uniform":
            return rng.randrange(n)
        weights = [1.0 / (k + 1) ** sp.zipf_s for k in range(n)]
        return rng.choices(range(n), weights=weights, k=1)[0]

    @staticmethod
    def _weighted(rng: random.Random, options: list[tuple[str, float]]) -> str | None:
        if not options:
            return None
        names = [name for name, _ in options]
        weights = [weight for _, weight in options]
        return rng.choices(names, weights=weights, k=1)[0]
