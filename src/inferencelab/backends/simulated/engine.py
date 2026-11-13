"""Deterministic model of a continuous-batching LLM engine (pure, synchronous).

This is a *systems* model for CPU-only testing of routing, admission and benchmarking
— not a GPU performance predictor. It reproduces the qualitative mechanics that drive
real serving behaviour:

* **Continuous batching**: every step schedules decode tokens for running sequences and
  admits waiting sequences as capacity frees up, bounded by ``max_num_seqs`` and a
  per-step token budget ``max_num_batched_tokens``.
* **Paged KV cache**: KV lives in fixed-size blocks; admission needs free blocks.
* **Prefix caching**: full blocks are identified by a chained hash of their token ids
  (as in vLLM's automatic prefix caching). Blocks of finished sequences stay cached
  (evictable, LRU) and are reused by later sequences with the same prefix — including
  multi-turn chats that resend generated text.
* **Chunked prefill**: long prompts are split across steps and co-scheduled with decodes.
  Without it, prefill-only steps stall all decodes (inter-token latency spikes).
* **Preemption by recompute**: when decode growth runs out of KV blocks, the most
  recently admitted sequence is evicted and later re-prefilled (often hitting cache).

Step duration follows a simple linear cost model::

    step = overhead + prefill_tokens * prefill_cost
                    + decode_seqs * decode_cost + attended_kv_tokens * kv_cost
"""

from __future__ import annotations

import math
import zlib
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import Field, model_validator

from inferencelab.schema import StrictModel
from inferencelab.vocabulary import VOCABULARY

__all__ = [
    "BlockManager",
    "Emission",
    "EngineRejectedError",
    "EngineStats",
    "SeqTimings",
    "SimulatedEngine",
    "SimulatedEngineConfig",
    "StepResult",
]


class SimulatedEngineConfig(StrictModel):
    block_size: int = Field(default=16, ge=1)
    num_kv_blocks: int = Field(default=2048, ge=1)
    max_num_seqs: int = Field(default=64, ge=1)
    max_num_batched_tokens: int = Field(default=2048, ge=1)
    max_model_len: int = Field(default=8192, ge=2)
    chunked_prefill: bool = True
    prefix_caching: bool = True
    step_overhead_s: float = Field(default=0.008, ge=0)
    prefill_s_per_token: float = Field(default=0.00004, ge=0)
    decode_s_per_seq: float = Field(default=0.00025, ge=0)
    decode_s_per_kv_token: float = Field(default=1e-7, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> SimulatedEngineConfig:
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError("max_num_batched_tokens must be >= max_num_seqs")
        if not self.chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                "without chunked prefill a whole prompt must fit in one step: "
                "max_num_batched_tokens must be >= max_model_len"
            )
        if self.num_kv_blocks * self.block_size < self.max_model_len:
            raise ValueError("KV cache must hold at least one max_model_len sequence")
        return self

    @property
    def kv_capacity_tokens(self) -> int:
        return self.num_kv_blocks * self.block_size


class EngineRejectedError(ValueError):
    """The engine refuses a request outright (e.g. longer than max_model_len)."""


# --------------------------------------------------------------------- block manager
class BlockManager:
    """Paged KV allocator with hash-addressed, reference-counted prefix blocks.

    Physical blocks are in one of three states: private to one sequence (``_private``
    count), hashed and in use (``_ref[hash] > 0``), or free. Free blocks are either
    uncached (``_free_uncached``) or cached-but-evictable (``_cached_free``, LRU order).
    """

    def __init__(self, num_blocks: int, block_size: int, enable_caching: bool) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_caching = enable_caching
        self._free_uncached = num_blocks
        self._cached_free: OrderedDict[int, None] = OrderedDict()
        self._ref: dict[int, int] = {}
        self._private = 0

    @property
    def free_blocks(self) -> int:
        return self._free_uncached + len(self._cached_free)

    @property
    def used_blocks(self) -> int:
        return len(self._ref) + self._private

    @property
    def usage(self) -> float:
        return self.used_blocks / self.num_blocks

    @property
    def cached_blocks(self) -> int:
        return len(self._cached_free) + len(self._ref)

    def check_invariants(self) -> None:
        """Raise ``RuntimeError`` if block accounting is inconsistent (used by tests)."""
        checks = {
            "block leak": self.used_blocks + self.free_blocks == self.num_blocks,
            "non-positive refcount": all(c > 0 for c in self._ref.values()),
            "block both used and free": not (self._ref.keys() & self._cached_free.keys()),
            "negative free count": self._free_uncached >= 0,
            "negative private count": self._private >= 0,
        }
        broken = [name for name, ok in checks.items() if not ok]
        if broken:
            raise RuntimeError(f"block manager invariants violated: {broken}")

    def cached_prefix(self, hashes: Sequence[int]) -> int:
        """Number of leading full blocks already present in the cache."""
        if not self.enable_caching:
            return 0
        n = 0
        for h in hashes:
            if h in self._ref or h in self._cached_free:
                n += 1
            else:
                break
        return n

    def _take_fresh(self) -> None:
        if self._free_uncached > 0:
            self._free_uncached -= 1
        elif self._cached_free:
            self._cached_free.popitem(last=False)  # evict least recently used
        else:  # pragma: no cover - callers check capacity first
            raise RuntimeError("out of KV blocks")
        self._private += 1

    def allocate(self, hashes: Sequence[int], total_blocks: int) -> tuple[list[int], int] | None:
        """Allocate ``total_blocks`` for a sequence whose full-block hashes are ``hashes``.

        Returns ``(hit_hashes, private_blocks)`` or ``None`` if there is not enough space.
        """
        n_hit = self.cached_prefix(hashes)
        hits = list(hashes[:n_hit])
        reactivated = sum(1 for h in hits if h in self._cached_free)
        fresh = total_blocks - n_hit
        if fresh > self.free_blocks - reactivated:
            return None
        for h in hits:
            if h in self._ref:
                self._ref[h] += 1
            else:
                del self._cached_free[h]
                self._ref[h] = 1
        for _ in range(fresh):
            self._take_fresh()
        return hits, fresh

    def allocate_private(self) -> bool:
        if self.free_blocks == 0:
            return False
        self._take_fresh()
        return True

    def promote(self, block_hash: int) -> None:
        """Turn one of the caller's private blocks into a shareable hashed block."""
        self._private -= 1
        if block_hash in self._ref:  # identical content already resident: dedupe
            self._ref[block_hash] += 1
            self._free_uncached += 1
        elif block_hash in self._cached_free:
            del self._cached_free[block_hash]
            self._ref[block_hash] = 1
            self._free_uncached += 1
        else:
            self._ref[block_hash] = 1

    def free(self, hashes: Sequence[int], private_blocks: int) -> None:
        for h in hashes:
            count = self._ref[h] - 1
            if count:
                self._ref[h] = count
                continue
            del self._ref[h]
            if self.enable_caching:
                self._cached_free[h] = None  # most recently used end
            else:
                self._free_uncached += 1
        self._private -= private_blocks
        self._free_uncached += private_blocks


# --------------------------------------------------------------------- sequences
@dataclass(slots=True)
class _Seq:
    seq_id: str
    prompt_ids: tuple[int, ...]
    max_tokens: int
    arrival_time: float
    seed: int
    output_ids: list[int] = field(default_factory=list)
    num_computed: int = 0
    hashes: list[int] = field(default_factory=list)  # chain hashes of full known blocks
    held_hashed: list[int] = field(default_factory=list)
    held_private: int = 0
    cached_tokens: int | None = None  # prefix-cache hit at first admission
    admitted_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    preemptions: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def remaining(self) -> int:
        return self.num_tokens - self.num_computed

    @property
    def is_decode(self) -> bool:
        return bool(self.output_ids) and self.remaining == 1

    def token_at(self, i: int) -> int:
        n = len(self.prompt_ids)
        return self.prompt_ids[i] if i < n else self.output_ids[i - n]

    def held_blocks(self) -> int:
        return len(self.held_hashed) + self.held_private


@dataclass(frozen=True, slots=True)
class Emission:
    seq_id: str
    token_id: int
    first: bool


@dataclass(frozen=True, slots=True)
class SeqTimings:
    prompt_tokens: int
    output_tokens: int
    cached_tokens: int
    queue_s: float
    prefill_s: float
    decode_s: float
    preemptions: int


@dataclass(frozen=True, slots=True)
class StepResult:
    start: float
    duration: float
    emitted: tuple[Emission, ...]
    finished: tuple[tuple[str, SeqTimings], ...]
    preempted: tuple[str, ...]
    prefill_tokens: int
    decode_seqs: int
    num_running: int
    num_waiting: int
    kv_usage: float


@dataclass(slots=True)
class EngineStats:
    steps: int = 0
    busy_time_s: float = 0.0
    prompt_tokens_computed: int = 0
    generation_tokens: int = 0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    preemptions: int = 0
    finished: int = 0
    aborted: int = 0


# --------------------------------------------------------------------- engine
class SimulatedEngine:
    def __init__(self, config: SimulatedEngineConfig | None = None) -> None:
        self.config = config or SimulatedEngineConfig()
        cfg = self.config
        self.blocks = BlockManager(cfg.num_kv_blocks, cfg.block_size, cfg.prefix_caching)
        self.stats = EngineStats()
        self._waiting: deque[_Seq] = deque()
        self._running: list[_Seq] = []
        self._by_id: dict[str, _Seq] = {}

    # ------------------------------------------------------------ public API
    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    def has_work(self) -> bool:
        return bool(self._running or self._waiting)

    def validate(self, prompt_len: int, max_tokens: int) -> None:
        cfg = self.config
        if prompt_len < 1:
            raise EngineRejectedError("prompt must contain at least one token")
        if max_tokens < 1:
            raise EngineRejectedError("max_tokens must be >= 1")
        if prompt_len + max_tokens > cfg.max_model_len:
            raise EngineRejectedError(
                f"prompt ({prompt_len}) + max_tokens ({max_tokens}) exceeds "
                f"max_model_len ({cfg.max_model_len})"
            )

    def submit(
        self, seq_id: str, prompt_ids: Sequence[int], max_tokens: int, arrival_time: float
    ) -> None:
        if seq_id in self._by_id:
            raise ValueError(f"duplicate sequence id {seq_id!r}")
        self.validate(len(prompt_ids), max_tokens)
        seq = _Seq(
            seq_id=seq_id,
            prompt_ids=tuple(prompt_ids),
            max_tokens=max_tokens,
            arrival_time=arrival_time,
            seed=zlib.crc32(seq_id.encode()),
        )
        self._extend_hashes(seq)
        self._waiting.append(seq)
        self._by_id[seq_id] = seq

    def abort(self, seq_id: str) -> bool:
        seq = self._by_id.pop(seq_id, None)
        if seq is None:
            return False
        if seq in self._running:
            self._running.remove(seq)
            self._release(seq)
        else:
            self._waiting.remove(seq)
        self.stats.aborted += 1
        return True

    def step(self, now: float) -> StepResult | None:
        """Run one engine iteration starting at ``now``; ``None`` if idle."""
        if not self.has_work():
            return None
        preempted: list[str] = []
        if self.config.chunked_prefill:
            scheduled = self._schedule_chunked(now, preempted)
        else:
            scheduled = self._schedule_prefill_first(now, preempted)
        if not scheduled:  # pragma: no cover - submit() rejects requests that can never fit
            raise RuntimeError("engine has work but nothing could be scheduled")
        return self._execute(now, scheduled, preempted)

    # ------------------------------------------------------------ scheduling
    def _schedule_chunked(self, now: float, preempted: list[str]) -> list[tuple[_Seq, int]]:
        cfg = self.config
        budget = cfg.max_num_batched_tokens
        scheduled: list[tuple[_Seq, int]] = []
        for seq in list(self._running):
            if seq.is_decode and seq in self._running and self._ensure_slot(seq, preempted):
                scheduled.append((seq, 1))
                budget -= 1
        for seq in self._running:
            if not seq.is_decode and budget > 0:
                chunk = min(seq.remaining, budget)
                scheduled.append((seq, chunk))
                budget -= chunk
        if not preempted:
            while self._waiting and budget > 0 and len(self._running) < cfg.max_num_seqs:
                seq = self._waiting[0]
                if not self._admit(seq, now):
                    break  # FCFS: no head-of-line bypass
                self._waiting.popleft()
                self._running.append(seq)
                chunk = min(seq.remaining, budget)
                scheduled.append((seq, chunk))
                budget -= chunk
        return scheduled

    def _schedule_prefill_first(self, now: float, preempted: list[str]) -> list[tuple[_Seq, int]]:
        cfg = self.config
        budget = cfg.max_num_batched_tokens
        scheduled: list[tuple[_Seq, int]] = []
        while self._waiting and len(self._running) < cfg.max_num_seqs:
            seq = self._waiting[0]
            hit_blocks = self.blocks.cached_prefix(seq.hashes)
            need = seq.num_tokens - min(hit_blocks * cfg.block_size, seq.num_tokens - 1)
            if need > budget or not self._admit(seq, now):
                break
            self._waiting.popleft()
            self._running.append(seq)
            scheduled.append((seq, seq.remaining))
            budget -= seq.remaining
        if scheduled:
            return scheduled  # prefill-only step: decodes stall
        for seq in list(self._running):
            if seq in self._running and self._ensure_slot(seq, preempted):
                scheduled.append((seq, 1))  # noqa: PERF401 - _ensure_slot mutates _running
        return scheduled

    def _admit(self, seq: _Seq, now: float) -> bool:
        bs = self.config.block_size
        total_blocks = math.ceil(seq.num_tokens / bs)
        allocation = self.blocks.allocate(seq.hashes, total_blocks)
        if allocation is None:
            return False
        hits, private = allocation
        seq.held_hashed = hits
        seq.held_private = private
        cached = min(len(hits) * bs, seq.num_tokens - 1)
        seq.num_computed = cached
        if seq.cached_tokens is None:
            seq.cached_tokens = cached
            seq.admitted_time = now
        self.stats.prefix_cache_queries += seq.num_tokens
        self.stats.prefix_cache_hits += cached
        return True

    def _ensure_slot(self, seq: _Seq, preempted: list[str]) -> bool:
        """Make sure ``seq`` has KV space for its next token, preempting if needed."""
        needed = math.ceil(seq.num_tokens / self.config.block_size)
        while seq.held_blocks() < needed:
            if self.blocks.allocate_private():
                seq.held_private += 1
                continue
            victim = self._running[-1]
            self._preempt(victim)
            preempted.append(victim.seq_id)
            if victim is seq:
                return False
        return True

    def _preempt(self, seq: _Seq) -> None:
        self._running.remove(seq)
        self._release(seq)
        seq.num_computed = 0
        seq.preemptions += 1
        self.stats.preemptions += 1
        self._waiting.appendleft(seq)

    def _release(self, seq: _Seq) -> None:
        # Only blocks whose KV was actually computed may stay cached.
        self.blocks.free(seq.held_hashed, seq.held_private)
        seq.held_hashed = []
        seq.held_private = 0

    # ------------------------------------------------------------ execution
    def _execute(
        self, now: float, scheduled: list[tuple[_Seq, int]], preempted: list[str]
    ) -> StepResult:
        cfg = self.config
        prefill_tokens = sum(n for seq, n in scheduled if not seq.is_decode)
        decodes = [seq for seq, _ in scheduled if seq.is_decode]
        attended = sum(seq.num_computed for seq in decodes)
        duration = (
            cfg.step_overhead_s
            + prefill_tokens * cfg.prefill_s_per_token
            + len(decodes) * cfg.decode_s_per_seq
            + attended * cfg.decode_s_per_kv_token
        )
        end = now + duration
        emitted: list[Emission] = []
        finished: list[tuple[str, SeqTimings]] = []
        for seq, n in scheduled:
            seq.num_computed += n
            self._promote_computed_blocks(seq)
            if seq.num_computed < seq.num_tokens:
                continue  # partial prefill chunk; no token yet
            token = VOCABULARY_IDS[(seq.seed + 7919 * len(seq.output_ids)) % len(VOCABULARY_IDS)]
            first = seq.first_token_time is None
            if first:
                seq.first_token_time = end
            seq.output_ids.append(token)
            self._extend_hashes(seq)
            emitted.append(Emission(seq.seq_id, token, first))
            if len(seq.output_ids) >= seq.max_tokens:
                finished.append((seq.seq_id, self._finish(seq, end)))
        self.stats.steps += 1
        self.stats.busy_time_s += duration
        self.stats.prompt_tokens_computed += prefill_tokens
        self.stats.generation_tokens += len(emitted)
        return StepResult(
            start=now,
            duration=duration,
            emitted=tuple(emitted),
            finished=tuple(finished),
            preempted=tuple(preempted),
            prefill_tokens=prefill_tokens,
            decode_seqs=len(decodes),
            num_running=len(self._running),
            num_waiting=len(self._waiting),
            kv_usage=self.blocks.usage,
        )

    def _finish(self, seq: _Seq, end: float) -> SeqTimings:
        seq.finish_time = end
        self._running.remove(seq)
        self._release(seq)
        del self._by_id[seq.seq_id]
        self.stats.finished += 1
        admitted = seq.admitted_time if seq.admitted_time is not None else seq.arrival_time
        first = seq.first_token_time if seq.first_token_time is not None else end
        return SeqTimings(
            prompt_tokens=len(seq.prompt_ids),
            output_tokens=len(seq.output_ids),
            cached_tokens=seq.cached_tokens or 0,
            queue_s=admitted - seq.arrival_time,
            prefill_s=first - admitted,
            decode_s=end - first,
            preemptions=seq.preemptions,
        )

    def _extend_hashes(self, seq: _Seq) -> None:
        """Compute chain hashes for newly *full* blocks of known tokens."""
        bs = self.config.block_size
        while (len(seq.hashes) + 1) * bs <= seq.num_tokens:
            i = len(seq.hashes)
            parent = seq.hashes[-1] if seq.hashes else 0
            block = tuple(seq.token_at(j) for j in range(i * bs, (i + 1) * bs))
            seq.hashes.append(hash((parent, block)))

    def _promote_computed_blocks(self, seq: _Seq) -> None:
        if not self.config.prefix_caching:
            return
        full_computed = min(seq.num_computed // self.config.block_size, len(seq.hashes))
        while len(seq.held_hashed) < full_computed and seq.held_private > 0:
            self.blocks.promote(seq.hashes[len(seq.held_hashed)])
            seq.held_hashed.append(seq.hashes[len(seq.held_hashed)])
            seq.held_private -= 1


#: Output token ids are drawn from the shared vocabulary so that generated text maps
#: back to the same ids when a client resends it (multi-turn prefix reuse).
VOCABULARY_IDS: tuple[int, ...] = tuple(range(len(VOCABULARY)))
