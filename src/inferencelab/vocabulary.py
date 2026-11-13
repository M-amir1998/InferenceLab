"""Deterministic vocabulary for synthetic text.

Common short English words; each is a single token (with a leading space) in typical
BPE vocabularies, so synthetic prompts of N words are close to N tokens. The mock and
simulated engines also emit these words, which lets a multi-turn client send generated
text back and have the simulator's block hashes line up (prefix-cache hits on history).
"""

from __future__ import annotations

import zlib

__all__ = ["VOCABULARY", "WORD_INDEX", "word_to_id"]

VOCABULARY: tuple[str, ...] = tuple(
    """
    the of and to in is for on that with as it at by from this be are or an was
    which not but all can has have one more were will their if we there two new also
    time some other into its than may only these first after any most such where use
    data system model value number group state level work part result point area case
    line order form rate cost test run load node graph table query cache block token
    batch queue stream route plan step phase field range scale index score track trace
    signal metric event store record layer stage frame limit window delay power speed
    space shape class type mode path page file code task job user team rule key map
    set list unit core host port link role view sort pair base edge gate loop
    """.split()  # noqa: SIM905 - a word list reads better as prose
)

WORD_INDEX: dict[str, int] = {w: i for i, w in enumerate(VOCABULARY)}

#: Out-of-vocabulary words hash into a disjoint id range.
_OOV_OFFSET = 1 << 20


def word_to_id(word: str) -> int:
    """Stable token id for a word (independent of PYTHONHASHSEED)."""
    idx = WORD_INDEX.get(word)
    if idx is not None:
        return idx
    return _OOV_OFFSET + zlib.crc32(word.encode("utf-8"))
