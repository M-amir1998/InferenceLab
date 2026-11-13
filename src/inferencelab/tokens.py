"""Token counting helpers.

InferenceLab prefers token counts *reported by the serving engine*. When an engine does
not report usage, counts are estimated locally and labelled
:attr:`~inferencelab.domain.response.TokenCountSource.ESTIMATED` everywhere.

The synthetic workload generator emits text built from common short English words that
are single tokens in typical BPE vocabularies, so ``WhitespaceTokenCounter`` is a good
estimate for synthetic prompts. ``HeuristicTokenCounter`` (≈4 characters per token)
is the fallback for arbitrary text. Neither is exact for any particular tokenizer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from inferencelab.domain.request import ChatMessage

__all__ = [
    "HeuristicTokenCounter",
    "TokenCounter",
    "WhitespaceTokenCounter",
    "estimate_chat_tokens",
]

#: Rough per-message overhead of common chat templates (role markers, separators).
CHAT_MESSAGE_OVERHEAD_TOKENS = 4


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class WhitespaceTokenCounter:
    """One token per whitespace-separated word (exact for the simulator's tokenizer)."""

    def count(self, text: str) -> int:
        return len(text.split())


class HeuristicTokenCounter:
    """``ceil(chars / chars_per_token)``: the classic English BPE rule of thumb."""

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token) if text else 0


def estimate_chat_tokens(messages: Sequence[ChatMessage], counter: TokenCounter) -> int:
    return sum(counter.count(m.content) + CHAT_MESSAGE_OVERHEAD_TOKENS for m in messages)
