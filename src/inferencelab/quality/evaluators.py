"""Quality and equivalence validation for optimised deployments.

An optimisation (quantization, speculative decoding, a different engine) is only a
win if quality holds. This module compares a *candidate* deployment against a
*reference* deployment of the same checkpoint, deterministically (temperature 0,
fixed seed), using cheap checks rather than an LLM judge:

* **exact match**        identical greedy outputs (expected for lossless changes such
                         as speculative decoding; not expected across quantization)
* **prefix agreement**   fraction of the reference output's words reproduced before
                         the first divergence
* **task accuracy**      small built-in tasks with checkable answers (arithmetic,
                         keyword retrieval, JSON structure)
* **JSON validity**      structured-output responses parse and contain required keys

Note: numerical non-determinism across batch sizes means even "lossless" engines can
diverge on long outputs; report the agreement distribution, not a single bit.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from inferencelab.backends.base import BackendRequest, InferenceBackend
from inferencelab.domain.request import ChatMessage, RequestKind, SamplingParams
from inferencelab.schema import MutableModel, StrictModel

__all__ = [
    "BUILTIN_CASES",
    "QualityCase",
    "QualityReport",
    "check_output",
    "compare_backends",
    "json_valid",
    "prefix_agreement",
]


class QualityCase(StrictModel):
    id: str
    prompt: str
    kind: Literal["exact", "contains", "number", "json"] = "exact"
    expected: str | None = None
    required_keys: list[str] = Field(default_factory=list)


BUILTIN_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        id="add",
        prompt="What is 17 + 25? Reply with the number only.",
        kind="number",
        expected="42",
    ),
    QualityCase(
        id="mul",
        prompt="What is 9 times 8? Reply with the number only.",
        kind="number",
        expected="72",
    ),
    QualityCase(
        id="sub",
        prompt="Compute 100 - 37. Reply with the number only.",
        kind="number",
        expected="63",
    ),
    QualityCase(
        id="capital",
        prompt="What is the capital of France? Answer in one word.",
        kind="contains",
        expected="paris",
    ),
    QualityCase(
        id="retrieve",
        prompt="The code word is 'heliotrope'. Repeat the code word and nothing else.",
        kind="contains",
        expected="heliotrope",
    ),
    QualityCase(
        id="json",
        prompt='Return only a JSON object with keys "name" and "age" for Ada, aged 36.',
        kind="json",
        required_keys=["name", "age"],
    ),
)


def _words(text: str) -> list[str]:
    return text.strip().split()


def prefix_agreement(reference: str, candidate: str) -> float:
    ref, cand = _words(reference), _words(candidate)
    if not ref:
        return 1.0 if not cand else 0.0
    n = 0
    for a, b in zip(ref, cand, strict=False):
        if a != b:
            break
        n += 1
    return n / len(ref)


def json_valid(text: str, required_keys: Sequence[str] = ()) -> bool:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and all(k in value for k in required_keys)


def check_output(case: QualityCase, text: str) -> bool:
    if case.kind == "json":
        return json_valid(text, case.required_keys)
    expected = (case.expected or "").strip().lower()
    got = text.strip().lower()
    if case.kind == "number":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", got)
        return bool(numbers) and numbers[0] == expected
    if case.kind == "contains":
        return expected in got
    return got == expected


class CaseResult(MutableModel):
    id: str
    exact_match: bool
    prefix_agreement: float
    reference_correct: bool
    candidate_correct: bool


class QualityReport(MutableModel):
    cases: int
    exact_match_rate: float
    mean_prefix_agreement: float
    reference_accuracy: float
    candidate_accuracy: float
    accuracy_delta: float
    per_case: list[CaseResult]
    notes: list[str] = Field(default_factory=list)


async def _generate(
    backend: InferenceBackend, model: str, case: QualityCase, max_tokens: int, seed: int
) -> str:
    request = BackendRequest(
        request_id=f"quality-{case.id}",
        model=model,
        kind=RequestKind.CHAT,
        sampling=SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0, seed=seed),
        messages=(ChatMessage("user", case.prompt),),
    )
    return (await backend.infer(request)).text


async def compare_backends(
    reference: InferenceBackend,
    candidate: InferenceBackend,
    *,
    reference_model: str,
    candidate_model: str,
    cases: Sequence[QualityCase] = BUILTIN_CASES,
    max_tokens: int = 64,
    seed: int = 0,
) -> QualityReport:
    results: list[CaseResult] = []
    for case in cases:
        ref = await _generate(reference, reference_model, case, max_tokens, seed)
        cand = await _generate(candidate, candidate_model, case, max_tokens, seed)
        results.append(
            CaseResult(
                id=case.id,
                exact_match=ref.strip() == cand.strip(),
                prefix_agreement=prefix_agreement(ref, cand),
                reference_correct=check_output(case, ref),
                candidate_correct=check_output(case, cand),
            )
        )
    n = len(results) or 1
    ref_acc = sum(r.reference_correct for r in results) / n
    cand_acc = sum(r.candidate_correct for r in results) / n
    return QualityReport(
        cases=len(results),
        exact_match_rate=sum(r.exact_match for r in results) / n,
        mean_prefix_agreement=sum(r.prefix_agreement for r in results) / n,
        reference_accuracy=ref_acc,
        candidate_accuracy=cand_acc,
        accuracy_delta=cand_acc - ref_acc,
        per_case=results,
        notes=["builtin cases are a smoke test, not a benchmark; add task-specific cases"],
    )
