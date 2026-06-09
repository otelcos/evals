"""Structured Exact Match: JSON equality (str2json + are_json_equal) or string EM."""

from __future__ import annotations

from collections.abc import Callable

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from evals.telecom_bench.postprocess import (
    are_json_equal,
    extract_non_reasoning_content,
    normalize_zh,
    str2json,
)


def judge_correct(
    completion: str,
    target: str,
    *,
    mode: str = "json",
    pre: Callable[[str], str] | None = None,
) -> bool:
    raw = extract_non_reasoning_content(completion)
    if pre is not None:
        raw = pre(raw)
    if mode == "json":
        pred = str2json(raw)
        gold = str2json(target)
        return pred is not None and gold is not None and are_json_equal(pred, gold)
    return normalize_zh(raw) == normalize_zh(target)


@scorer(metrics=[accuracy(), stderr()])
def structured_em_scorer(mode: str = "json", pre: Callable[[str], str] | None = None):
    async def score(state: TaskState, target: Target) -> Score:
        correct = judge_correct(
            state.output.completion, target.text, mode=mode, pre=pre
        )
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=extract_non_reasoning_content(state.output.completion)[:500],
            metadata={"mode": mode, "correct": correct},
        )

    return score
