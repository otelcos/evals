"""Macro-F1 + exact-set accuracy for multi-select MCQ (faithful to ZTE)."""

from __future__ import annotations

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Target,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from evals.telecom_bench.postprocess import (
    extract_non_reasoning_content,
    multiple_select_postprocess,
)


def options_of(text: str) -> set[str]:
    return set(multiple_select_postprocess(text))


def f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)


@metric
def macro_f1() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        vals = [float(s.score.metadata.get("f1", 0.0)) for s in scores]
        return sum(vals) / len(vals) if vals else 0.0

    return compute


@scorer(metrics=[accuracy(), stderr(), macro_f1()])
def multiselect_f1_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        pred = options_of(extract_non_reasoning_content(state.output.completion))
        gold = options_of(target.text)
        exact = pred == gold
        return Score(
            value=CORRECT if exact else INCORRECT,
            answer="".join(sorted(pred)),
            metadata={"f1": f1(pred, gold), "pred": sorted(pred), "gold": sorted(gold)},
        )

    return score
