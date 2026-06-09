"""Tri-expert configurable LLM-judge panel (5-point Likert), faithful to ZTE.

Mirrors BaseJudgeACCEvaluator: each judge scores 1-5 against the reference;
we report the normalized mean as the Score value and the raw Likert mean +
inter-judge spread as metrics.
"""

from __future__ import annotations

import re

from inspect_ai.model import get_model
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Target,
    mean,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from evals.telecom_bench.config import DEFAULT_JUDGES

LIKERT_RUBRIC = """你是通信领域的资深专家评审。请根据[参考答案]评估[模型回答]的质量。
评分标准（5分制）：
5 = 完全正确、完整、专业；
4 = 基本正确，少量遗漏；
3 = 部分正确，有明显遗漏或错误；
2 = 大部分错误；
1 = 完全错误或答非所问。
[问题]
{question}
[参考答案]
{reference}
[模型回答]
{answer}
请只输出一个1到5之间的整数分数，不要输出任何其他内容。"""


def parse_likert(text: str) -> int | None:
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None


def aggregate(scores: list[int]) -> tuple[float, float, int]:
    mean_likert = sum(scores) / len(scores)
    norm = (mean_likert - 1) / 4
    spread = max(scores) - min(scores)
    return norm, mean_likert, spread


@metric
def mean_likert_metric() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        vals = [
            float(s.score.metadata["likert_mean"])
            for s in scores
            if s.score.metadata.get("likert_mean") is not None
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    return compute


@scorer(metrics=[mean(), stderr(), mean_likert_metric()])
def judge_panel_scorer(judges: list[str | None] | None = None, single: bool = False):
    panel = list(judges) if judges is not None else list(DEFAULT_JUDGES)
    if single:
        panel = panel[:1]

    async def score(state: TaskState, target: Target) -> Score:
        prompt = LIKERT_RUBRIC.format(
            question=state.input_text,
            reference=target.text,
            answer=state.output.completion,
        )
        raw: list[int] = []
        for judge in panel:
            out = await get_model(judge).generate(prompt)
            parsed = parse_likert(out.completion)
            if parsed is not None:
                raw.append(parsed)
        if not raw:
            return Score(
                value=0.0, answer="no judge score", metadata={"likert_mean": None}
            )
        norm, mean_likert, spread = aggregate(raw)
        return Score(
            value=norm,
            answer=str(mean_likert),
            metadata={"likert_mean": mean_likert, "panel": raw, "spread": spread},
        )

    return score
