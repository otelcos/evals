"""TeleCom-Bench Knowledge Comprehension: Core Network (subjective QA).

Reference: upstream_ref/zte_domain/ume_inclusion/

Record shape (Chinese keys): {难度, 大类, 题目, 答案, product, id}
- 题目 = question text
- 答案 = free-text reference answer

Scored by a tri-expert LLM judge panel (5-point Likert), faithful to ZTE's
BaseJudgeACCEvaluator.
"""

from __future__ import annotations

import logging

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from evals.telecom_bench.config import KC
from evals.telecom_bench.loaders import load_json
from evals.telecom_bench.scorers.judge_panel import judge_panel_scorer

logger = logging.getLogger(__name__)

GOLD_KEY = "答案"
INPUT_KEY = "题目"
DATA_FILE = KC / "Product Knowledge" / "Core_Network" / "core_network.json"


def record_to_sample(record: dict) -> Sample:
    return Sample(
        id=str(record.get("id", "")),
        input=record[INPUT_KEY],
        target=str(record[GOLD_KEY]),
        metadata={
            "set": "core_network",
            "product": record.get("product", ""),
            "difficulty": record.get("难度", ""),
            "category": record.get("大类", ""),
            "raw": record,
        },
    )


def load_dataset() -> list[Sample]:
    raw = load_json(DATA_FILE)
    records: list[dict] = raw.get("questions", [])
    kept = [r for r in records if GOLD_KEY in r and INPUT_KEY in r]
    skipped = len(records) - len(kept)
    if skipped:
        logger.warning(
            "core_network: skipped %d record(s) missing %r or %r",
            skipped,
            GOLD_KEY,
            INPUT_KEY,
        )
    return [record_to_sample(r) for r in kept]


@task
def telecom_bench_core_network() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=judge_panel_scorer(),
    )
