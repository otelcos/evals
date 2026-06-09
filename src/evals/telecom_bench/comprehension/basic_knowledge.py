"""TeleCom-Bench Knowledge Comprehension: Basic Knowledge (MCQ).

Reference: KC/Basic Theory/Basic_Knowledge/basic_knowledge.json
Top-level dict {total_sampled, questions:[...]}; 23 records.
Each record has flat A/B/C/D keys and a single-letter `answer` field.
Input: render_mcq(record). Gold: record["answer"].
Scorer: multiselect_f1_scorer().
"""

from __future__ import annotations

import logging

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from evals.telecom_bench.config import KC
from evals.telecom_bench.loaders import load_json, render_mcq
from evals.telecom_bench.scorers.multiselect_f1 import multiselect_f1_scorer

logger = logging.getLogger(__name__)

GOLD_KEY = "answer"
DATA_FILE = KC / "Basic Theory" / "Basic_Knowledge" / "basic_knowledge.json"


def record_to_sample(record: dict) -> Sample:
    return Sample(
        id=str(record.get("id", "")),
        input=render_mcq(record),
        target=str(record[GOLD_KEY]),
        metadata={"set": "basic_knowledge", "raw": record},
    )


def load_dataset() -> list[Sample]:
    raw = load_json(DATA_FILE)
    records: list[dict] = raw.get("questions", []) if isinstance(raw, dict) else raw
    kept = [r for r in records if GOLD_KEY in r]
    skipped = len(records) - len(kept)
    if skipped:
        logger.warning(
            "basic_knowledge: skipped %d record(s) missing %r", skipped, GOLD_KEY
        )
    return [record_to_sample(r) for r in kept]


@task
def telecom_bench_basic_knowledge() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=multiselect_f1_scorer(),
    )
