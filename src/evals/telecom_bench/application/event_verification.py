"""TeleCom-Bench Knowledge Application: Event Verification.

Reference: upstream_ref/zte_domain/ai_cs/ai_cs.py

The upstream evaluator (AICSScoreEvaluator) uses an LLM judge rather than
deterministic dict comparison (AICSQAMatchEvaluator is a separate class used
for different tasks). This set therefore uses judge_panel_scorer().

The data file is a single top-level dict {question, best_answer} where
best_answer is a 1-item list of a structured dict. The single record is
wrapped into a 1-element dataset. Gold/target is json.dumps(best_answer).
"""

from __future__ import annotations

import json
import logging

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from evals.telecom_bench.config import KA
from evals.telecom_bench.loaders import load_json
from evals.telecom_bench.scorers.judge_panel import judge_panel_scorer

logger = logging.getLogger(__name__)

GOLD_KEY = "best_answer"
DATA_FILE = KA / "Event_Verification" / "event_verification.json"


def record_to_sample(record: dict) -> Sample:
    gold = json.dumps(record[GOLD_KEY], ensure_ascii=False)
    return Sample(
        input=record["question"],
        target=gold,
        metadata={"set": "event_verification", "raw": record},
    )


def load_dataset() -> list[Sample]:
    raw = load_json(DATA_FILE)
    # The file is a single dict, not a list; wrap it so we can filter uniformly.
    records = [raw] if isinstance(raw, dict) else raw
    kept = [r for r in records if GOLD_KEY in r]
    skipped = len(records) - len(kept)
    if skipped:
        logger.warning(
            "event_verification: skipped %d record(s) missing %r", skipped, GOLD_KEY
        )
    return [record_to_sample(r) for r in kept]


@task
def telecom_bench_event_verification() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=judge_panel_scorer(),
    )
