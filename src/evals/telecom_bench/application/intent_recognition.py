"""TeleCom-Bench Knowledge Application: Intent Recognition (faithful static).

Reference: upstream_ref/zte_domain/IDA/intent_recognize.py

The gold label is a bare class string ("DONE"/"UNDONE"/"ORDER"/"NO"). Upstream
applies str2json to both sides then ==, but str2json returns None for bare class
labels (it would score everything incorrect), so we use normalized exact-string
match (mode="exact") with INTENT_PRE replicating upstream's Output:/Thought: split.
"""

from __future__ import annotations

import logging

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from evals.telecom_bench.config import KA
from evals.telecom_bench.loaders import load_json
from evals.telecom_bench.scorers.structured_em import structured_em_scorer

logger = logging.getLogger(__name__)

GOLD_KEY = "output"
DATA_FILE = KA / "Intent_Recognition" / "intent_recognition.json"


def INTENT_PRE(text: str) -> str:
    """ZTE preprocessing: keep the Output: segment before the next Thought:."""
    return text.split("Output:")[-1].split("\nThought:")[0].strip()


def record_to_sample(record: dict) -> Sample:
    return Sample(
        id=str(record.get("id", "")),
        input=record["input"],
        target=str(record[GOLD_KEY]),
        metadata={"set": "intent_recognition", "raw": record},
    )


def load_dataset() -> list[Sample]:
    raw = load_json(DATA_FILE)
    records = raw if isinstance(raw, list) else raw.get("questions", [])
    kept = [r for r in records if GOLD_KEY in r]
    skipped = len(records) - len(kept)
    if skipped:
        logger.warning(
            "intent_recognition: skipped %d record(s) missing %r", skipped, GOLD_KEY
        )
    return [record_to_sample(r) for r in kept]


@task
def telecom_bench_intent_recognition() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=structured_em_scorer(mode="exact", pre=INTENT_PRE),
    )
