"""TeleCom-Bench Knowledge Application: Entity Extraction.

Reference: upstream_ref/zte_domain/IDA/parameter_extract.py

Gold is a JSON string (e.g. '{"机房名称":"郑州金水机房","专业":"网管网"}').
structured_em_scorer(mode="json") applies str2json to both sides then are_json_equal.
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

GOLD_KEY = "answer"
DATA_FILE = KA / "Entity_Extraction" / "entity_extraction.json"


def record_to_sample(record: dict) -> Sample:
    return Sample(
        id=str(record.get("id", "")),
        input=record["question"],
        target=str(record[GOLD_KEY]),
        metadata={"set": "entity_extraction", "raw": record},
    )


def load_dataset() -> list[Sample]:
    raw = load_json(DATA_FILE)
    records = raw if isinstance(raw, list) else raw.get("questions", [])
    kept = [r for r in records if GOLD_KEY in r]
    skipped = len(records) - len(kept)
    if skipped:
        logger.warning(
            "entity_extraction: skipped %d record(s) missing %r", skipped, GOLD_KEY
        )
    return [record_to_sample(r) for r in kept]


@task
def telecom_bench_entity_extraction() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=structured_em_scorer(mode="json"),
    )
