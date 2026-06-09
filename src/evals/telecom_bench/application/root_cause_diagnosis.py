"""TeleCom-Bench Knowledge Application: Root Cause Diagnosis (faithful static).

Reference: upstream_ref/zte_domain/ai_cs/alarm_nodes.py

Two files are merged into a single sample:
- input.json  — the alarm graph {nodes:[15], edges:[15]}
- label.json  — the ground-truth root causes {nodes:[2]}

Upstream scoring uses alarm_nodes.are_json_equal, which is faithfully ported
into the shared structured_em scorer (mode="json").
"""

from __future__ import annotations

import json
import logging

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from evals.telecom_bench.config import KA
from evals.telecom_bench.loaders import load_json
from evals.telecom_bench.scorers.structured_em import structured_em_scorer

logger = logging.getLogger(__name__)

INPUT_FILE = KA / "Root_Cause_Diagnosis" / "input.json"
LABEL_FILE = KA / "Root_Cause_Diagnosis" / "label.json"


def load_dataset() -> list[Sample]:
    input_data = load_json(INPUT_FILE)
    label_data = load_json(LABEL_FILE)

    skipped = 0
    if not isinstance(input_data, dict) or "nodes" not in input_data:
        logger.warning(
            "root_cause_diagnosis: input.json missing 'nodes'; skipping sample"
        )
        skipped += 1
    if not isinstance(label_data, dict) or "nodes" not in label_data:
        logger.warning(
            "root_cause_diagnosis: label.json missing 'nodes'; skipping sample"
        )
        skipped += 1

    if skipped:
        logger.warning(
            "root_cause_diagnosis: skipped %d file(s) with missing data", skipped
        )
        return []

    return [
        Sample(
            id="root_cause_diagnosis_0",
            input=json.dumps(input_data, ensure_ascii=False),
            target=json.dumps(label_data, ensure_ascii=False),
            metadata={"set": "root_cause_diagnosis"},
        )
    ]


@task
def telecom_bench_root_cause_diagnosis() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=structured_em_scorer(mode="json"),
    )
