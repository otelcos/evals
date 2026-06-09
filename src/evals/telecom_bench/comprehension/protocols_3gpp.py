"""TeleCom-Bench Knowledge Comprehension: 3GPP Protocols (all MCQ).

Reference: upstream_ref/zte_domain/tele_3gpp/

Dataset: KC/Basic Theory/3GPP_Protocols/3GPP_protocols.json
Shape: dict {total_sampled, stratify_by, questions:[...]} — 36 records.
Record keys: id, 题型, question, answer, A, B, C, D, difficulty, prompt.
Gold field: answer — comma-separated letters e.g. "A,B,C,D" (single or multi).
Input: record["prompt"] (pre-rendered); fall back to render_mcq only if absent.
Scorer: multiselect_f1_scorer()
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
DATA_FILE = KC / "Basic Theory" / "3GPP_Protocols" / "3GPP_protocols.json"


def record_to_sample(record: dict) -> Sample:
    input_text = record.get("prompt") or render_mcq(record)
    return Sample(
        id=str(record.get("id", "")),
        input=input_text,
        target=str(record[GOLD_KEY]),
        metadata={
            "set": "protocols_3gpp",
            "题型": record.get("题型", ""),
            "raw": record,
        },
    )


def load_dataset() -> list[Sample]:
    raw = load_json(DATA_FILE)
    records = raw.get("questions", [])
    kept = [r for r in records if GOLD_KEY in r]
    skipped = len(records) - len(kept)
    if skipped:
        logger.warning(
            "protocols_3gpp: skipped %d record(s) missing %r", skipped, GOLD_KEY
        )
    return [record_to_sample(r) for r in kept]


@task
def telecom_bench_protocols_3gpp() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=multiselect_f1_scorer(),
    )
