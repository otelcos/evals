"""TeleCom-Bench Knowledge Comprehension: Wired Network (MCQ, Chinese nested keys).

Reference: KC/Product Knowledge/Wired_Nerwork/wired_network.json
Top-level dict {total_sampled, questions:[...]}; 30 records.
Each record is {id, <nested>} where <nested> is either 单项选择题 (×12)
or 多项选择题 (×18). The nested dict is {问题, 选项 (LIST of "A. ..." strings), 答案}.
Input: 问题 + newline + newline-joined 选项. Gold: 答案.
Scorer: multiselect_f1_scorer().
"""

from __future__ import annotations

import logging

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from evals.telecom_bench.config import KC
from evals.telecom_bench.loaders import load_json
from evals.telecom_bench.scorers.multiselect_f1 import multiselect_f1_scorer

logger = logging.getLogger(__name__)

GOLD_KEY = "答案"
DATA_FILE = KC / "Product Knowledge" / "Wired_Nerwork" / "wired_network.json"

NESTED_KEYS = ("单项选择题", "多项选择题")


def render_wired(inner: dict) -> str:
    """Render 问题 + 选项 list into a single prompt string."""
    stem: str = inner["问题"]
    options: list[str] = inner["选项"]
    return stem + "\n" + "\n".join(options)


def record_to_sample(record: dict) -> Sample:
    # Pick the nested key that actually carries the gold answer (matches the
    # load_dataset filter), not merely the first nested key present.
    nested_key = next(k for k in NESTED_KEYS if k in record and GOLD_KEY in record[k])
    inner: dict = record[nested_key]
    return Sample(
        id=str(record.get("id", "")),
        input=render_wired(inner),
        target=str(inner[GOLD_KEY]),
        metadata={"set": "wired_network", "type": nested_key, "raw": record},
    )


def load_dataset() -> list[Sample]:
    raw = load_json(DATA_FILE)
    records: list[dict] = raw.get("questions", []) if isinstance(raw, dict) else raw
    kept = [r for r in records if any(k in r and GOLD_KEY in r[k] for k in NESTED_KEYS)]
    skipped = len(records) - len(kept)
    if skipped:
        logger.warning(
            "wired_network: skipped %d record(s) missing %r", skipped, GOLD_KEY
        )
    return [record_to_sample(r) for r in kept]


@task
def telecom_bench_wired_network() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=multiselect_f1_scorer(),
    )
