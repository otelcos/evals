"""TeleCom-Bench Knowledge Comprehension: 5G Network (mixed MCQ + true/false).

File: KC/Basic Theory/5G_Network/5G_network.json
Shape: dict {total_sampled, questions:[23 records]}
Record keys: id, source_file, question, A, B, C, D, answer

Mixed question types (from source_file):
  - 单选题: single-select MCQ; answer is a letter (A/B/C/D)
  - 多选题: multi-select MCQ; answer is concatenated letters (e.g. "ABCD")
  - 判断题: true/false; A="正确", B="错误", C="None", D="None";
            answer is "T" (correct) or "F" (incorrect/false)

T/F normalization: "T" -> "A" (正确), "F" -> "B" (错误).
Without this, multiselect_f1 compares {"F"} vs the model's {"A"|"B"} and always
scores 0. The mapping is validated at load time by checking A=="正确" and B=="错误"
for 判断题 records (otherwise we map by matching the 正确/错误 text).

Option values equal to the string "None" are skipped when rendering.
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

GOLD_KEY = "answer"
DATA_FILE = KC / "Basic Theory" / "5G_Network" / "5G_network.json"

_LETTERS = ("A", "B", "C", "D")


def _normalize_tf(record: dict) -> str:
    """Map 判断题 gold "T"/"F" to the matching option letter.

    Checks that A=="正确" and B=="错误"; if not, falls back to searching all
    options for 正确/错误 text.
    """
    raw = record[GOLD_KEY]
    if raw not in ("T", "F"):
        return raw

    # Preferred: static mapping validated against the record's option text
    a_val = record.get("A", "")
    b_val = record.get("B", "")
    if a_val == "正确" and b_val == "错误":
        return "A" if raw == "T" else "B"

    # Fallback: search all options for 正确/错误
    target_text = "正确" if raw == "T" else "错误"
    for ltr in _LETTERS:
        if record.get(ltr, "") == target_text:
            return ltr

    # Should never reach here given the verified data shape
    logger.warning(
        "network_5g: cannot map T/F answer for record id=%s", record.get("id")
    )
    return raw


def render_5g_mcq(record: dict) -> str:
    """Render question + options, skipping options whose value is the string 'None'."""
    stem = record["question"]
    lines = [
        f"{ltr}. {record[ltr]}"
        for ltr in _LETTERS
        if ltr in record and record[ltr] and record[ltr] != "None"
    ]
    return stem + "\n" + "\n".join(lines)


def record_to_sample(record: dict) -> Sample:
    gold = _normalize_tf(record)
    return Sample(
        id=str(record.get("id", "")),
        input=render_5g_mcq(record),
        target=gold,
        metadata={
            "set": "network_5g",
            "source_file": record.get("source_file", ""),
            "raw_answer": record[GOLD_KEY],
            "raw": record,
        },
    )


def load_dataset() -> list[Sample]:
    raw = load_json(DATA_FILE)
    records: list[dict] = raw.get("questions", []) if isinstance(raw, dict) else raw
    kept = [r for r in records if GOLD_KEY in r]
    skipped = len(records) - len(kept)
    if skipped:
        logger.warning("network_5g: skipped %d record(s) missing %r", skipped, GOLD_KEY)
    return [record_to_sample(r) for r in kept]


@task
def telecom_bench_network_5g() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=multiselect_f1_scorer(),
    )
