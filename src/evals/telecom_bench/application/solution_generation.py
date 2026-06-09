r"""TeleCom-Bench Knowledge Application: Solution Generation.

Reference: upstream_ref/zte_domain/ume_exclusion/solution.py

Each record is a fault scenario; the gold is a step-sequence string
(best_answer) listing tool invocations in order. We expose two @tasks:

  telecom_bench_solution_generation        -- tool-step exact match
  telecom_bench_solution_generation_judged -- judge_panel (LLM judge)

Faithfulness: upstream UMESolutionEvaluator's binary metric is tool_step_accuracy,
computed by extracting the bracketed tool steps (re.findall(r"\\[(.*?)\\]")) from
both prediction and reference and requiring the extracted step LISTS to be equal
(upstream also reports ROUGE, which the _judged variant proxies). We therefore
extract the bracketed tool steps before exact comparison rather than matching the
full free-text prose, which would never match the irregular gold strings.
"""

from __future__ import annotations

import logging
import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from evals.telecom_bench.config import KA
from evals.telecom_bench.loaders import load_json
from evals.telecom_bench.scorers.judge_panel import judge_panel_scorer
from evals.telecom_bench.scorers.structured_em import structured_em_scorer

logger = logging.getLogger(__name__)

GOLD_KEY = "best_answer"
DATA_FILE = KA / "Solution_Generation" / "solution_generation.json"


def SOLUTION_PRE(text: str) -> str:
    r"""Extract the bracketed tool steps, joined, for exact comparison.

    Mirrors upstream UMESolutionEvaluator._extract_tool_steps: the ordered list of
    re.findall(r"\[(.*?)\]", text). Applied to both prediction and gold so the
    structured_em exact comparison is over tool-step sequences, not raw prose.
    """
    return "|".join(re.findall(r"\[(.*?)\]", text))


def record_to_sample(record: dict, *, extract_steps: bool = False) -> Sample:
    """Map a record to a Sample.

    extract_steps=True reduces the gold to its bracketed tool-step sequence (for the
    tool-step EM task, whose scorer applies SOLUTION_PRE to the model output too).
    extract_steps=False keeps the full prose gold (for the LLM-judge variant).
    """
    gold = str(record[GOLD_KEY])
    return Sample(
        input=record["question"],
        target=SOLUTION_PRE(gold) if extract_steps else gold,
        metadata={"set": "solution_generation", "raw": record},
    )


def load_dataset(*, extract_steps: bool = False) -> list[Sample]:
    raw = load_json(DATA_FILE)
    records = raw if isinstance(raw, list) else raw.get("questions", [])
    kept = [r for r in records if GOLD_KEY in r]
    skipped = len(records) - len(kept)
    if skipped:
        logger.warning(
            "solution_generation: skipped %d record(s) missing %r", skipped, GOLD_KEY
        )
    return [record_to_sample(r, extract_steps=extract_steps) for r in kept]


@task
def telecom_bench_solution_generation() -> Task:
    return Task(
        dataset=load_dataset(extract_steps=True),
        solver=generate(),
        scorer=structured_em_scorer(mode="exact", pre=SOLUTION_PRE),
    )


@task
def telecom_bench_solution_generation_judged() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=judge_panel_scorer(),
    )
