"""TeleCom-Bench Knowledge Comprehension: Wireless Network (MCQ; two merged files).

References:
  KC/Product Knowledge/Wireless_Network/fault_maintenance.json  (n=33)
  KC/Product Knowledge/Wireless_Network/network_optimization.json  (n=33)
Total: 66 samples.

fault_maintenance records: {id, question, options (LIST of "A. ..." strings), answer (letter), ...}
  Render: question + newline-joined options list.  Gold: answer.

network_optimization records: {id, question_type, type, stem, options (DICT {"A":..}),
  correct_answers (LIST e.g. ["B"]), knowledge, capability}
  Render: stem + newline-joined "K. V" items.  Gold: "".join(correct_answers).

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

_WIRELESS_DIR = KC / "Product Knowledge" / "Wireless_Network"
FAULT_FILE = _WIRELESS_DIR / "fault_maintenance.json"
OPTIM_FILE = _WIRELESS_DIR / "network_optimization.json"


def _render_fault(record: dict) -> str:
    """Render a fault_maintenance record: question + newline-joined options list."""
    return record["question"] + "\n" + "\n".join(record["options"])


def _render_optim(record: dict) -> str:
    """Render a network_optimization record: stem + 'K. V' lines from options dict."""
    opts = "\n".join(f"{k}. {v}" for k, v in record["options"].items())
    return record["stem"] + "\n" + opts


def _load_fault_samples() -> list[Sample]:
    """Load and normalize fault_maintenance.json records."""
    records: list[dict] = load_json(FAULT_FILE)
    kept = [r for r in records if "answer" in r]
    skipped = len(records) - len(kept)
    if skipped:
        logger.warning(
            "wireless_network/fault_maintenance: skipped %d record(s) missing 'answer'",
            skipped,
        )
    return [
        Sample(
            id=f"fm_{r.get('id', i)}",
            input=_render_fault(r),
            target=str(r["answer"]),
            metadata={
                "set": "wireless_network",
                "source": "fault_maintenance",
                "raw": r,
            },
        )
        for i, r in enumerate(kept)
    ]


def _load_optim_samples() -> list[Sample]:
    """Load and normalize network_optimization.json records."""
    records: list[dict] = load_json(OPTIM_FILE)
    kept = [r for r in records if "correct_answers" in r]
    skipped = len(records) - len(kept)
    if skipped:
        logger.warning(
            "wireless_network/network_optimization: skipped %d record(s) missing 'correct_answers'",
            skipped,
        )
    return [
        Sample(
            id=f"no_{r.get('id', i)}",
            input=_render_optim(r),
            target="".join(r["correct_answers"]),
            metadata={
                "set": "wireless_network",
                "source": "network_optimization",
                "raw": r,
            },
        )
        for i, r in enumerate(kept)
    ]


def load_dataset() -> list[Sample]:
    """Merge fault_maintenance and network_optimization into one dataset (~66 samples)."""
    return _load_fault_samples() + _load_optim_samples()


@task
def telecom_bench_wireless_network() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=multiselect_f1_scorer(),
    )
