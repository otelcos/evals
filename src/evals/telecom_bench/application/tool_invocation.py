r"""TeleCom-Bench Knowledge Application: Tool Invocation (faithful static).

Reference: lagent / teval boxed-answer convention.

The data file is a single dict {conversations, extra_info}. The 7-turn
conversation contains 3 assistant turns each holding a \boxed{...} conclusion.
extra_info has 3 keys (事件核查结果, 一级根因, 二级根因) whose values match those
boxed conclusions in order.

Input:  system turn content + first user turn content.
Target: the 3 extra_info values joined with "|".
Scorer: structured_em_scorer(mode="exact", pre=BOXED_PRE) where BOXED_PRE
        extracts all \boxed{...} contents from the model output and joins them
        with "|", mirroring the target construction.

NOTE: this is the loosest set -- flag for reviewer attention.
"""

from __future__ import annotations

import logging
import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from evals.telecom_bench.config import KA
from evals.telecom_bench.loaders import load_json
from evals.telecom_bench.scorers.structured_em import structured_em_scorer

logger = logging.getLogger(__name__)

# Same regex used in src/evals/telco_challenge/track_a/config.py
_BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")

GOLD_KEYS = ("事件核查结果", "一级根因", "二级根因")
DATA_FILE = KA / "Tool_Invocation" / "tool_invocation.json"


def BOXED_PRE(text: str) -> str:
    r"""Extract all \boxed{...} contents from model output and join with '|'."""
    matches = _BOXED_PATTERN.findall(text)
    return "|".join(m.strip() for m in matches)


def _build_target(extra_info: dict) -> str:
    """Join the 3 extra_info values in canonical order with '|'."""
    return "|".join(str(extra_info[k]) for k in GOLD_KEYS)


def record_to_sample(record: dict) -> Sample:
    conversations = record["conversations"]
    extra_info = record["extra_info"]

    # Input: system turn + first user turn
    system_content = conversations[0]["content"]
    user_content = conversations[1]["content"]
    input_text = system_content + "\n" + user_content

    target = _build_target(extra_info)

    return Sample(
        id="tool_invocation_0",
        input=input_text,
        target=target,
        metadata={"set": "tool_invocation", "extra_info": extra_info},
    )


def load_dataset() -> list[Sample]:
    raw = load_json(DATA_FILE)
    # raw is a single dict; check required keys
    required = {"conversations", "extra_info"}
    if not required.issubset(raw.keys()):
        missing = required - raw.keys()
        logger.warning("tool_invocation: skipped 1 record(s) missing %r", missing)
        return []
    # Verify all 3 gold keys are present in extra_info
    extra_info = raw.get("extra_info", {})
    missing_gold = [k for k in GOLD_KEYS if k not in extra_info]
    if missing_gold:
        logger.warning(
            "tool_invocation: skipped 1 record(s) missing gold keys %r",
            missing_gold,
        )
        return []
    return [record_to_sample(raw)]


@task
def telecom_bench_tool_invocation() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=structured_em_scorer(mode="exact", pre=BOXED_PRE),
    )
