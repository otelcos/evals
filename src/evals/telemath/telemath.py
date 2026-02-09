import math
import re
from textwrap import dedent

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate, system_message

from evals._utils import resolve_dataset

DEFAULT_DATASET = "GSMA/ot_sample_data"
DEFAULT_DATASET_NAME = "telemath"
DEFAULT_SPLIT = "test"

BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
WHITESPACE_PATTERN = re.compile(r"\n\s*")

SYSTEM_PROMPT = dedent(r"""
    You are an expert problem solver. Your task is to solve numerical exercises by following these guidelines:
    1.  **Understand the Goal:** Clearly identify what the problem is asking you to find, paying close attention to the required units for the final answer.
    2.  **Reason Step-by-Step:** Provide a clear, sequential reasoning process. Explain the formulas, principles, or logic used in each step. Show intermediate calculations if they clarify your thought process. The detailed structure of your sub-steps is up to you, as long as the reasoning is sound and easy to follow.
    3.  **Unit Management:**
        *   Track units throughout your calculations.
        *   **Crucially, ensure your final numerical answer is converted to the specific units requested in the problem statement.** If intermediate calculations result in a different unit, perform a final conversion step.
        *   State the unit of the final answer clearly in your explanatory text *before* the boxed answer.
    4.  **Final Numerical Answer Format:**
        *   The final answer must be a single numerical value (integer or float).
        *   Present this numerical value exclusively within the `\$\boxed{{...}}\$` format.
        *   **CRITICAL:** The `\$\boxed{{...}}\$` block must contain *only* the number. No text, no units, no labels (e.g., NOT `\$\boxed{{Result: 50}}\$` or `\$\boxed{{50 \text{{ mA}}}}\$`, but `\$\boxed{{50}}\$`).
    """).strip()


def parse_boxed_answer(response: str) -> str:
    r"""Extract the last \boxed{...} content from response."""
    if not response:
        return ""
    matches = BOXED_PATTERN.findall(response)
    if not matches:
        return ""
    answer = WHITESPACE_PATTERN.sub("", matches[-1].strip())
    return answer.lstrip(":").rstrip("./")


@scorer(metrics=[accuracy(), stderr()])
def telemath_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        parsed = parse_boxed_answer(state.output.completion)
        try:
            is_correct = math.isclose(
                float(parsed), float(target.text), rel_tol=0.01, abs_tol=0.01
            )
        except (ValueError, TypeError):
            is_correct = parsed == target.text

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=parsed,
            explanation=f"Parsed '{parsed}', target '{target.text}'",
        )

    return score


@task
def telemath(
    dataset_path: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    full: bool = False,
) -> Task:
    ds_path, ds_split = resolve_dataset(full, dataset_path, DEFAULT_DATASET, split)
    return Task(
        dataset=hf_dataset(
            ds_path,
            name=DEFAULT_DATASET_NAME,
            sample_fields=FieldSpec(input="question", target="answer"),
            split=ds_split,
        ),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=telemath_scorer(),
    )
