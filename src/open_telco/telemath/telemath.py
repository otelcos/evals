import re
from textwrap import dedent

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import accuracy, CORRECT, INCORRECT, Score, scorer, stderr, Target
from inspect_ai.solver import generate, system_message, TaskState

DEFAULT_DATASET = "GSMA/open_telco"
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
    """Extract the last \\boxed{...} content from response."""
    if not response:
        return ""
    matches = BOXED_PATTERN.findall(response)
    if not matches:
        return ""
    answer = WHITESPACE_PATTERN.sub("", matches[-1].strip())
    return answer.lstrip(":").rstrip("./")


def normalize_numeric(value: str) -> str | None:
    """Normalize numeric string (e.g., '5.0' -> '5')."""
    if not value:
        return None
    try:
        num = float(value)
        return str(int(num)) if num == int(num) else str(num)
    except ValueError:
        return None


@scorer(metrics=[accuracy(), stderr()])
def telemath_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        parsed = parse_boxed_answer(state.output.completion)
        pred_norm = normalize_numeric(parsed)
        target_norm = normalize_numeric(target.text)
        is_correct = parsed == target.text or (pred_norm is not None and pred_norm == target_norm)
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
) -> Task:
    return Task(
        dataset=hf_dataset(
            dataset_path,
            name=DEFAULT_DATASET_NAME,
            sample_fields=FieldSpec(input="question", target="answer"),
            split=split,
        ),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=telemath_scorer(),
    )
