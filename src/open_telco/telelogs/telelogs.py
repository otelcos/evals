import re
from typing import Literal

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
from inspect_ai.solver import TaskState, generate

from open_telco.telelogs.utils import maj_at_k

DEFAULT_EVAL_TYPE: Literal["soft", "hard"] = "soft"
DEFAULT_DATASET = "GSMA/open_telco"
DEFAULT_DATASET_NAME = "telelogs"
DEFAULT_SPLIT = "test"

BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
DIGIT_PATTERN = re.compile(r"\d+")
WHITESPACE_PATTERN = re.compile(r"\n\s*")


def parse_boxed_answer(response: str) -> str:
    r"""Extract content from \boxed{...} in response."""
    if not response:
        return ""
    matches = BOXED_PATTERN.findall(response)
    if not matches:
        return ""
    answer = WHITESPACE_PATTERN.sub("", matches[-1].strip())
    return answer.lstrip(":").rstrip("./")


def extract_first_int(text: str) -> int | None:
    if match := DIGIT_PATTERN.search(text):
        return int(match.group())
    return None


@scorer(metrics=[accuracy(), stderr(), maj_at_k()])
def telelogs_scorer(eval_type: Literal["soft", "hard"] = DEFAULT_EVAL_TYPE):
    async def score(state: TaskState, target: Target) -> Score:
        parsed = parse_boxed_answer(state.output.completion)
        if eval_type == "soft":
            pred_int = extract_first_int(parsed)
            gt_int = extract_first_int(target.text)
            is_correct = pred_int is not None and pred_int == gt_int
        else:
            is_correct = parsed.lower() == target.text.lower()
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=parsed,
            explanation=f"Parsed '{parsed}', target '{target.text}'",
        )

    return score


@task
def telelogs(
    eval_type: Literal["soft", "hard"] = DEFAULT_EVAL_TYPE,
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
        solver=generate(),
        scorer=telelogs_scorer(eval_type=eval_type),
    )
