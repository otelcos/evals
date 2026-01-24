import hashlib

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

DEFAULT_DATASET = "GSMA/open_telco"
DEFAULT_DATASET_NAME = "teletables"
DEFAULT_SPLIT = "test"


def record_to_sample(record: dict) -> Sample:
    """Convert dataset into MCQ format."""
    # Create stable ID from table_id and question hash
    question_hash = hashlib.md5(record["question"].encode()).hexdigest()[:6]
    sample_id = f"{record['table_id']}_{question_hash}"

    return Sample(
        id=sample_id,
        input=record["question"],
        choices=record["choices"],
        target=chr(65 + record["answer"]),
    )


@task
def teletables(
    dataset_path: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
) -> Task:
    dataset = hf_dataset(
        dataset_path,
        name=DEFAULT_DATASET_NAME,
        sample_fields=record_to_sample,
        split=split,
    )
    return Task(
        dataset=dataset,
        solver=multiple_choice(cot=False),
        scorer=choice(),
    )
