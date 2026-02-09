from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from evals._utils import resolve_dataset

DEFAULT_SUBJECT = "full"
DEFAULT_DATASET = "GSMA/ot_sample_data"
DEFAULT_DATASET_NAME = "teleqna"
DEFAULT_SPLIT = "test"


def record_to_sample(record: dict) -> Sample:
    """Convert dataset record to Sample with subject metadata."""
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=chr(65 + record["answer"]),
        metadata={"subject": record.get("subject")},
    )


@task
def teleqna(
    subject: str = DEFAULT_SUBJECT,
    dataset_path: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    full: bool = False,
) -> Task:
    ds_path, ds_split = resolve_dataset(full, dataset_path, DEFAULT_DATASET, split)
    dataset = hf_dataset(
        ds_path,
        name=DEFAULT_DATASET_NAME,
        sample_fields=record_to_sample,
        split=ds_split,
    )
    if subject != DEFAULT_SUBJECT:
        dataset = dataset.filter(
            lambda sample: sample.metadata is not None
            and sample.metadata.get("subject") == subject
        )
    return Task(
        dataset=dataset,
        solver=multiple_choice(cot=False),
        scorer=choice(),
    )
