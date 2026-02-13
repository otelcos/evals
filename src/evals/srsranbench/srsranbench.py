from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from evals._utils import resolve_dataset

DEFAULT_DATASET = "GSMA/ot_sample_data"
DEFAULT_DATASET_NAME = "srsranbench"
DEFAULT_SPLIT = "test"


def record_to_sample(record: dict) -> Sample:
    """Convert dataset record to Sample."""
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=chr(65 + record["answer"]),
    )


@task
def srsranbench(
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
    return Task(
        dataset=dataset,
        solver=multiple_choice(cot=False),
        scorer=choice(),
    )
