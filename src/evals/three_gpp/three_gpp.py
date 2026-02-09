from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import pattern
from inspect_ai.solver import generate

from evals._utils import resolve_dataset

DEFAULT_DATASET = "GSMA/ot_sample_data"
DEFAULT_DATASET_NAME = "3gpp_tsg"
DEFAULT_SPLIT = "test"

WG_PATTERN = r"([A-Z]+\d+(?:-[A-Z]+)?)"


@task
def three_gpp(
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
        solver=generate(),
        scorer=pattern(WG_PATTERN, ignore_case=True),
    )
