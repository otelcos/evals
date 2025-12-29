from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import pattern
from inspect_ai.solver import generate

DEFAULT_DATASET = "GSMA/open_telco"
DEFAULT_DATASET_NAME = "3gpp_tsg"
DEFAULT_SPLIT = "test"

WG_PATTERN = r"([A-Z]+\d+(?:-[A-Z]+)?)"


@task
def three_gpp(
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
        scorer=pattern(WG_PATTERN, ignore_case=True),
    )
