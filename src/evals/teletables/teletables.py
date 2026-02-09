import hashlib

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from evals._utils import resolve_dataset

DEFAULT_DATASET = "GSMA/ot_sample_data"
DEFAULT_DATASET_NAME = "teletables"
DEFAULT_SPLIT = "test"

R18_DATASET = "netop/TeleTables"
R18_38_SERIES_SPECS = {"38101-1", "38211", "38212", "38213"}


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


def _extract_base_spec(document_id: str) -> str:
    """Extract base spec number: '38211-i60' -> '38211', '38101-1-j10' -> '38101-1'."""
    parts = document_id.split("-")
    base_parts = []
    for p in parts:
        if p and p[0].isalpha():
            break
        base_parts.append(p)
    return "-".join(base_parts)


def _record_to_sample_r18(record: dict) -> Sample:
    """Convert a netop/TeleTables record into MCQ format with document_id metadata."""
    question_hash = hashlib.md5(record["question"].encode()).hexdigest()[:6]
    sample_id = f"{record['table_id']}_{question_hash}"

    return Sample(
        id=sample_id,
        input=record["question"],
        choices=record["choices"],
        target=chr(65 + record["answer"]),
        metadata={"document_id": record["document_id"]},
    )


@task
def teletables_r18(
    dataset_path: str = R18_DATASET,
    split: str = DEFAULT_SPLIT,
) -> Task:
    """MCQ TeleTables filtered to Rel-18 38-series specs only.

    Uses netop/TeleTables (same source as teletables_agentic_r18) to
    ensure identical samples, enabling fair MCQ vs agentic comparison.
    """
    dataset = hf_dataset(
        dataset_path,
        sample_fields=_record_to_sample_r18,
        split=split,
    )

    dataset = [
        s
        for s in dataset
        if _extract_base_spec(s.metadata["document_id"]) in R18_38_SERIES_SPECS
    ]

    return Task(
        dataset=dataset,
        solver=multiple_choice(cot=False),
        scorer=choice(),
    )
