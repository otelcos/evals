"""Tests for telecom_bench core_network set."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evals.telecom_bench.comprehension.core_network import (
    load_dataset,
    record_to_sample,
)
from evals.telecom_bench.scorers.judge_panel import aggregate, parse_likert

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORD = {
    "难度": "简单题",
    "大类": "安装调试",
    "题目": "Ga口主要和哪些网元进行对接？",
    "答案": "SMF网元和AMF网元",
    "product": "CG",
    "id": 0,
}


# ---------------------------------------------------------------------------
# record_to_sample
# ---------------------------------------------------------------------------


def test_record_to_sample_input():
    s = record_to_sample(SAMPLE_RECORD)
    assert "Ga口" in s.input


def test_record_to_sample_target():
    s = record_to_sample(SAMPLE_RECORD)
    assert s.target == "SMF网元和AMF网元"


def test_record_to_sample_metadata():
    s = record_to_sample(SAMPLE_RECORD)
    assert s.metadata["set"] == "core_network"
    assert s.metadata["product"] == "CG"


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_count():
    samples = load_dataset()
    assert len(samples) == 10


def test_load_dataset_all_have_input_and_target():
    for s in load_dataset():
        assert s.input
        assert s.target


# ---------------------------------------------------------------------------
# parse_likert / aggregate
# ---------------------------------------------------------------------------


def test_parse_likert_valid():
    assert parse_likert("4") == 4


def test_aggregate_normalizes_correctly():
    norm, mean_likert, spread = aggregate([5, 5, 5])
    assert norm == 1.0
    assert mean_likert == 5.0
    assert spread == 0


# ---------------------------------------------------------------------------
# judge_panel_scorer — golden scores > 0, wrong scores 0
# Monkeypatches inspect_ai.model.get_model so tests run fully offline.
# ---------------------------------------------------------------------------


def _make_mock_model(response_text: str) -> MagicMock:
    mock_output = MagicMock()
    mock_output.completion = response_text
    mock_model = MagicMock()
    mock_model.generate = AsyncMock(return_value=mock_output)
    return mock_model


@pytest.mark.asyncio
async def test_golden_response_scores_nonzero():
    """A response identical to the gold gets Likert 5 -> norm 1.0 > 0."""
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from evals.telecom_bench.scorers.judge_panel import judge_panel_scorer

    sample = load_dataset()[0]

    state = MagicMock(spec=TaskState)
    state.input_text = sample.input
    state.output = MagicMock()
    state.output.completion = sample.target  # perfect answer

    target = Target(sample.target)
    mock_model = _make_mock_model("5")

    scorer_fn = judge_panel_scorer(judges=[None], single=True)

    with patch(
        "evals.telecom_bench.scorers.judge_panel.get_model", return_value=mock_model
    ):
        result = await scorer_fn(state, target)

    assert result.value > 0.0


@pytest.mark.asyncio
async def test_wrong_response_scores_zero():
    """A completely wrong response gets Likert 1 -> norm 0.0."""
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from evals.telecom_bench.scorers.judge_panel import judge_panel_scorer

    sample = load_dataset()[0]

    state = MagicMock(spec=TaskState)
    state.input_text = sample.input
    state.output = MagicMock()
    state.output.completion = "我不知道。"

    target = Target(sample.target)
    mock_model = _make_mock_model("1")

    scorer_fn = judge_panel_scorer(judges=[None], single=True)

    with patch(
        "evals.telecom_bench.scorers.judge_panel.get_model", return_value=mock_model
    ):
        result = await scorer_fn(state, target)

    assert result.value == 0.0  # (1 - 1) / 4 == 0.0
