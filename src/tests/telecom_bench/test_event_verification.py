"""Tests for telecom_bench event_verification set."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evals.telecom_bench.application.event_verification import (
    load_dataset,
    record_to_sample,
)
from evals.telecom_bench.scorers.judge_panel import aggregate, parse_likert

# ---------------------------------------------------------------------------
# record_to_sample
# ---------------------------------------------------------------------------

BEST_ANSWER = [
    {
        "source_ishighloadcell": "TRUE",
        "highload_time": ["2025-08-01 01:00:00"],
        "target": {"subnet_id": "3315"},
        "load_unbalance_result": {"result": 1},
    }
]

SAMPLE_RECORD = {
    "question": "请分析小区负荷不均衡情况。",
    "best_answer": BEST_ANSWER,
}


def test_record_to_sample_input():
    s = record_to_sample(SAMPLE_RECORD)
    assert "负荷不均衡" in s.input


def test_record_to_sample_target_is_json_dumps():
    s = record_to_sample(SAMPLE_RECORD)
    parsed = json.loads(s.target)
    assert isinstance(parsed, list)
    assert parsed[0]["source_ishighloadcell"] == "TRUE"


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_returns_one_sample():
    samples = load_dataset()
    assert len(samples) == 1


def test_load_dataset_sample_has_target():
    samples = load_dataset()
    parsed = json.loads(samples[0].target)
    assert isinstance(parsed, list)
    assert len(parsed) == 1


# ---------------------------------------------------------------------------
# parse_likert / aggregate (reused from judge_panel, exercised here for coverage)
# ---------------------------------------------------------------------------


def test_parse_likert_valid():
    assert parse_likert("3") == 3


def test_aggregate_normalizes_correctly():
    norm, mean_likert, spread = aggregate([5, 5, 5])
    assert norm == 1.0
    assert mean_likert == 5.0
    assert spread == 0


# ---------------------------------------------------------------------------
# judge_panel_scorer — golden response scores > 0, wrong scores 0
# These tests monkeypatch inspect_ai.model.get_model so no network is needed.
# ---------------------------------------------------------------------------


def _make_mock_model(response_text: str) -> MagicMock:
    """Return a mock model whose .generate() returns the given text."""
    mock_output = MagicMock()
    mock_output.completion = response_text

    mock_model = MagicMock()
    mock_model.generate = AsyncMock(return_value=mock_output)
    return mock_model


@pytest.mark.asyncio
async def test_golden_response_scores_nonzero():
    """A response that matches the gold should get a Likert score > 1 (norm > 0)."""
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from evals.telecom_bench.scorers.judge_panel import judge_panel_scorer

    sample = load_dataset()[0]

    state = MagicMock(spec=TaskState)
    state.input_text = sample.input
    state.output = MagicMock()
    state.output.completion = sample.target  # perfect answer = gold itself

    target = Target(sample.target)

    mock_model = _make_mock_model("5")  # judge says 5/5

    scorer_fn = judge_panel_scorer(judges=[None], single=True)

    with patch(
        "evals.telecom_bench.scorers.judge_panel.get_model", return_value=mock_model
    ):
        result = await scorer_fn(state, target)

    assert result.value > 0.0


@pytest.mark.asyncio
async def test_wrong_response_scores_zero():
    """A completely wrong response should get Likert 1 (norm == 0.0)."""
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from evals.telecom_bench.scorers.judge_panel import judge_panel_scorer

    sample = load_dataset()[0]

    state = MagicMock(spec=TaskState)
    state.input_text = sample.input
    state.output = MagicMock()
    state.output.completion = "我不知道。"  # clearly wrong

    target = Target(sample.target)

    mock_model = _make_mock_model("1")  # judge says 1/5

    scorer_fn = judge_panel_scorer(judges=[None], single=True)

    with patch(
        "evals.telecom_bench.scorers.judge_panel.get_model", return_value=mock_model
    ):
        result = await scorer_fn(state, target)

    assert result.value == 0.0  # (1-1)/4 == 0.0
