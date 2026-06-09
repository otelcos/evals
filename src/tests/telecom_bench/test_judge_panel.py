from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evals.telecom_bench.scorers.judge_panel import (
    aggregate,
    judge_panel_scorer,
    parse_likert,
)


def test_parse_likert_finds_digit():
    assert parse_likert("评分：4") == 4
    assert parse_likert("the score is 5/5") == 5


def test_parse_likert_none_when_absent():
    assert parse_likert("no number here") is None


def test_aggregate_mean_and_spread():
    norm, mean_likert, spread = aggregate([5, 3, 4])
    assert mean_likert == 4.0
    assert norm == 0.75  # (4-1)/4
    assert spread == 2


def _state_and_target(completion: str, reference: str):
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    state = TaskState(
        model="mock/model",
        sample_id=0,
        epoch=1,
        input="问题",
        messages=[],
        output=MagicMock(completion=completion),
    )
    return state, Target(reference)


@pytest.mark.asyncio
async def test_panel_averages_three_judges():
    """A three-judge panel averages the parsed Likert scores and normalizes."""
    state, target = _state_and_target("模型回答", "参考答案")
    outs = [MagicMock(completion=c) for c in ("5", "3", "4")]
    mock_model = MagicMock()
    mock_model.generate = AsyncMock(side_effect=outs)
    with patch(
        "evals.telecom_bench.scorers.judge_panel.get_model", return_value=mock_model
    ):
        score = await judge_panel_scorer(judges=[None, None, None])(state, target)
    assert score.metadata["likert_mean"] == 4.0
    assert score.value == pytest.approx((4.0 - 1) / 4)
    assert score.metadata["panel"] == [5, 3, 4]


@pytest.mark.asyncio
async def test_no_parseable_judge_score_returns_zero():
    """If no judge emits a 1-5 digit, the scorer returns 0.0, not a crash."""
    state, target = _state_and_target("模型回答", "参考答案")
    mock_model = MagicMock()
    mock_model.generate = AsyncMock(return_value=MagicMock(completion="无法评分"))
    with patch(
        "evals.telecom_bench.scorers.judge_panel.get_model", return_value=mock_model
    ):
        score = await judge_panel_scorer(single=True)(state, target)
    assert score.value == 0.0
    assert score.metadata["likert_mean"] is None
