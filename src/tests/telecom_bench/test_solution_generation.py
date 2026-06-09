"""Tests for telecom_bench solution_generation tasks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evals.telecom_bench.application.solution_generation import (
    SOLUTION_PRE,
    load_dataset,
    record_to_sample,
)
from evals.telecom_bench.scorers.judge_panel import aggregate, parse_likert
from evals.telecom_bench.scorers.structured_em import judge_correct

# A minimal golden record extracted from the real data file.
_GOLDEN_RECORD = {
    "question": "NR基站发生超级小区CP退服告警,提示请在RANCLI中使用eac defragmentIQ命令执行IQ碎片清理",
    "best_answer": "step1.使用[IQ碎片清理]执行&defragmentIQ&命令step2.使用[告警恢复判断]观察故障是否恢复  step3.若未恢复，使用[通知人工处理]联系技术支持",
}


# ---------------------------------------------------------------------------
# record_to_sample
# ---------------------------------------------------------------------------


def test_record_to_sample_input_and_target():
    # Default (judged variant): target is the full prose gold.
    s = record_to_sample(_GOLDEN_RECORD)
    assert "defragmentIQ" in s.input
    assert s.target == _GOLDEN_RECORD["best_answer"]


def test_record_to_sample_extract_steps_reduces_gold_to_tool_steps():
    # Plain variant: target is the bracketed tool-step sequence (upstream-faithful).
    s = record_to_sample(_GOLDEN_RECORD, extract_steps=True)
    assert s.target == "IQ碎片清理|告警恢复判断|通知人工处理"


def test_solution_pre_extracts_bracketed_steps():
    assert SOLUTION_PRE("step1.[A]做事 step2.[B]检查") == "A|B"
    assert SOLUTION_PRE("no brackets here") == ""


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_returns_expected_count():
    samples = load_dataset()
    assert len(samples) == 5


def test_load_dataset_skips_missing_gold(caplog):
    """Records without best_answer are filtered and a warning is emitted."""
    fake_records = [
        {"question": "q1", "best_answer": "step1.a"},
        {"question": "q2"},  # missing gold
    ]
    with patch(
        "evals.telecom_bench.application.solution_generation.load_json",
        return_value=fake_records,
    ):
        import logging

        with caplog.at_level(
            logging.WARNING,
            logger="evals.telecom_bench.application.solution_generation",
        ):
            samples = load_dataset()
    assert len(samples) == 1
    assert "skipped 1" in caplog.text


# ---------------------------------------------------------------------------
# structured_em (exact mode) scorer helpers
# ---------------------------------------------------------------------------


def test_golden_record_scores_correct():
    # The plain task scores on extracted tool steps (SOLUTION_PRE on both sides).
    target = SOLUTION_PRE(_GOLDEN_RECORD["best_answer"])  # the stored gold target
    # A model output with the SAME tool steps but different prose still matches.
    model_out = "首先[IQ碎片清理]，然后[告警恢复判断]，最后如未恢复[通知人工处理]。"
    assert judge_correct(model_out, target, mode="exact", pre=SOLUTION_PRE) is True


def test_wrong_record_scores_incorrect():
    target = SOLUTION_PRE(_GOLDEN_RECORD["best_answer"])
    wrong = "step1. 使用[基站复位]执行&resetSystem&命令"
    assert judge_correct(wrong, target, mode="exact", pre=SOLUTION_PRE) is False


# ---------------------------------------------------------------------------
# judge_panel scorer helpers (offline)
# ---------------------------------------------------------------------------


def test_parse_likert_extracts_integer():
    assert parse_likert("4") == 4
    assert parse_likert("score: 3 points") == 3
    assert parse_likert("no digit here") is None


def test_aggregate_normalizes_correctly():
    norm, mean_l, spread = aggregate([4, 4, 4])
    assert norm == pytest.approx((4 - 1) / 4)
    assert mean_l == pytest.approx(4.0)
    assert spread == 0


def test_aggregate_spread():
    norm, mean_l, spread = aggregate([2, 4])
    assert spread == 2


@pytest.mark.asyncio
async def test_judge_panel_scorer_offline():
    """judge_panel_scorer runs offline with a mocked get_model."""
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from evals.telecom_bench.scorers.judge_panel import judge_panel_scorer

    # Build a minimal TaskState.
    sample = Sample(
        input=_GOLDEN_RECORD["question"], target=_GOLDEN_RECORD["best_answer"]
    )
    state = TaskState(
        model="mock/model",
        sample_id=0,
        epoch=1,
        input=sample.input,
        messages=[],
        output=MagicMock(completion=_GOLDEN_RECORD["best_answer"]),
    )
    target = Target(_GOLDEN_RECORD["best_answer"])

    # Mock get_model to return a model whose generate returns "5".
    mock_output = MagicMock()
    mock_output.completion = "5"
    mock_model = MagicMock()
    mock_model.generate = AsyncMock(return_value=mock_output)

    with patch(
        "evals.telecom_bench.scorers.judge_panel.get_model", return_value=mock_model
    ):
        scorer_fn = judge_panel_scorer(single=True)
        score = await scorer_fn(state, target)

    assert score.value == pytest.approx((5 - 1) / 4)
    assert score.metadata["likert_mean"] == 5.0
