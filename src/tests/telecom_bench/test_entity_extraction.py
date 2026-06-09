from evals.telecom_bench.application.entity_extraction import (
    record_to_sample,
)
from evals.telecom_bench.scorers.structured_em import judge_correct

# Verbatim first record from entity_extraction.json
_GOLD_RECORD = {
    "id": "q_0000",
    "question": "请给出郑州金水机房的网管网设备信息",
    "answer": '{"机房名称":"郑州金水机房","专业":"网管网"}',
}


def test_record_to_sample_input_and_target():
    s = record_to_sample(_GOLD_RECORD)
    assert "郑州金水机房" in s.input
    assert s.target == '{"机房名称":"郑州金水机房","专业":"网管网"}'


def test_golden_record_scores_correct():
    # Exact JSON string matching the gold value should score CORRECT.
    assert (
        judge_correct(
            '{"机房名称":"郑州金水机房","专业":"网管网"}',
            '{"机房名称":"郑州金水机房","专业":"网管网"}',
            mode="json",
        )
        is True
    )


def test_wrong_record_scores_incorrect():
    # Wrong value for 专业 should score INCORRECT.
    assert (
        judge_correct(
            '{"机房名称":"郑州金水机房","专业":"传输网"}',
            '{"机房名称":"郑州金水机房","专业":"网管网"}',
            mode="json",
        )
        is False
    )


def test_missing_key_scores_incorrect():
    # Prediction missing a required key should score INCORRECT.
    assert (
        judge_correct(
            '{"机房名称":"郑州金水机房"}',
            '{"机房名称":"郑州金水机房","专业":"网管网"}',
            mode="json",
        )
        is False
    )
