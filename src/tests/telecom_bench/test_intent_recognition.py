from evals.telecom_bench.application.intent_recognition import (
    INTENT_PRE,
    record_to_sample,
)
from evals.telecom_bench.scorers.structured_em import judge_correct


def test_record_to_sample_input_and_target():
    rec = {"id": "q_0000", "input": "请改善黄家庄村的高负荷问题", "output": "DONE"}
    s = record_to_sample(rec)
    assert "黄家庄村" in s.input
    assert s.target == "DONE"


def test_intent_pre_extracts_output_segment():
    assert INTENT_PRE("Thought: a\nOutput: NO\nThought: b") == "NO"


def test_golden_record_scores_correct():
    # mode="exact" + INTENT_PRE: a clean class label matches the gold
    assert judge_correct("DONE", "DONE", mode="exact", pre=INTENT_PRE) is True


def test_wrong_record_scores_incorrect():
    assert judge_correct("NO", "DONE", mode="exact", pre=INTENT_PRE) is False
