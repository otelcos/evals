from evals.telecom_bench.comprehension.basic_knowledge import (
    record_to_sample,
)
from evals.telecom_bench.scorers.multiselect_f1 import f1, options_of


def test_record_to_sample_input_and_target():
    rec = {
        "id": "q_0",
        "question": "以下哪个是正确答案？",
        "A": "选项A",
        "B": "选项B",
        "C": "选项C",
        "D": "选项D",
        "answer": "C",
    }
    s = record_to_sample(rec)
    assert "以下哪个是正确答案？" in s.input
    assert "选项A" in s.input
    assert s.target == "C"


def test_golden_record_scores_correct():
    # Exact match: pred == gold -> f1 == 1.0
    gold = options_of("C")
    pred = options_of("C")
    assert f1(pred, gold) == 1.0


def test_wrong_record_scores_incorrect():
    # No overlap: pred != gold -> f1 == 0.0
    gold = options_of("C")
    pred = options_of("A")
    assert f1(pred, gold) == 0.0
