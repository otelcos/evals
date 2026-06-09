from evals.telecom_bench.comprehension.network_5g import (
    _normalize_tf,
    load_dataset,
    record_to_sample,
    render_5g_mcq,
)
from evals.telecom_bench.scorers.multiselect_f1 import f1, options_of

# ---------------------------------------------------------------------------
# Unit tests for render_5g_mcq
# ---------------------------------------------------------------------------


def test_render_skips_none_options():
    rec = {
        "question": "Is this correct?",
        "A": "正确",
        "B": "错误",
        "C": "None",
        "D": "None",
    }
    rendered = render_5g_mcq(rec)
    assert "C." not in rendered
    assert "D." not in rendered
    assert "A. 正确" in rendered
    assert "B. 错误" in rendered


def test_render_includes_all_present_options():
    rec = {
        "question": "Which?",
        "A": "opt1",
        "B": "opt2",
        "C": "opt3",
        "D": "opt4",
    }
    rendered = render_5g_mcq(rec)
    assert "A. opt1" in rendered
    assert "D. opt4" in rendered


# ---------------------------------------------------------------------------
# Unit tests for T/F normalization
# ---------------------------------------------------------------------------


def test_normalize_T_maps_to_A():
    rec = {"answer": "T", "A": "正确", "B": "错误", "C": "None", "D": "None"}
    assert _normalize_tf(rec) == "A"


def test_normalize_F_maps_to_B():
    rec = {"answer": "F", "A": "正确", "B": "错误", "C": "None", "D": "None"}
    assert _normalize_tf(rec) == "B"


def test_normalize_passthrough_for_regular_answer():
    rec = {"answer": "C", "A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"}
    assert _normalize_tf(rec) == "C"


def test_normalize_multi_select_passthrough():
    rec = {"answer": "ABCD", "A": "a", "B": "b", "C": "c", "D": "d"}
    assert _normalize_tf(rec) == "ABCD"


# ---------------------------------------------------------------------------
# record_to_sample: shape checks
# ---------------------------------------------------------------------------


def test_record_to_sample_regular_mcq():
    rec = {
        "id": 1,
        "source_file": "5G基础_单选题",
        "question": "NR系统中，一个SS/PBCH block包含（ ）个OFDM symbols。",
        "A": "4",
        "B": "1",
        "C": "3",
        "D": "2",
        "answer": "A",
    }
    s = record_to_sample(rec)
    assert s.target == "A"
    assert "NR系统中" in s.input
    assert s.metadata["raw_answer"] == "A"


def test_record_to_sample_true_false():
    rec = {
        "id": 0,
        "source_file": "5G基础_判断题",
        "question": "多模光纤的传输距离长于单模光纤",
        "A": "正确",
        "B": "错误",
        "C": "None",
        "D": "None",
        "answer": "F",
    }
    s = record_to_sample(rec)
    # F -> B (错误)
    assert s.target == "B"
    assert s.metadata["raw_answer"] == "F"
    # C and D should not appear in the rendered input
    assert "C." not in s.input
    assert "D." not in s.input


# ---------------------------------------------------------------------------
# Scoring: golden record scores CORRECT (f1==1.0)
# ---------------------------------------------------------------------------


def test_golden_single_select_scores_correct():
    # pred == gold -> f1 == 1.0, exact match
    gold = options_of("A")
    pred = options_of("A")
    assert f1(pred, gold) == 1.0


def test_golden_multi_select_scores_correct():
    gold = options_of("ABCD")
    pred = options_of("ABCD")
    assert f1(pred, gold) == 1.0


def test_golden_tf_normalized_scores_correct():
    # After T->A normalization, gold is "A"; model predicts "A"
    gold = options_of("A")
    pred = options_of("A")
    assert f1(pred, gold) == 1.0


# ---------------------------------------------------------------------------
# Scoring: wrong answer scores 0
# ---------------------------------------------------------------------------


def test_wrong_answer_scores_zero():
    gold = options_of("A")
    pred = options_of("B")
    assert f1(pred, gold) == 0.0


def test_wrong_multi_select_scores_zero():
    gold = options_of("ABC")
    pred = options_of("D")
    assert f1(pred, gold) == 0.0


def test_unmapped_tf_scores_zero():
    # Without T/F normalization: gold would be {"F"}, pred {"A"} -> 0
    gold = options_of("F")  # raw, un-normalized
    pred = options_of("A")  # model's answer to 判断题
    assert f1(pred, gold) == 0.0


# ---------------------------------------------------------------------------
# Integration: load_dataset returns 23 samples
# ---------------------------------------------------------------------------


def test_load_dataset_returns_23_samples():
    samples = load_dataset()
    assert len(samples) == 23


def test_load_dataset_all_have_target():
    samples = load_dataset()
    for s in samples:
        assert s.target, f"Sample {s.id} has empty target"


def test_load_dataset_no_raw_tf_in_targets():
    """After normalization, no sample should have 'T' or 'F' as its target."""
    samples = load_dataset()
    for s in samples:
        assert s.target not in ("T", "F"), (
            f"Sample {s.id} still has raw T/F target: {s.target!r}"
        )


def test_load_dataset_no_none_in_inputs():
    """Rendered inputs must not contain the literal string 'C. None' or 'D. None'."""
    samples = load_dataset()
    for s in samples:
        assert "C. None" not in s.input, f"Sample {s.id} input contains 'C. None'"
        assert "D. None" not in s.input, f"Sample {s.id} input contains 'D. None'"
