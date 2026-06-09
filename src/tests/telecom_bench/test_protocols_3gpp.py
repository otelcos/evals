from evals.telecom_bench.comprehension.protocols_3gpp import (
    load_dataset,
    record_to_sample,
)
from evals.telecom_bench.scorers.multiselect_f1 import f1, options_of


def test_record_to_sample_uses_prompt_field():
    rec = {
        "id": 0,
        "题型": "多选题",
        "question": "用途包括？",
        "answer": "A,B,C,D",
        "A": "beamManagement",
        "B": "antennaSwitching",
        "C": "codebook",
        "D": "nonCodebook",
        "difficulty": "中",
        "prompt": "PRERENDERED PROMPT TEXT",
    }
    s = record_to_sample(rec)
    assert s.input == "PRERENDERED PROMPT TEXT"
    assert s.target == "A,B,C,D"


def test_record_to_sample_fallback_to_render_mcq():
    rec = {
        "id": 4,
        "题型": "单选题",
        "question": "5G切换到LTE的IE为？",
        "answer": "A",
        "A": "MobilityFromNRCommand",
        "B": "MobilityToLTECommand",
        "C": "MobilityFrom5GCommand",
        "D": "MobilityTo4GCommand",
        "difficulty": "中",
    }
    s = record_to_sample(rec)
    assert "MobilityFromNRCommand" in s.input
    assert s.target == "A"


def test_golden_record_scores_correct_multiselect():
    # Comma-separated gold "A,B,C,D" — options_of extracts letters correctly
    gold = options_of("A,B,C,D")
    pred = options_of("A,B,C,D")
    assert f1(pred, gold) == 1.0


def test_golden_record_scores_correct_single():
    gold = options_of("A")
    pred = options_of("A")
    assert f1(pred, gold) == 1.0


def test_wrong_record_scores_zero():
    gold = options_of("A,B,C,D")
    # partial overlap gives non-zero f1, so test a completely wrong prediction
    pred_wrong = options_of("E")
    assert f1(pred_wrong, gold) == 0.0


def test_wrong_single_scores_zero():
    gold = options_of("A")
    pred = options_of("B")
    assert f1(pred, gold) == 0.0


def test_load_dataset_count():
    samples = load_dataset()
    assert len(samples) == 36


def test_load_dataset_all_have_target():
    samples = load_dataset()
    for s in samples:
        assert s.target, f"Sample {s.id} has empty target"


def test_load_dataset_all_have_input():
    samples = load_dataset()
    for s in samples:
        assert s.input, f"Sample {s.id} has empty input"
