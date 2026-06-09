"""Tests for telecom_bench_wireless_network (multiselect_f1 scorer)."""

from evals.telecom_bench.comprehension.wireless_network import (
    _render_fault,
    _render_optim,
    load_dataset,
)
from evals.telecom_bench.scorers.multiselect_f1 import f1, options_of

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def test_render_fault_includes_question_and_options():
    rec = {
        "id": "1",
        "question": "基站故障排查首选步骤？",
        "options": ["A. 重启设备", "B. 检查日志", "C. 联系厂商", "D. 更换硬件"],
        "answer": "B",
    }
    rendered = _render_fault(rec)
    assert "基站故障排查首选步骤" in rendered
    assert "A. 重启设备" in rendered
    assert "D. 更换硬件" in rendered


def test_render_optim_includes_stem_and_kv_options():
    rec = {
        "id": "2",
        "stem": "以下哪项属于网络优化目标？",
        "options": {"A": "提升覆盖率", "B": "降低成本", "C": "增加用户数"},
        "correct_answers": ["A", "C"],
    }
    rendered = _render_optim(rec)
    assert "以下哪项属于网络优化目标" in rendered
    assert "A. 提升覆盖率" in rendered
    assert "C. 增加用户数" in rendered


# ---------------------------------------------------------------------------
# load_dataset smoke test
# ---------------------------------------------------------------------------


def test_load_dataset_returns_expected_count():
    samples = load_dataset()
    # 33 fault + 33 optimization = 66
    assert len(samples) == 66


def test_load_dataset_sources_present():
    samples = load_dataset()
    sources = {s.metadata["source"] for s in samples}
    assert "fault_maintenance" in sources
    assert "network_optimization" in sources


# ---------------------------------------------------------------------------
# Golden-record test: gold answer scores CORRECT (f1 = 1.0)
# ---------------------------------------------------------------------------


def test_golden_single_answer_scores_correct():
    # Single-letter gold (fault_maintenance style)
    pred = options_of("C")
    gold = options_of("C")
    assert f1(pred, gold) == 1.0


def test_golden_multi_answer_scores_correct():
    # Multi-letter gold (network_optimization style, e.g. ["D", "E"] -> "DE")
    pred = options_of("DE")
    gold = options_of("DE")
    assert f1(pred, gold) == 1.0


# ---------------------------------------------------------------------------
# Known-wrong test: wrong answer scores 0
# ---------------------------------------------------------------------------


def test_wrong_answer_scores_zero():
    pred = options_of("A")
    gold = options_of("C")
    assert f1(pred, gold) == 0.0


def test_wrong_multi_answer_scores_zero():
    pred = options_of("AB")
    gold = options_of("DE")
    assert f1(pred, gold) == 0.0


# ---------------------------------------------------------------------------
# Partial credit (macro-F1 non-trivial)
# ---------------------------------------------------------------------------


def test_partial_overlap_gives_nonzero_f1():
    # pred={A,B}, gold={A,C} -> tp=1, prec=0.5, rec=0.5 -> f1=0.5
    pred = options_of("AB")
    gold = options_of("AC")
    score = f1(pred, gold)
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Sample shape
# ---------------------------------------------------------------------------


def test_sample_has_required_fields():
    samples = load_dataset()
    s = samples[0]
    assert s.input
    assert s.target
    assert s.metadata["set"] == "wireless_network"
