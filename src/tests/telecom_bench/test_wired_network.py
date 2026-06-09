from evals.telecom_bench.comprehension.wired_network import (
    record_to_sample,
    render_wired,
)
from evals.telecom_bench.scorers.multiselect_f1 import f1, options_of

# --- record helpers ---


def _make_single_record() -> dict:
    return {
        "id": 0,
        "单项选择题": {
            "问题": "在VPLS实例为qualified模式时，PW封装的默认模式是什么？",
            "选项": [
                "A. untagged模式",
                "B. tagged模式",
                "C. raw模式",
                "D. qualified模式",
            ],
            "答案": "B",
        },
    }


def _make_multi_record() -> dict:
    return {
        "id": 4,
        "多项选择题": {
            "问题": "以下哪些是DHCPv6 server防止地址或前缀冲突的措施？",
            "选项": [
                "A. 选项A内容",
                "B. 选项B内容",
                "C. 选项C内容",
                "D. 选项D内容",
            ],
            "答案": "A, B",
        },
    }


# --- record_to_sample tests ---


def test_record_to_sample_single():
    rec = _make_single_record()
    s = record_to_sample(rec)
    assert "VPLS" in s.input
    assert "tagged模式" in s.input
    assert s.target == "B"
    assert s.id == "0"


def test_record_to_sample_multi():
    rec = _make_multi_record()
    s = record_to_sample(rec)
    assert "DHCPv6" in s.input
    assert s.target == "A, B"


def test_render_wired_joins_options():
    inner = {
        "问题": "测试问题",
        "选项": ["A. 选项A", "B. 选项B"],
        "答案": "A",
    }
    rendered = render_wired(inner)
    assert rendered == "测试问题\nA. 选项A\nB. 选项B"


# --- scorer tests (pure helpers, no inspect_ai runtime) ---


def test_golden_single_scores_correct():
    # Single-select: exact match -> f1 == 1.0
    gold = options_of("B")
    pred = options_of("B")
    assert f1(pred, gold) == 1.0


def test_wrong_single_scores_incorrect():
    # Wrong answer -> f1 == 0.0
    gold = options_of("B")
    pred = options_of("A")
    assert f1(pred, gold) == 0.0


def test_golden_multi_scores_correct():
    # Multi-select: "A, B" parses correctly and exact match gives f1 == 1.0
    gold = options_of("A, B")
    pred = options_of("A, B")
    assert f1(pred, gold) == 1.0


def test_partial_multi_scores_partial():
    # Partial overlap: pred={A}, gold={A,B} -> recall=0.5, precision=1 -> f1=2/3
    gold = options_of("A, B")
    pred = options_of("A")
    score = f1(pred, gold)
    assert 0.0 < score < 1.0


def test_wrong_multi_scores_zero():
    # No overlap at all
    gold = options_of("A, B")
    pred = options_of("C")
    assert f1(pred, gold) == 0.0
