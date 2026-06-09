from evals.telecom_bench.application.tool_invocation import (
    BOXED_PRE,
    _build_target,
    record_to_sample,
)
from evals.telecom_bench.scorers.structured_em import judge_correct

# Canonical gold values from the data file
_GOLD_EXTRA_INFO = {
    "事件核查结果": "小区存在4G负荷不均衡",
    "一级根因": "覆盖差异大导致不均衡",
    "二级根因": "重叠覆盖度低导致不均衡",
}
_GOLD_TARGET = "小区存在4G负荷不均衡|覆盖差异大导致不均衡|重叠覆盖度低导致不均衡"


def _minimal_record() -> dict:
    return {
        "conversations": [
            {"role": "system", "content": "You are a telecom assistant."},
            {"role": "user", "content": "请分析该小区的负荷问题。"},
            {"role": "assistant", "content": r"\boxed{小区存在4G负荷不均衡}"},
            {"role": "user", "content": "请分析根因。"},
            {"role": "assistant", "content": r"\boxed{覆盖差异大导致不均衡}"},
            {"role": "user", "content": "请分析二级根因。"},
            {"role": "assistant", "content": r"\boxed{重叠覆盖度低导致不均衡}"},
        ],
        "extra_info": _GOLD_EXTRA_INFO,
    }


def test_build_target_joins_in_order():
    assert _build_target(_GOLD_EXTRA_INFO) == _GOLD_TARGET


def test_record_to_sample_input_contains_both_turns():
    rec = _minimal_record()
    s = record_to_sample(rec)
    assert "telecom assistant" in s.input
    assert "负荷问题" in s.input
    assert s.target == _GOLD_TARGET


def test_boxed_pre_extracts_and_joins():
    model_output = (
        r"Some reasoning... \boxed{小区存在4G负荷不均衡} "
        r"more text \boxed{覆盖差异大导致不均衡} "
        r"final \boxed{重叠覆盖度低导致不均衡}"
    )
    result = BOXED_PRE(model_output)
    assert result == _GOLD_TARGET


def test_boxed_pre_returns_empty_when_no_boxed():
    assert BOXED_PRE("no boxed content here") == ""


def test_golden_record_scores_correct():
    # A model output with the correct 3 boxed values scores CORRECT
    model_output = (
        r"\boxed{小区存在4G负荷不均衡} "
        r"\boxed{覆盖差异大导致不均衡} "
        r"\boxed{重叠覆盖度低导致不均衡}"
    )
    assert (
        judge_correct(model_output, _GOLD_TARGET, mode="exact", pre=BOXED_PRE) is True
    )


def test_wrong_record_scores_incorrect():
    # Wrong first boxed value
    model_output = (
        r"\boxed{小区不存在负荷问题} "
        r"\boxed{覆盖差异大导致不均衡} "
        r"\boxed{重叠覆盖度低导致不均衡}"
    )
    assert (
        judge_correct(model_output, _GOLD_TARGET, mode="exact", pre=BOXED_PRE) is False
    )


def test_wrong_order_scores_incorrect():
    # Same values but wrong order also fails exact match
    model_output = (
        r"\boxed{覆盖差异大导致不均衡} "
        r"\boxed{小区存在4G负荷不均衡} "
        r"\boxed{重叠覆盖度低导致不均衡}"
    )
    assert (
        judge_correct(model_output, _GOLD_TARGET, mode="exact", pre=BOXED_PRE) is False
    )


def test_missing_boxed_scores_incorrect():
    assert judge_correct("无法判断", _GOLD_TARGET, mode="exact", pre=BOXED_PRE) is False
