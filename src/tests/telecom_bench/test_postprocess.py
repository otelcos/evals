from evals.telecom_bench.postprocess import (
    are_json_equal,
    extract_non_reasoning_content,
    multiple_select_postprocess,
    normalize_zh,
    str2json,
)


def test_multiple_select_extracts_sorted_unique_uppercase():
    assert multiple_select_postprocess("the answer is C and A") == "AC"


def test_extract_non_reasoning_strips_think():
    assert extract_non_reasoning_content("<think>x</think>final") == "final"
    assert extract_non_reasoning_content("no tags") == "no tags"


def test_str2json_parses_embedded_object():
    assert str2json('blah {"a": 1} tail') == {"a": 1}


def test_str2json_returns_last_candidate():
    assert str2json('{"a":1} then {"b":2}') == {"b": 2}


def test_str2json_none_on_garbage():
    assert str2json("not json at all") is None


def test_str2json_ignores_brackets_inside_strings():
    # string-state machine must not mis-split on { / [ inside a string value
    assert str2json('{"msg": "[alarm] down {now}"}') == {"msg": "[alarm] down {now}"}


def test_str2json_handles_think_wrapped_and_fenced():
    # candidates inside <think> are scanned too; LAST top-level value wins
    assert str2json('<think>ignore {"a": 1}</think>{"b": 2}') == {"b": 2}
    # fenced JSON still parses: the braces are real candidates even without fence-stripping
    assert str2json('```json\n{"a": 1}\n```') == {"a": 1}


def test_are_json_equal_order_insensitive_list_of_dicts():
    a = [{"x": 1}, {"y": 2}]
    b = [{"y": 2}, {"x": 1}]
    assert are_json_equal(a, b) is True


def test_are_json_equal_detects_difference():
    assert are_json_equal({"a": 1}, {"a": 2}) is False


def test_normalize_zh_folds_fullwidth():
    assert normalize_zh("ＡＢＣ　") == "ABC"
