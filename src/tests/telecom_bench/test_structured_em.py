from evals.telecom_bench.scorers.structured_em import judge_correct


def test_json_mode_equal():
    assert judge_correct('{"a": 1}', '{"a": 1}', mode="json") is True


def test_json_mode_unequal():
    assert judge_correct('{"a": 2}', '{"a": 1}', mode="json") is False


def test_exact_mode_normalizes():
    assert judge_correct("ＤＯＮＥ", "DONE", mode="exact") is True


def test_pre_callable_applied():
    pred = "Thought: x\nOutput: DONE\nThought: y"
    assert (
        judge_correct(
            pred,
            "DONE",
            mode="exact",
            pre=lambda t: t.split("Output:")[-1].split("\nThought:")[0],
        )
        is True
    )
