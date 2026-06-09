from evals.telecom_bench.scorers.multiselect_f1 import f1, options_of


def test_options_of_extracts_letters():
    assert options_of("AC") == {"A", "C"}


def test_f1_perfect():
    assert f1({"A", "C"}, {"A", "C"}) == 1.0


def test_f1_partial():
    # pred {A,B} vs gold {A,C}: p=0.5, r=0.5 -> F1=0.5
    assert f1({"A", "B"}, {"A", "C"}) == 0.5


def test_f1_no_overlap():
    assert f1({"B"}, {"A", "C"}) == 0.0
