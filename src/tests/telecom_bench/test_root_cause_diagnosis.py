"""Tests for telecom_bench root_cause_diagnosis set."""

import json

from evals.telecom_bench.application.root_cause_diagnosis import load_dataset
from evals.telecom_bench.scorers.structured_em import judge_correct

# Minimal label structure matching the real label.json shape
GOLD_LABEL = {
    "nodes": [
        {
            "label": "TargetAlarm",
            "properties": {
                "alarmCode": "198097605",
                "alarmtitle": "RRU链路断",
                "_id": "1662777542407",
            },
            "@rid": "Fault_1",
        },
        {
            "label": "RootCause",
            "properties": {
                "causeId": "00000638",
                "causeName": "光纤链路异常",
                "alarmCode": "198097605",
                "alarmtitle": "RRU链路断",
                "_id": "1662777542407",
            },
            "@rid": "RootCause_00000638_Fault_1",
        },
    ]
}

GOLD_STR = json.dumps(GOLD_LABEL, ensure_ascii=False)


def test_load_dataset_returns_one_sample():
    samples = load_dataset()
    assert len(samples) == 1
    s = samples[0]
    assert s.id == "root_cause_diagnosis_0"
    # input must contain nodes and edges from the alarm graph
    input_data = json.loads(s.input)
    assert "nodes" in input_data
    assert "edges" in input_data
    # target must contain the label nodes
    target_data = json.loads(s.target)
    assert "nodes" in target_data
    assert len(target_data["nodes"]) == 2


def test_golden_record_scores_correct():
    assert judge_correct(GOLD_STR, GOLD_STR, mode="json") is True


def test_wrong_record_scores_incorrect():
    wrong = json.dumps({"nodes": []}, ensure_ascii=False)
    assert judge_correct(wrong, GOLD_STR, mode="json") is False


def test_wrong_node_content_scores_incorrect():
    wrong_label = {"nodes": [{"label": "Other", "@rid": "X"}]}
    wrong_str = json.dumps(wrong_label, ensure_ascii=False)
    assert judge_correct(wrong_str, GOLD_STR, mode="json") is False
