import json
import re
from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import str2json


@ICL_EVALUATORS.register_module()
class IDAIntentRecEvaluator(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref in zip(predictions, references):
            pred = pred.split("Output:")[-1].split("\nThought:")[0].strip()
            pred = str2json(pred)
            ref = str2json(ref)
            if pred is None or ref is None:
                is_correct.append(False)
                processed_pred.append(pred)
                processed_gold.append(ref)
                continue

            correct = self._check_dict(ref, pred)
            is_correct.append(correct)
            processed_pred.append(pred)
            processed_gold.append(ref)
        accuracy = sum(is_correct) / len(is_correct) if is_correct else 0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": is_correct,
            },
        }

    def _check_dict(self, expected_answer, llm_result_answer):
        instance_result = True
        if expected_answer == llm_result_answer:
            instance_result = True
        else:
            instance_result = False
        return instance_result
