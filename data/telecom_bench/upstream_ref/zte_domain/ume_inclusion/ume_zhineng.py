import json
from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import str2json


# 字符串判断
@ICL_EVALUATORS.register_module()
class UMEZhinengEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()

    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref in zip(predictions, references):
            pred = str2json(pred)
            ref = str2json(ref)
            correct = True

            if pred is None or ref is None:
                correct = False
            else:
                for key in ref:
                    if key not in pred:
                        correct = False
                        break

                    ref_val = ref[key]
                    pred_val = pred[key]

                    if pred_val != ref_val:
                        correct = False
                        break

            is_correct.append(correct)
            processed_pred.append(pred)
            processed_gold.append(ref)

        accuracy = sum(is_correct) / len(is_correct) if len(is_correct) > 0 else 0.0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": is_correct
            }
        }
