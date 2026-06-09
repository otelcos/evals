import json
import re
from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


def scene_postprocessor(text: str):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return text


def intent_postprocessor(text: str):
    match = re.search(r'\b(true|false)\b', text, re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()
    else:
        return text


@ICL_EVALUATORS.register_module()
class UMEGuzhangSceneEvaluator(BaseEvaluator):

    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref in zip(predictions, references):
            try:
                pred = json.loads(pred)
            except Exception:
                pred = None
            try:
                ref = json.loads(ref)
            except Exception:
                ref = None

            if pred is None or ref is None:
                correct = False
            else:
                # 必须所有key和value都完全一致
                correct = pred == ref

            is_correct.append(correct)
            processed_pred.append(pred)
            processed_gold.append(ref)

        accuracy = sum(is_correct) / len(is_correct) if is_correct else 0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": is_correct
            }
        }


@ICL_EVALUATORS.register_module()
class UMEGuzhangIntentEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()

    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref in zip(predictions, references):
            correct = False
            # 统一大小写和去除空格
            ref = ref.strip().upper()
            if pred == ref:
                correct = True

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
