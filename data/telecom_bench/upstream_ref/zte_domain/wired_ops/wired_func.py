from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


@ICL_EVALUATORS.register_module()
class WiredFuncFindEvaluator(BaseEvaluator):

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred: List[str] = []
        processed_gold: List[str] = []
        is_correct: List[bool] = []

        for pred, ref in zip(predictions, references):

            correct = pred == ref

            processed_pred.append(pred)
            processed_gold.append(ref)
            is_correct.append(correct)

        accuracy = (sum(is_correct) / len(is_correct) * 100) if is_correct else 0.0

        return {
            "accuracy": accuracy,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": is_correct,
            },
        }
