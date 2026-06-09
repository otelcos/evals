import re
from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import str2json, json_str


def pass_postprocessor1(text: str) -> str:
    return text


def str_postprocessor2(text: str) -> str:
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)


def str_postprocessor3(text: str) -> str:
    return re.sub(r'(?i)Action:\s*(.*?)\s*Action\s+Input:', '', text)


# 实体抽取判断
@ICL_EVALUATORS.register_module()
class UMEWangyouEvaluator1(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
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

                    if isinstance(ref_val, list):
                        if sorted(pred_val) != sorted(ref_val):
                            correct = False
                            break
                    else:
                        if pred_val != ref_val:
                            correct = False
                            break

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


# 字符串判断
@ICL_EVALUATORS.register_module()
class UMEWangyouEvaluator2(BaseEvaluator):
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
            pred = str_postprocessor2(pred)
            ref = str_postprocessor2(ref)
            if pred.upper() == ref.upper():
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


# action字符串判断
@ICL_EVALUATORS.register_module()
class UMEWangyouEvaluator3(BaseEvaluator):
    def __init__(self):
        super().__init__()

    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred = []
        processed_gold = []
        is_correct = []
        keywords = ['thought:', 'action:', 'action input:']

        for pred, ref in zip(predictions, references):
            correct = True
            if pred is None or ref is None:
                correct = False

            elif not all(keyword in pred.lower() for keyword in keywords):
                correct = False

            else:
                pred_action = str_postprocessor3(pred)
                ref_action = str_postprocessor3(ref)

                if pred_action != ref_action:
                    correct = False

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


# 实体抽取判断
@ICL_EVALUATORS.register_module()
class UMEWangyouEvaluator4(BaseEvaluator):
    def __init__(self):
        super().__init__()

    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref in zip(predictions, references):
            correct = True

            if pred is None or ref is None:
                correct = False
            else:
                pred, ref = map(str2json, map(json_str, [pred, ref]))
                for key in ref:
                    if key not in pred:
                        correct = False
                        break

                    ref_val = ref[key]
                    pred_val = pred[key]

                    # 处理None，避免判断sub str报错
                    if not ref_val or not pred_val:
                        if pred_val != ref_val:
                            correct = False
                            break
                    else:
                        if ref_val in pred_val or pred_val in ref_val:
                            keywords = ['网元']
                            if any(word in ref_val and word not in pred_val for word in keywords):
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


# 子串判断，网优专家_网优总控agent分类
@ICL_EVALUATORS.register_module()
class UMEWangyouEvaluator5(BaseEvaluator):
    def __init__(self):
        super().__init__()

    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref in zip(predictions, references):
            pred = str_postprocessor2(pred)
            ref = str_postprocessor2(ref)
            correct = True

            if pred is None or ref is None:
                correct = False
            else:
                if pred not in ref and ref not in pred:
                    correct = False

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
