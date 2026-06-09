from typing import Any, Dict, List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import str2json


def _json_recall_similarity(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    """
    计算基于参考答案 gold 的键值严格匹配比例（百分制）。

    - 分母为 gold 的键数；
    - 仅当 pred 中存在相同键且值严格相等时计为命中；
    - 当 gold 为空时记为 100；
    """
    if not gold:
        return 100.0

    match_count = 0
    for key, value in gold.items():
        if key in pred and str(pred[key]) == str(value):
            match_count += 1

    total = len(gold)
    return (match_count / total) * 100 if total else 100.0


@ICL_EVALUATORS.register_module()
class WiredParasEvaluator(BaseEvaluator):
    """参数解析评估器
    将模型输出与参考答案进行 JSON 级别的键值严格匹配评估，返回匹配比例。
    """

    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred: List[Dict[str, Any]] = []
        processed_gold: List[Dict[str, Any]] = []
        similarities: List[float] = []

        for pred, ref in zip(predictions, references):
            pred_json = str2json(pred) or {}
            ref_json = str2json(ref) or {}

            sim = _json_recall_similarity(ref_json, pred_json)

            processed_pred.append(pred_json)
            processed_gold.append(ref_json)
            similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        return {
            "accuracy": avg_similarity,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "similarities": similarities,
            },
        }
