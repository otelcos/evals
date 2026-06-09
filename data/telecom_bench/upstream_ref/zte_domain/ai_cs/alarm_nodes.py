from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import str2json, extract_non_reasoning_content, json_str


def alarm_nodes_processor(text):
    text = extract_non_reasoning_content(text)
    text = json_str(text)
    return text


def are_json_equal(json1, json2):
    """
    比较两个 JSON 对象是否内容完全相同，忽略键的顺序。

    参数:
        json1: 第一个 JSON 对象（字典、列表等）
        json2: 第二个 JSON 对象

    返回:
        bool: 如果内容完全相同返回 True，否则返回 False
    """
    # 如果是字典，递归比较每个键值对
    if isinstance(json1, dict) and isinstance(json2, dict):
        if set(json1.keys()) != set(json2.keys()):
            return False
        for key in json1:
            if not are_json_equal(json1[key], json2[key]):
                return False
        return True

    # 如果是列表，递归比较每个元素
    elif isinstance(json1, list) and isinstance(json2, list):
        if len(json1) != len(json2):
            return False
        # 如果列表中的元素是字典，并且顺序可能不同，可以先排序再比较
        # 这里我们假设列表中是字典类型，并以所有键的排序元组作为排序依据
        try:
            if all(isinstance(item, dict) for item in json1) and all(isinstance(item, dict) for item in json2):
                # 将每个字典转换为排序后的元组进行比较
                sorted_json1 = sorted(json1, key=lambda d: tuple(sorted(d.items())))
                sorted_json2 = sorted(json2, key=lambda d: tuple(sorted(d.items())))
                return sorted_json1 == sorted_json2
            else:
                # 否则直接逐个比较
                for item1, item2 in zip(json1, json2):
                    if not are_json_equal(item1, item2):
                        return False
                return True
        except TypeError:
            # 如果无法排序（比如包含不可哈希类型），则直接比较
            for item1, item2 in zip(json1, json2):
                if not are_json_equal(item1, item2):
                    return False
            return True

    # 基本类型直接比较
    else:
        return json1 == json2


@ICL_EVALUATORS.register_module()
class AlarmNodesEvaluator(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref in zip(predictions, references):
            pred = str2json(pred)
            ref = str2json(ref)

            if pred is None or ref is None:
                correct = False
            else:
                correct = are_json_equal(pred, ref)

            is_correct.append(correct)
            processed_pred.append(pred)
            processed_gold.append(ref)

        accuracy = sum(is_correct) / len(is_correct) if is_correct else 0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                # "processed_pred": processed_pred,
                # "processed_gold": processed_gold,
                "is_correct": is_correct
            }
        }
