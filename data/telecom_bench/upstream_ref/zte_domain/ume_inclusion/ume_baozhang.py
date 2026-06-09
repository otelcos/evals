from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import str2json


# 保障专家-创建任务
def baozhang_domain_words(pred: dict, ref: dict, input_str: str):
    pref_guarantee_type = pred.get('guarantee_type', '')
    ref_scene = ref.get('scene', '')
    ref_operation = ref.get('operation', '')

    if pref_guarantee_type in ['容量', '感知'] and pref_guarantee_type not in input_str:
        pred['guarantee_type'] = '无法识别'
    if ref_scene == ['无法识别']:
        ref['operation'] = ['无法识别']
        pred['operation'] = '无法识别'
    if ref_operation == '删除':
        pred['scene'] = ref.get('scene', ['无'])[0]

    # pred = {k: list(v) if isinstance(v, list) else [v] for k, v in pred.items()}
    return pred, ref


# 保障专家-创建任务
@ICL_EVALUATORS.register_module()
class UMEBaozhangEvaluator1(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str], inputs: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref, input_str in zip(predictions, references, inputs):
            # 解析模型回答和参考答案
            pred = str2json(pred)
            ref = str2json(ref)

            if pred is None or ref is None:
                is_correct.append(False)
                processed_pred.append(pred)
                processed_gold.append(ref)
                continue

            # 应用领域特定的处理规则
            pred, ref = baozhang_domain_words(pred, ref, input_str)

            # 执行核心比较逻辑
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        """业务侧提供的函数"""
        CHECK_ITEMS = ["scene", "operation", "guarantee_type", "time", "location"]

        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            llm_result = llm_result[0] if isinstance(llm_result, list) else llm_result
            item_result = False

            for expected in expected_items:
                if isinstance(expected, list):
                    if expected == llm_result:
                        item_result = True
                        break
                else:
                    if expected in llm_result or llm_result in expected:
                        item_result = True
                        break

            if not item_result:
                return False

        return True


# 保障专家-任务中
@ICL_EVALUATORS.register_module()
class UMEBaozhangEvaluator2(BaseEvaluator):
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

                    if key == 'goal' and pred_val == ref_val == '无法识别':
                        break

                    if key == 'operation' and ref_val == '空':
                        continue

                    if isinstance(ref_val, list):
                        if pred_val not in ref_val:
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


# 保障专家-指标增删
@ICL_EVALUATORS.register_module()
class UMEBaozhangEvaluator3(BaseEvaluator):
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

                    if '空' in ref_val:
                        continue

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
