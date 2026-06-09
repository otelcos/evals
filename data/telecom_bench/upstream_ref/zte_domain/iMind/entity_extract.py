from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import str2json
import json
import re


@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator1(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['rootCauseType', 'rootNetwork', 'rootNe']
        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result
        # if not item_result:
        #     return False
        # return instance_result


@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator2(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref in zip(predictions, references):
            # print(f"pred:{type(pred)}, ref:{type(ref)}")
            # print(pred)
            # print(ref)
            # Started by AICoder, pid:v8f5ape714f7afe14a3b0be1e0c1720af5748e5d
            # 提取 JSON 部分
            json_str = re.search(r'\{.*\}', pred).group()
            # 转换为字典
            pred = json.loads(json_str)
            # Ended by AICoder, pid:v8f5ape714f7afe14a3b0be1e0c1720af5748e5d
            pred = str2json(pred)
            ref = str2json(ref)
            # print(f"pred:{type(pred)}, ref:{type(ref)}")
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['device', 'city', 'network']
        # print(f"llm_result_dict:{llm_result_dict}, type:{type(llm_result_dict)}")
        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if item=='network':
                llm_result=llm_result.lower()
                print(llm_result)
                # print(expected_items)
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result


@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator3(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['专业', '告警级别']

        # for item in CHECK_ITEMS:
        #     expected_items = expected_dict.get(item, [])
        #     llm_result = llm_result_dict.get(item, item)
        #     llm_result = llm_result[0] if isinstance(llm_result, list) and llm_result!=[] else llm_result
        #     item_result = False
        #     for expected in expected_items:
        #         if isinstance(expected, list):
        #             if expected == llm_result:
        #                 item_result = True
        #                 break
        #         else:
        #             if expected in llm_result or llm_result in expected:
        #                 item_result = True
        #                 break

        # if not item_result:
        #     return False
        # return True
        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if isinstance(llm_result, list):
                llm_result.sort()
                expected_items.sort()
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result


@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator4(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['时间范围', '告警级别', '专业', '地市', '网元', '告警标题']

        # for item in CHECK_ITEMS:
        #     expected_items = expected_dict.get(item, [])
        #     llm_result = llm_result_dict.get(item, item)
        #     llm_result = llm_result[0] if isinstance(llm_result, list) and llm_result!=[] else llm_result
        #     item_result = False
        #     for expected in expected_items:
        #         if isinstance(expected, list):
        #             if expected == llm_result:
        #                 item_result = True
        #                 break
        #         else:
        #             if expected in llm_result or llm_result in expected:
        #                 item_result = True
        #                 break

        # if not item_result:
        #     return False
        # return True
        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if isinstance(llm_result, list) and llm_result!=[] and len(llm_result)==1:
                llm_result = llm_result[0]
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result


@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator5(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['title', 'device']

        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result


@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator6(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str] ) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref  in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['时间范围', '事件级别', '地市', '网元']
        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if isinstance(llm_result, list) and llm_result!=[] and len(llm_result)==1:
                llm_result = llm_result[0]
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result


@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator7(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str] ) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref  in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['时间范围' , '专业', '网元名称']

        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result


@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator8(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str] ) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref  in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['本端设备', '对端设备', '地市']

        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if isinstance(llm_result, list) and llm_result!=[] and len(llm_result)==1:
                llm_result = llm_result[0]
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result


@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator9(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str] ) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref  in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['机房名称', '专业']

        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result


@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator10(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str] ) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref  in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['网元']

        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result


@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator11(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str] ) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref  in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['网元']

        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result

@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator12(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str] ) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref  in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['room', 'resource_type']

        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result

@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator13(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str] ) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref  in zip(predictions, references):
            pred = str2json(pred)
            ref = str2json(ref)
            if pred is None or ref is None:
                is_correct.append(False)
                processed_pred.append(pred)
                processed_gold.append(ref)
                continue
            # print(f"processed_pred:{processed_pred[-1]}\nprocessed_gold:{processed_gold[-1]}")
            correct = self._check_dict(ref, pred)
            is_correct.append(correct)
            processed_pred.append(pred)
            processed_gold.append(ref)
            # print(f"processed_pred:{processed_pred[-1]}\nprocessed_gold:{processed_gold[-1]}")
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
        CHECK_ITEMS = ['时间范围', '网元', '端口', '方向', '指标']
        # print(llm_result_dict)
        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if isinstance(llm_result, list) and llm_result!=[] and len(llm_result)==1:
                llm_result = llm_result[0]
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result

@ICL_EVALUATORS.register_module()
class instance_extract_Evaluator14(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str] ) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref  in zip(predictions, references):
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
                "is_correct": is_correct
            }
        }

    def _check_dict(self, expected_dict, llm_result_dict):
        CHECK_ITEMS = ['时间范围', '网元']

        instance_result = True
        for item in CHECK_ITEMS:
            expected_items = expected_dict.get(item, [])
            llm_result = llm_result_dict.get(item, item)
            if expected_items!=llm_result:
                instance_result = False
                break
        return instance_result
