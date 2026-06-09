
from datetime import datetime
from typing import Counter, List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import str2json


def format_time_to_custom(time_str: str) -> str:
    """
    将 "2025-07-28 10:51:20" 转为 "2025/7/28 10:51"
    如果输入为空或格式不正确，则返回原字符串
    """
    if not time_str or not isinstance(time_str, str):
        return time_str if time_str else ""

    try:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return f"{dt.year}/{dt.month}/{dt.day} {dt.hour:02}:{dt.minute:02}"
    except (ValueError, TypeError):
        # 如果解析失败，返回原字符串
        return time_str


@ICL_EVALUATORS.register_module()
class IDAPrameterEvaluator(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref in zip(predictions, references):
            # pred = pred.split('Output:')[-1].split('\nThought:')[0].strip()
            pred = pred.replace("\\{", "{").replace("\\}", "}").strip()
            pred = pred.replace("{{", "{").replace("}}", "}").strip().replace("\\", "")
            print(pred)
            pred = str2json(pred)
            ref = str2json(ref)

            # 先检查是否为 None，再进行后续操作
            if pred is None or ref is None:
                is_correct.append(False)
                processed_pred.append(pred)
                processed_gold.append(ref)
                continue

            # 只有在 pred 不为 None 且包含 time 字段时才进行格式化
            if "time" in pred and pred["time"]:
                pred["time"] = format_time_to_custom(pred["time"])

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
        for key in expected_answer.keys():
            if expected_answer[key] == llm_result_answer[key]:
                continue
            else:
                instance_result = False
                break
        return instance_result


@ICL_EVALUATORS.register_module()
class IDASQLPrameterEvaluator(BaseEvaluator):
    """按key_list校验KV，返回每个key的准确率"""

    def __init__(self, key_list: List[str]):
        assert isinstance(key_list, list) and all(
            isinstance(k, str) for k in key_list
        ), "key_list 必须为字符串列表"
        self.original_key_list = key_list  # 保存原始key用于返回结果
        self.key_list = [k.upper() for k in key_list]  # 转大写用于忽略大小写比较

    def _scalar_equal(self, a, b) -> bool:
        return a == b

    def _list_equal_ignore_order(self, a: list, b: list) -> bool:
        try:
            return Counter(a) == Counter(b)
        except TypeError:
            try:
                return sorted(a, key=str) == sorted(b, key=str)
            except Exception:
                return False

    def _is_empty(self, v) -> bool:
        return (isinstance(v, list) and len(v) == 0) or (isinstance(v, str) and v == "")

    def _split_by_commas(self, value):
        """如果value是字符串且包含逗号（半角或全角），则拆分成列表；否则返回原值"""
        if not isinstance(value, str):
            return value
        # 检查是否包含半角或全角逗号
        if ',' in value or '，' in value:
            # 先按全角逗号分割，再按半角逗号分割
            parts = []
            for part in value.split(','):
                parts.extend(part.split('，'))
            # 去除空白并过滤空字符串
            return [p.strip() for p in parts if p.strip()]
        return value

    def _is_substring_match(self, ref_str: str, pred_str: str) -> bool:
        """判断ref_str是否是pred_str的子串（模糊匹配）"""
        if not isinstance(ref_str, str) or not isinstance(pred_str, str):
            return False
        return ref_str in pred_str

    def _list_substring_match(self, ref_list: list, pred_list: list) -> bool:
        """判断ref_list中的每个元素是否是pred_list中某个元素的子串"""
        if not isinstance(ref_list, list) or not isinstance(pred_list, list):
            return False
        if len(ref_list) == 0:
            return len(pred_list) == 0

        # 对于ref_list中的每个元素，检查是否在pred_list的某个元素中作为子串出现
        for ref_item in ref_list:
            ref_str = str(ref_item).strip()
            if not ref_str:  # 空字符串跳过
                continue
            found = False
            for pred_item in pred_list:
                pred_str = str(pred_item).strip()
                if ref_str in pred_str:
                    found = True
                    break
            if not found:
                return False
        return True

    def _compare_by_rule(self, ref_v, pred_v) -> bool:
        """模糊匹配：ref是pred的子串即认为正确"""
        # 空值等价：[]和''视为相等
        if self._is_empty(ref_v):
            return self._is_empty(pred_v)

        # 如果pred_v是字符串且包含逗号，先转换为列表
        if isinstance(pred_v, str):
            pred_v = self._split_by_commas(pred_v)

        # 如果ref_v是列表，pred_v也必须是列表或转换为列表
        if isinstance(ref_v, list):
            if not isinstance(pred_v, list):
                # 如果pred_v是字符串，尝试转换为列表
                if isinstance(pred_v, str):
                    pred_v = [pred_v] if pred_v else []
                else:
                    pred_v = [pred_v]

            # 两个都是列表：ref中的每个元素都是pred中某个元素的子串
            return self._list_substring_match(ref_v, pred_v)

        # ref_v是字符串
        if isinstance(ref_v, str):
            # 如果pred_v是列表，遍历列表中的每个元素
            if isinstance(pred_v, list):
                if len(pred_v) == 0:
                    return False
                # 检查ref_v是否是pred_v中任意元素的子串
                ref_str = ref_v.strip()
                if not ref_str:
                    return self._is_empty(pred_v)
                for pred_item in pred_v:
                    pred_str = str(pred_item).strip()
                    if ref_str in pred_str:
                        return True
                return False
            else:
                # 两个都是字符串：ref是pred的子串
                return self._is_substring_match(ref_v, pred_v)

        # 其他类型：回退到标量比较
        return self._scalar_equal(ref_v, pred_v)

    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        kv_detail = []

        for pred, ref in zip(predictions, references):
            # 转大写解析JSON以忽略大小写
            pred_json = str2json(pred.upper() if isinstance(pred, str) else "")
            ref_json = str2json(ref.upper() if isinstance(ref, str) else "")

            processed_pred.append(pred_json)
            processed_gold.append(ref_json)

            sample_kv_result = {}

            # JSON解析失败：所有key标记为False
            if (
                pred_json is None
                or ref_json is None
                or not isinstance(ref_json, dict)
                or not isinstance(pred_json, dict)
            ):
                for original_k in self.original_key_list:
                    sample_kv_result[original_k] = False
                kv_detail.append(sample_kv_result)
                continue

            # 标准答案缺失key：所有key标记为False
            missing_in_gold = [k for k in self.key_list if k not in ref_json]
            if missing_in_gold:
                print(f"\nMissing key in gold: {missing_in_gold}\n")
                for original_k in self.original_key_list:
                    sample_kv_result[original_k] = False
                kv_detail.append(sample_kv_result)
                continue

            # 逐个key判断匹配：缺key为False，存在则按规则比较value
            for original_k, upper_k in zip(self.original_key_list, self.key_list):
                if upper_k not in pred_json:
                    sample_kv_result[original_k] = False
                else:
                    # 对pred的value进行字符串转列表处理（如果包含逗号）
                    pred_value = pred_json[upper_k]
                    if isinstance(pred_value, str):
                        pred_value = self._split_by_commas(pred_value)
                        # 如果转换成了列表，更新pred_json以便后续处理
                        if isinstance(pred_value, list):
                            pred_json[upper_k] = pred_value

                    is_match = self._compare_by_rule(ref_json[upper_k], pred_value)
                    sample_kv_result[original_k] = is_match

            kv_detail.append(sample_kv_result)

        # 计算每个key的准确率
        key_accuracies = {}
        num_samples = len(kv_detail)

        if num_samples == 0:
            avg_accuracy = 0.0
        else:
            for original_k in self.original_key_list:
                correct_count = sum(1 for sample_result in kv_detail if sample_result.get(original_k, False))
                key_acc = correct_count / num_samples
                key_accuracies[f"{original_k}"] = key_acc * 100

            # 所有key的平均准确率作为最终accuracy
            avg_accuracy = sum(key_accuracies.values()) / len(key_accuracies) if key_accuracies else 0.0

        result = {
            "accuracy": avg_accuracy,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "kv_detail": kv_detail,
            },
        }

        # 添加每个key的准确率到返回结果
        result.update(key_accuracies)

        return result
