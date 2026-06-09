# -*- coding: utf-8 -*-
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


JsonDict = Dict[str, Any]
FieldSpec = Union[str, Tuple[str, str], Tuple[Any, ...]]


def _extract_json_4g(text: Any) -> Optional[JsonDict]:
    """4G评估器JSON提取：支持对象和数组（取首元素）"""
    if not isinstance(text, str):
        return None

    def try_parse(s):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
        return None

    # 直接解析
    result = try_parse(text)
    if result:
        return result

    # 提取数组
    start_array = text.find('[')
    end_array = text.rfind(']')
    if start_array != -1 and end_array > start_array:
        result = try_parse(text[start_array:end_array + 1])
        if result:
            return result

    # 提取对象
    start_obj = text.find('{')
    end_obj = text.rfind('}')
    if start_obj != -1 and end_obj > start_obj:
        result = try_parse(text[start_obj:end_obj + 1])
        if result:
            return result

    # 移除前缀重试
    if "best answer:" in text.lower():
        prefix_pos = text.lower().find("best answer:")
        return try_parse(text[prefix_pos + 12:].strip())

    return None


def _extract_json_5g(text: Any) -> Optional[JsonDict]:
    """5G评估器JSON提取：仅支持对象"""
    if not isinstance(text, str):
        return None

    def try_parse(s):
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    # 直接解析
    result = try_parse(text)
    if result:
        return result

    # 提取对象
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        result = try_parse(text[start:end + 1])
        if result:
            return result

    # 移除前缀重试
    if text.startswith("best answer:"):
        return try_parse(text[12:].strip())

    return None


def _compare_list_values(list1: Any, list2: Any) -> bool:
    """比较列表（忽略顺序）"""
    if list1 is None and list2 is None:
        return True
    if not isinstance(list1, list) or not isinstance(list2, list):
        return list1 == list2
    try:
        return set(list1) == set(list2)
    except TypeError:
        return sorted(list1) == sorted(list2) if len(list1) == len(list2) else False


def _field_name(field: FieldSpec) -> str:
    if isinstance(field, tuple):
        if len(field) == 2:
            return f"{field[0]}.{field[1]}"
        return ".".join(str(f) for f in field)
    return str(field)


def _get_field_value(json_obj: Optional[JsonDict], field: FieldSpec) -> Any:
    """取字段值，支持嵌套"""
    if not isinstance(json_obj, dict):
        return None
    if not isinstance(field, tuple):
        return json_obj.get(field)
    current = json_obj
    for key in field:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


class _WangyouBaseEvaluator(BaseEvaluator):
    """网优评估器基类"""
    key_fields: Sequence[FieldSpec] = ()
    list_compare_fields: Iterable[str] = ()
    json_extractor: Callable[[Any], Optional[JsonDict]] = _extract_json_4g

    def __init__(self):
        super().__init__()
        self._field_names = [_field_name(f) for f in self.key_fields]
        self._list_compare_fields = set(self.list_compare_fields)

    def score(self, predictions: List[str], references: List[str]) -> dict:
        field_matches: Dict[str, List[bool]] = {name: [] for name in self._field_names}
        full_matches: List[bool] = []
        accuracy_scores: List[float] = []
        processed_pred: List[Optional[JsonDict]] = []
        processed_gold: List[Optional[JsonDict]] = []
        details: List[dict] = []
        total_fields = len(self.key_fields)

        for pred, ref in zip(predictions, references):
            pred_json = self.json_extractor(pred)
            ref_json = self.json_extractor(ref)
            processed_pred.append(pred_json)
            processed_gold.append(ref_json)

            field_scores: Dict[str, dict] = {}
            correct_fields = 0

            # 处理每个字段
            for field in self.key_fields:
                name = _field_name(field)
                pred_value = _get_field_value(pred_json, field) if pred_json else None
                ref_value = _get_field_value(ref_json, field) if ref_json else None

                field_match = (_compare_list_values(pred_value, ref_value)
                              if name in self._list_compare_fields
                              else pred_value == ref_value)

                field_scores[name] = {"predicted": pred_value, "reference": ref_value, "match": field_match}
                field_matches[name].append(field_match)
                correct_fields += field_match

            accuracy_score = correct_fields / total_fields if total_fields else 0
            full_matches.append(accuracy_score == 1.0)
            accuracy_scores.append(accuracy_score)
            details.append({"field_scores": field_scores, "accuracy_score": accuracy_score, "full_match": accuracy_score == 1.0})

        # 汇总结果
        total_cases = len(predictions)
        overall_accuracy = sum(accuracy_scores) / total_cases if total_cases else 0.0
        field_accuracy_breakdown = {_field_name(f): sum(field_matches[_field_name(f)]) / total_cases
                                    for f in self.key_fields} if total_cases else {}

        result = {
            "total_cases": total_cases,
            "correct_cases": sum(full_matches),
            "bad_cases_count": total_cases - sum(full_matches),
            "overall_accuracy": overall_accuracy * 100,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": full_matches,
                "details": details,
            },
        }
        result.update(field_accuracy_breakdown)
        return result


@ICL_EVALUATORS.register_module()
class Wangyou4GEvaluator(_WangyouBaseEvaluator):
    """4G网优评估器：6个字段，highload_time为列表（忽略顺序）"""
    def __init__(self):
        self.key_fields = [
            'source_ishighloadcell',
            'highload_time',
            ('target', 'subnet_id'),
            ('target', 'me_id'),
            ('target', 'ldn'),
            ('load_unbalance_result', 'result'),
        ]
        self.list_compare_fields = {"highload_time"}
        self.json_extractor = _extract_json_4g
        super().__init__()


@ICL_EVALUATORS.register_module()
class Wangyou5GEvaluator(_WangyouBaseEvaluator):
    """5G网优评估器：4个字段"""
    def __init__(self):
        self.key_fields = [
            ('target', 'subnet_id'),
            ('target', 'me_id'),
            ('target', 'ldn'),
            ('load_unbalance_result', 'result'),
        ]
        self.list_compare_fields = set()
        self.json_extractor = _extract_json_5g
        super().__init__()
