import ast
import json
import re
from typing import List

from sympy import true

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import json_str


def kanwang_standardize_str(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    # 使用字典映射进行替换，避免多次字符串操作
    replacements = {
        '\\n': '\n',
        '\\': '\\\\',
        "'": '"',
        'NR': '5G',
        'LTE': '4G'
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # 使用单个正则表达式处理所有空白字符
    text = re.sub(r'\s+', ' ', text)

    # 处理 None 值和花括号
    text = re.sub(r'["]?None["]?', 'null', text)
    text = re.sub(r'{\s+', '{', text)
    text = re.sub(r'\s+}', '}', text)

    return text


def kanwang_domain_words(json_obj: dict) -> dict:
    if not isinstance(json_obj, dict):
        return json_obj

    # 预定义网络配置映射
    NETWORK_MAPPINGS = {
        'have_4_have_5': {'有5有4', '有4有5', 'have 4 and 5', 'have 5 and 4'},
        'have_4_no_5': {'4g only', 'only 4', '4gonly', 'only4', '无5有4', '有4无5'},
        'no_4_have_5': {'5g only', 'only 5', '5gonly', 'only5', '有5无4', '无4有5'},
        'all_networks': {'全网', 'ALL NETWORK', '4G，5G', '4G、5G', '4G,5G', 'All', '4G_&_5G',
                         '5G,4G', '5G，4G', "['4G','5G']", "['5G','4G']", "['NR','LTE']", "['LTE','NR']"}
    }

    # 预定义结果映射
    NETWORK_RESULTS = {
        'have_4_have_5': '有4有5',
        'have_4_no_5': '4G only',
        'no_4_have_5': '5G only',
        'all_networks': 'ALL NETWORK'
    }

    try:
        for key, value in json_obj.items():
            if key == "siteconfig":
                value_lower = str(value).lower()
                for category, values in NETWORK_MAPPINGS.items():
                    if value_lower in values:
                        json_obj[key] = NETWORK_RESULTS[category]
                        break

            elif key == 'coverage_scenario':
                json_obj[key] = re.sub(r'\s*area\s*', '', str(value), flags=re.IGNORECASE)

            elif key == 'city':
                json_obj[key] = str(value).replace("市", "")

            elif key == 'product':
                value_str = str(value).replace("NR", "5G").replace("LTE", "4G")
                if value_str.upper() in {x.upper() for x in NETWORK_MAPPINGS['all_networks']}:
                    json_obj[key] = None
                else:
                    json_obj[key] = value_str

        return json_obj

    except Exception:
        return json_obj


def extract_api_kv(text) -> dict:
    if not isinstance(text, str):
        return text

    text = kanwang_standardize_str(text)

    # 首先尝试从 Action Input 中提取
    action_input_match = re.search(r'Action Input:\s*(\{.*?\})', text, re.DOTALL | re.IGNORECASE)
    if action_input_match:
        try:
            json_str = action_input_match.group(1)
            parsed_dict = json.loads(json_str)
            standardized_dict = kanwang_domain_words(parsed_dict)
            return json.loads(json.dumps(standardized_dict))
        except json.JSONDecodeError:
            pass

    # 如果 Action Input 提取失败，尝试提取最内层 JSON
    try:
        json_objects = re.finditer(r'\{[^{}]*\}', text)
        for match in json_objects:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    # 检查是否是最内层字典
                    if not any(isinstance(v, dict) for v in parsed.values()):
                        standardized_dict = kanwang_domain_words(parsed)
                        return json.loads(json.dumps(standardized_dict))
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return None


def extract_gold_kv(text) -> dict:
    """提取标准答案的键值对，并返回标准化的字典

    Args:
        text (str or dict): 标准答案文本或字典

    Returns:
        dict: 标准化后的字典
    """
    if isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False)

    text = kanwang_standardize_str(text)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return kanwang_domain_words(result)
    except json.JSONDecodeError:
        pass
    return {}


@ICL_EVALUATORS.register_module()
class UMEApiAbstractEvaluator(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:

        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref_kv in zip(predictions, references):
            ref = extract_gold_kv(ref_kv)

            if pred is None or ref is None:
                correct = False
            else:
                if len(pred) > len(ref):
                    correct = False
                else:
                    correct = True
                    for key in ref:
                        # 如果ref中的键值为空，则pred中可以没有这个键
                        if ref[key] is None or ref[key] == "None":
                            if key in pred and pred[key] is not None and pred[key] != "None":
                                correct = False
                                break
                            continue

                        # 如果ref中的键值不为空，则pred中必须有这个键且值相等
                        if key not in pred or pred[key] != ref[key]:
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


def get_inner_json(text):
    text = json_str(text)
    try:
        json_objects = re.finditer(r'\{[^{}]*\}', text)
        for match in json_objects:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    # 检查是否是最内层字典
                    if not any(isinstance(v, dict) for v in parsed.values()):
                        # standardized_dict = ran_domain_standardize(parsed)
                        return parsed
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return {"text": text}


@ICL_EVALUATORS.register_module()
class UMETableSelectEvaluator(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:

        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref_kv in zip(predictions, references):
            ref = ast.literal_eval(ref_kv)
            if pred:
                pred = json.loads(pred)
                lower_ref = [item.lower() for item in ref]
                print(lower_ref, pred)
                for key in pred:
                    correct = False
                    if isinstance(pred[key], str) and pred[key].lower() in lower_ref:
                        correct = True
            else:
                correct = False

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
