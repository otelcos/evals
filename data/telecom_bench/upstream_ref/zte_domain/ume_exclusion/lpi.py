import json
from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import str2json


@ICL_EVALUATORS.register_module()
class UMELPIEvaluator(BaseEvaluator):
    """LPI (Logical Processing Identifier) 评估器

    评估指标包括：
    1. LPI 准确率（LPI 名称是否匹配）
    2. Extract 准确率（实体提取结果是否匹配）
    3. 整体准确率（LPI 和 Extract 都匹配）
    """

    def __init__(self):
        super().__init__()

    def _format_extract_result(self, extract_raw):
        """格式化 extract_result 为字符串"""
        if isinstance(extract_raw, dict):
            return json.dumps(extract_raw, ensure_ascii=False, sort_keys=True)
        return str(extract_raw).strip()

    def _normalize_for_match(self, text):
        """标准化文本用于匹配（去除空格）"""
        return text.replace(' ', '')

    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred = []
        processed_gold = []
        lpi_matches = []
        extract_matches = []
        full_matches = []

        for pred, ref in zip(predictions, references):
            pred_json = str2json(pred) or {}
            ref_json = str2json(ref) or {}

            pred_lpi = pred_json.get('lpi', '')
            pred_extract = self._format_extract_result(pred_json.get('extract_result', {}))
            gold_lpi = ref_json.get('lpi', '')
            gold_extract = self._format_extract_result(ref_json.get('extract_result', {}))

            lpi_match = self._normalize_for_match(pred_lpi) == self._normalize_for_match(gold_lpi)
            extract_match = self._normalize_for_match(pred_extract) == self._normalize_for_match(gold_extract)
            full_match = lpi_match and extract_match

            lpi_matches.append(lpi_match)
            extract_matches.append(extract_match)
            full_matches.append(full_match)

            processed_pred.append({
                'lpi': pred_lpi,
                'extract_result': pred_extract
            })
            processed_gold.append({
                'lpi': gold_lpi,
                'extract_result': gold_extract
            })

        total = len(predictions)
        lpi_accuracy = sum(lpi_matches) / total if total > 0 else 0.0
        extract_accuracy = sum(extract_matches) / total if total > 0 else 0.0
        full_accuracy = sum(full_matches) / total if total > 0 else 0.0

        return {
            'lpi_accuracy': lpi_accuracy * 100,
            'extract_accuracy': extract_accuracy * 100,
            'full_accuracy': full_accuracy * 100,
            'detail_dict': {
                'processed_pred': processed_pred,
                'processed_gold': processed_gold,
                'lpi_matches': lpi_matches,
                'extract_matches': extract_matches,
                'full_matches': full_matches,
            }
        }
