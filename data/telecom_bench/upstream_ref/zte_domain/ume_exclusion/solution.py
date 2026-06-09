import difflib
import re
from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS

from rouge_chinese import Rouge


@ICL_EVALUATORS.register_module()
class UMESolutionEvaluator(BaseEvaluator):
    """故障解决方案评估器

    评估指标包括：
    1. Rouge指标（rouge1, rouge2, rougeL）
    2. 工具使用准确率（工具步骤完全一致）
    3. 工具使用相似度（基于序列匹配）
    """

    def __init__(self):
        super().__init__()
        self.rouge = Rouge()

    def _extract_tool_steps(self, text: str) -> List[str]:
        """提取文本中方括号内的工具步骤"""
        return re.findall(r"\[(.*?)\]", text)

    def _calculate_rouge(self, pred: str, gold: str) -> tuple:
        """计算Rouge指标，返回 (rouge1, rouge2, rougeL)"""
        if not pred or not gold:
            return (0.0, 0.0, 0.0)

        try:
            rouge_scores = self.rouge.get_scores(pred, gold)
            rouge1 = rouge_scores[0]["rouge-1"]["f"]
            rouge2 = rouge_scores[0]["rouge-2"]["f"]
            rougeL = rouge_scores[0]["rouge-l"]["f"]
            return (rouge1, rouge2, rougeL)
        except Exception as e:
            print(f"计算Rouge指标时发生错误: {e}")
            return (0.0, 0.0, 0.0)

    def _calculate_tool_similarity(
        self, pred_steps: List[str], gold_steps: List[str]
    ) -> float:
        """计算工具步骤序列相似度"""
        return difflib.SequenceMatcher(None, str(pred_steps), str(gold_steps)).ratio()

    def score(
        self, predictions: List[str], references: List[str]
    ) -> dict:
        if len(predictions) != len(references):
            raise ValueError("Predictions and references have different lengths")

        # 初始化detail_dict
        detail_dict = {
            "processed_pred": [],
            "processed_gold": [],
            "pred_tool_steps": [],
            "gold_tool_steps": [],
            "tool_step_accuracy": [],
            "tool_step_similarity": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
        }

        for pred, ref in zip(predictions, references):
            detail_dict["processed_pred"].append(pred)
            detail_dict["processed_gold"].append(ref)

            # 提取工具步骤
            pred_tool_steps = self._extract_tool_steps(pred)
            ref_tool_steps = self._extract_tool_steps(ref)
            detail_dict["pred_tool_steps"].append(pred_tool_steps)
            detail_dict["gold_tool_steps"].append(ref_tool_steps)

            # 计算工具步骤准确率（必须完全一致）
            tool_step_accuracy = 1 if pred_tool_steps == ref_tool_steps else 0
            detail_dict["tool_step_accuracy"].append(tool_step_accuracy)

            # 计算工具使用相似度
            tool_step_similarity = self._calculate_tool_similarity(
                pred_tool_steps, ref_tool_steps
            )
            detail_dict["tool_step_similarity"].append(tool_step_similarity)

            # 计算Rouge指标
            rouge1, rouge2, rougeL = self._calculate_rouge(pred, ref)
            detail_dict["rouge1"].append(rouge1)
            detail_dict["rouge2"].append(rouge2)
            detail_dict["rougeL"].append(rougeL)

        # 计算平均指标
        total = len(predictions)
        avg_tool_accuracy = sum(detail_dict["tool_step_accuracy"]) / total if total > 0 else 0.0
        avg_tool_similarity = sum(detail_dict["tool_step_similarity"]) / total if total > 0 else 0.0
        avg_rouge1 = sum(detail_dict["rouge1"]) / total if total > 0 else 0.0
        avg_rouge2 = sum(detail_dict["rouge2"]) / total if total > 0 else 0.0
        avg_rougeL = sum(detail_dict["rougeL"]) / total if total > 0 else 0.0

        return {
            "tool_step_accuracy": avg_tool_accuracy * 100,
            "tool_step_similarity": avg_tool_similarity * 100,
            "rouge1": avg_rouge1 * 100,
            "rouge2": avg_rouge2 * 100,
            "rougeL": avg_rougeL * 100,
            "detail_dict": detail_dict,
        }
