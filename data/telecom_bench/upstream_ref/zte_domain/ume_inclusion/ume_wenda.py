import json
import re

from opencompass.datasets import BaseJudgeScoreEvaluator
from opencompass.registry import ICL_EVALUATORS


@ICL_EVALUATORS.register_module()
class UMEWendaScoreEvaluator(BaseJudgeScoreEvaluator):
    def __init__(self, prompt=None, judge_model=None):
        super().__init__(
            prompt=prompt,
            judge_model=judge_model,
            score_levels=[1, 2, 3, 4, 5])

    def _get_prompt(self) -> str:
        prompt = """
请根据以下步骤对<考生回答>进行评分。严格按照格式要求输出：

### 评分任务说明
- 评分范围：0-5分（整数），5分为最优
- 评分维度：准确性、完整性、逻辑性、有用性
- 输出格式：必须包含BEGIN和END标记

### 评分步骤
1. **理解回答**：分析回答是否准确解决用户问题
2. **维度检查**：
   - 准确性（事实是否正确）
   - 完整性（是否覆盖关键点）
   - 逻辑性（推理是否连贯）
   - 有用性（是否实际有帮助）
3. **综合评分**：加权各维度后取整

### 输出格式
BEGIN
{{
  "analysis": "逐维度分析文本...",
  "score": X  # 0-5整数
}}
END

### 题目和考生回答
问题：{question}
回答：{prediction}
标准答案：{reference}

请开始评分：
"""
        return prompt

    def _get_judge_model(self):
        if self._judge_model is not None:
            return self._judge_model
        from opencompass.judge_models.judge_qwen3 import Qwen3
        return Qwen3()

    def _extract_judge(self, judge_message: str):
        try:
            # 首先尝试匹配BEGIN/END格式的JSON
            match = re.search(r"BEGIN\s*(\{.*?\})\s*END", judge_message.strip(), re.DOTALL)
            if match:
                try:
                    json_data = json.loads(match.group(1))
                    if isinstance(json_data, dict) and "score" in json_data:
                        score = int(json_data["score"])
                        if score in self._score_levels:
                            return score
                except:
                    pass

            # 如果没有找到BEGIN/END格式，直接搜索"score": X格式
            score_match = re.search(r'"score"\s*:\s*(\d+)', judge_message)
            if score_match:
                score = int(score_match.group(1))
                if score in self._score_levels:
                    return score

            # 最后尝试匹配没有引号的score格式
            score_match2 = re.search(r'score\s*:\s*(\d+)', judge_message)
            if score_match2:
                score = int(score_match2.group(1))
                if score in self._score_levels:
                    return score

            return None
        except Exception:
            return None
