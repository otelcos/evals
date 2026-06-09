import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from opencompass.datasets import BaseJudgeACCEvaluator
from opencompass.registry import ICL_EVALUATORS

from opencompass.datasets import BaseJudgeScoreEvaluator


@ICL_EVALUATORS.register_module()
class CCNJudgeACCEvaluator(BaseJudgeACCEvaluator):
    def __init__(self, prompt=None, judge_model=None):
        super().__init__(prompt=prompt, judge_model=judge_model)

    def _get_prompt(self) -> str:
        prompt = """# 考题与考生答案
当前有一个考试问题、该问题的标准答案和一位考生的答案：
考试问题：{question}
标准答案：{reference}
考生答案：{prediction}
# 任务
请分析考生答案与标准答案的最终结果是否一致。
# 判断依据
1、如果考生答案与标准答案中的最终结果不一致，请给0分。
2、如果考生答案与标准答案中的最终结果一致，请给1分。
3、请注意，考生答案与标准答案中最终结果的一致性是评价的唯一标准，任何考生补充的额外内容都应该被忽略。
# 输出格式要求
请你按照以下句子格式输出，除了该句子之外不得输出任何多余字符！其中分值是一个纯数字，不要输出‘x分’。
分析过程：...；分值：x。
# 开始，输出句子"""
        return prompt

    def _get_judge_model(self):
        if self._judge_model is not None:
            return self._judge_model
        from opencompass.judge_models.judge_llama import JudgeLlama

        return JudgeLlama()

    def _extract_judge(self, judge_message: str):
        try:
            match = re.search(r"(?:分值：|分值:)\s*(\d+)", judge_message.strip())
            if match:
                score = int(match.group(1))
                if score == 1:
                    return True
                elif score == 0:
                    return False
                return None
            return None
        except Exception:
            return None


@ICL_EVALUATORS.register_module()
class CCNJudgeScoreEvaluator(BaseJudgeScoreEvaluator):
    def __init__(self, prompt=None, judge_model=None):
        super().__init__(
            prompt=prompt, judge_model=judge_model, score_levels=[0, 1, 2, 3, 4, 5]
        )

    def _get_prompt(self) -> str:
        prompt = """
请根据以下步骤对<考生回答>进行评分。严格按照格式要求输出：
### 评分任务说明
- 评分范围：0-5分（整数），5分为最优
- 评分维度：准确性、完整性、逻辑性、有用性
- 输出格式：必须包含BEGIN和END标记
### 评分步骤
1. **理解回答**：分析回答是否准确完整并解决对应问题
2. **维度检查**：
   - 准确性（事实是否正确，无知识点错误无虚假信息）
   - 完整性（是否覆盖标准答案的关键要点）
   - 逻辑性（推理是否连贯，逻辑步骤顺序正确，方案排查符合实际运维规范）
   - 有用性（是否对问题解答实际有帮助）
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
            match = re.search(
                r"BEGIN\s*(\{.*?\})\s*END", judge_message.strip(), re.DOTALL
            )
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
            score_match2 = re.search(r"score\s*:\s*(\d+)", judge_message)
            if score_match2:
                score = int(score_match2.group(1))
                if score in self._score_levels:
                    return score

            return None
        except Exception:
            return None


@ICL_EVALUATORS.register_module()
class CCNJudgeScoreEvaluator2(BaseJudgeScoreEvaluator):
    """CCN 评分评估器（0-10分），支持必答要点和可选要点"""

    def __init__(
        self, prompt=None, judge_model=None, score_levels=None, score_weight_func=None
    ):
        if score_levels is None:
            score_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        super().__init__(
            score_levels=score_levels,
            prompt=prompt,
            judge_model=judge_model,
            score_weight_func=score_weight_func,
        )

    def _get_prompt(self) -> str:
        prompt = """请根据以下步骤对<考生回答>进行评分。严格按照格式要求输出：
### 评分任务说明
- 评分范围：0-10分（整数），10分为最优
- 评分维度：准确性、完整性、逻辑性、有用性
- 输出格式：必须包含BEGIN和END标记
### 评分步骤
1. **理解回答**：分析回答是否准确完整并解决对应问题
2. **维度检查**：
   - 准确性（事实是否正确，无知识点错误无虚假信息）：全部正确给10分，若有错误或虚假信息依次减分
   - 完整性（是否覆盖标准答案的关键要点）：将模型的答案和标准答案中的必答要点对比，全部答出给9分，少答出1个必答要点依次减分；可选要点全部答给1分
   - 逻辑性（推理是否连贯，逻辑步骤顺序正确，方案排查符合实际运维规范）：全部正确给10分，若有错误或虚假信息依次减分
   - 有用性（是否对问题解答实际有帮助）
3. **综合评分**：各维度加权后平均数取整
### 输出格式
BEGIN
{{
  "analysis": "逐维度分析文本...",
  "score": X  # 0-10整数
}}
END
### 题目和考生回答
问题：{question}
回答：{prediction}
必答要点：{key_point}
可选要点：{opt_point}
标准答案：{answer}
请开始评分："""
        return prompt

    def _get_judge_model(self):
        if self._judge_model is not None:
            return self._judge_model
        from opencompass.judge_models.judge_qwen235b import Qwen235B

        return Qwen235B()

    def _extract_judge(self, judge_message: str) -> Optional[int]:
        """从评判消息中提取分数（0-10）"""
        try:
            # 尝试提取 JSON 格式的分数
            # 查找 BEGIN 和 END 之间的内容
            begin_match = re.search(r"BEGIN\s*(.*?)\s*END", judge_message, re.DOTALL)
            if begin_match:
                json_str = begin_match.group(1).strip()
                try:
                    data = json.loads(json_str)
                    score = data.get("score")
                    if isinstance(score, int) and score in self._score_levels:
                        return score
                except json.JSONDecodeError:
                    pass

            # 如果 JSON 解析失败，尝试直接搜索分数
            # 查找 "score": X 格式
            score_match = re.search(r'"score"\s*:\s*(\d+)', judge_message)
            if score_match:
                score = int(score_match.group(1))
                if score in self._score_levels:
                    return score

            # 查找 "分值" 或 "分数" 关键词
            score_match = re.search(
                r"(?:分值|分数|score)[：:：]\s*(\d+)", judge_message
            )
            if score_match:
                score = int(score_match.group(1))
                if score in self._score_levels:
                    return score

            return None
        except Exception:
            return None

    def _process_sample(
        self,
        question: str,
        prediction: str,
        reference: str,
        key_point: str = "",
        opt_point: str = "",
    ):
        """处理单个样本"""
        # 如果 key_point 或 opt_point 为空，使用默认值
        key_point = key_point if key_point else "无"
        opt_point = opt_point if opt_point else "无"

        judge_prompt = self._get_prompt().format(
            question=question,
            prediction=prediction,
            answer=reference,
            key_point=key_point,
            opt_point=opt_point,
        )
        judge_message = self._get_judge_model().chat(judge_prompt)
        judge_result = self._extract_judge(judge_message)

        return {
            "judge_prompt": judge_prompt,
            "judge_message": judge_message,
            "judge_result": judge_result,
        }

    def score(
        self,
        predictions: List,
        references: List,
        questions: List,
        key_points: List = None,
        opt_points: List = None,
    ) -> dict:
        """评分方法，支持必答要点和可选要点"""
        if len(predictions) != len(references) or len(predictions) != len(questions):
            raise ValueError(
                "Predictions, references and questions have different lengths"
            )

        # 处理默认值
        if key_points is None:
            key_points = [""] * len(questions)
        if opt_points is None:
            opt_points = [""] * len(questions)

        if len(key_points) != len(questions):
            key_points = [""] * len(questions)
        if len(opt_points) != len(questions):
            opt_points = [""] * len(questions)

        valid_count = 0
        score_count = {level: 0 for level in self._score_levels}
        results_order = [None] * len(questions)
        detail_dict = {}

        with ThreadPoolExecutor(max_workers=256) as executor:
            futures = {
                executor.submit(self._process_sample, q, p, r, kp, op): idx
                for idx, (q, p, r, kp, op) in enumerate(
                    zip(questions, predictions, references, key_points, opt_points)
                )
            }

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results_order[idx] = result
                score = result["judge_result"]
                if score is not None and score in score_count:
                    score_count[score] += 1
                    valid_count += 1

        seen_keys = set()
        for r in results_order:
            if r is not None:
                seen_keys.update(r.keys())
        for k in seen_keys:
            detail_dict[k] = []
        for r in results_order:
            for k in seen_keys:
                if r is not None and k in r:
                    detail_dict[k].append(r[k])
                else:
                    detail_dict[k].append(None)

        total_weighted_score = sum(
            self._score_weight_func(score) * count
            for score, count in score_count.items()
        )
        score_rate = (
            (total_weighted_score / (valid_count * self._score_levels[-1]))
            if valid_count > 0
            else 0
        )
        result = {"score": score_rate * 100, "detail_dict": detail_dict}

        post_result = self._postprocess(result)
        return post_result
