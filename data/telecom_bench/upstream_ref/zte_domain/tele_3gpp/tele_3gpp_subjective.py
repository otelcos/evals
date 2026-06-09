import json
import os.path as osp

from datasets import Dataset

from opencompass.judge_models import JudgeLlama
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS
from opencompass.datasets.base import BaseDataset
from opencompass.datasets.subjective.judge_eval import BaseJudgeACCEvaluator


def strip_values(item):
    return {k: v.strip() if isinstance(v, str) else v for k, v in item.items()}


@LOAD_DATASET.register_module()
class Tele3gppSubjectiveDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        filename = osp.join(path)
        with open(filename, 'r', encoding='utf-8') as file:
            raw_data = [strip_values(row) for row in json.load(file)]
        return Dataset.from_list(raw_data)


tele_3gpp_judge_prompt = """
Please evaluate the provided response based on the user's question and the standard answer. Return 1 if the provided response is correct; otherwise, return 0. Do not include any additional explanations.

User Question: {question}
Standard Answer: {reference}
Provided Response: {prediction}
"""

tele_3gpp_score_prompt = """"
# 考题与考生答案
当前有一个考试问题、该问题的标准答案和一位考生的答案：
考试问题：{reference}
标准答案：{prediction}
考生答案：{question}
# 任务
请分析考生答案包含了多少标准答案的内容，给出包含程度评分。
# 判断依据
1、如果考生答案与标准答案中的关键内容完全不相关，存在事实性或逻辑性错误，与标准答案没有任何重合的内容，请给0分。
2、如果考生答案与标准答案相关但不完全一样，与标准答案只有一部分重合的内容，没有涵盖标准答案的所有内容，此时无论考生补充了多少额外内容，都请只给1分。
3、如果考生答案与标准答案完全一致，或考生答案中的内容完全包含了标准答案中的内容，并进行了更多补充，使答案更完整，请给2分。
4、请注意，考生答案包含了多少标准答案的内容是评价的唯一标准，任何考生补充的额外内容都应该被忽略。
# 输出格式要求
请你按照以下句子格式输出，除了该句子之外不得输出任何多余字符！其中分值是一个纯数字，不要输出'x分'。
分析过程：...；分值：x。
# 开始，输出句子
"""


@ICL_EVALUATORS.register_module()
class Tele3gppSubjectiveEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.result = {}

    def score(self, predictions, references, questions):
        evaluator = Tele3gppJudge()
        self.result = evaluator.score(predictions, references, questions)
        return self.result


class Tele3gppJudge(BaseJudgeACCEvaluator):
    def score(self, predictions, references, questions):
        return self._evaluate(predictions, references, questions, tele_3gpp_judge_prompt, JudgeLlama())
