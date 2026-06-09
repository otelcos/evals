import json
import re
from typing import List

from datasets import Dataset

from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


@LOAD_DATASET.register_module()
class CpacityCompositeDataset(BaseDataset):
    """
    读取JSONL格式的组合题数据集。
    每行JSON可能包含多个sub_questions，需要分别提取为独立的数据项。
    """

    @staticmethod
    def load(path: str):
        data = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析每行的JSON对象
                item = json.loads(line)

                # 获取顶层的tags（作为info）
                tags = item.get("tags", {})

                # 遍历sub_questions列表
                sub_questions = item.get("sub_questions", [])
                for sub_q in sub_questions:
                    question_obj = sub_q.get("question", {})

                    # 提取question_type
                    question_type = sub_q.get("question_type", "")

                    # 提取question stem
                    question = question_obj.get("stem", "")

                    # 提取options (A, B, C, D, E)
                    options = question_obj.get("options", {})
                    option_a = options.get("A", "")
                    option_b = options.get("B", "")
                    option_c = options.get("C", "")
                    option_d = options.get("D", "")
                    option_e = options.get("E", "")

                    # 提取correct_answers并连接成字符串
                    correct_answers = question_obj.get("correct_answers", [])
                    answer = "".join(map(str, correct_answers))

                    data.append(
                        {
                            "question_type": question_type,
                            "question": question,
                            "A": option_a,
                            "B": option_b,
                            "C": option_c,
                            "D": option_d,
                            "E": option_e,
                            "answer": answer,
                            "info": tags,
                        }
                    )

        return Dataset.from_list(data)


@LOAD_DATASET.register_module()
class CapacityQuestionsDataset(BaseDataset):
    """
    读取JSONL格式的根因分析题目数据集。
    每行JSON是一个独立的问题，直接提取为数据项。
    """
    @staticmethod
    def load(path: str):
        data = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析每行的JSON对象
                item = json.loads(line)

                # 获取顶层的tags（作为info）
                tags = item.get('tags', {})

                # 提取question_type（在顶层）
                question_type = item.get('question_type', '')

                # 提取question对象
                question_obj = item.get('question', {})

                # 提取question stem
                question = question_obj.get('stem', '')

                # 提取options (A, B, C, D, E)
                options = question_obj.get('options', {})
                option_a = options.get('A', '')
                option_b = options.get('B', '')
                option_c = options.get('C', '')
                option_d = options.get('D', '')
                option_e = options.get('E', '')

                # 提取correct_answers并连接成字符串
                correct_answers = question_obj.get('correct_answers', [])
                answer = ''.join(map(str, correct_answers))

                data.append({
                    'question_type': question_type,
                    'question': question,
                    'A': option_a,
                    'B': option_b,
                    'C': option_c,
                    'D': option_d,
                    'E': option_e,
                    'answer': answer,
                    'info': tags,
                })

        return Dataset.from_list(data)
