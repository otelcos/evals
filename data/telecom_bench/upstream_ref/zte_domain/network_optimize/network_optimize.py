import json
import os

from datasets import Dataset

from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


@LOAD_DATASET.register_module()
class NetOptmDataset(BaseDataset):
    """
    读取JSON格式的无线参数自优化评测集。
    数据格式为JSON数组，每个元素是一个问题对象。
    """

    @staticmethod
    def load(path: str, name: str):
        data = []

        # 构建文件路径
        file_path = os.path.join(path, f"{name}.json")

        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            # 读取每个JSON对象
            for item in items:
                # 提取question_type并做映射转换
                question_type_raw = item.get("question_type", "")
                question_type_map = {
                    "single_choice": "单项选择题",
                    "multi_choice": "多项选择题",
                }
                question_type = question_type_map.get(
                    question_type_raw, question_type_raw
                )

                # 提取type
                type_value = item.get("type", "")

                # 提取stem
                stem = item.get("stem", "")

                # 提取options (A, B, C, D, E)
                options = item.get("options", {})
                option_a = options.get("A", "")
                option_b = options.get("B", "")
                option_c = options.get("C", "")
                option_d = options.get("D", "")
                option_e = options.get("E", "")

                # 提取correct_answers并拼接成字符串
                correct_answers = item.get("correct_answers", [])
                answer = "".join(map(str, correct_answers))

                data.append(
                    {
                        "question_type": question_type,
                        "type": type_value,
                        "stem": stem,
                        "A": option_a,
                        "B": option_b,
                        "C": option_c,
                        "D": option_d,
                        "E": option_e,
                        "answer": answer,
                    }
                )

        return Dataset.from_list(data)
