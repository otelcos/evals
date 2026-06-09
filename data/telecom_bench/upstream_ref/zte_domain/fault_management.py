import json
import os

from datasets import Dataset

from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class FaultMgmtDataset(BaseDataset):
    """
    读取JSONL格式的组合题数据集。
    每行JSON可能包含多个sub_questions，需要分别提取为独立的数据项。
    """

    @staticmethod
    def load(path: str, name: str):
        data = []
        file_path = os.path.join(path, f"{name}.jsonl")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析每行的JSON对象
                item = json.loads(line)

                question = item.get("question", "")
                options = item.get("options", [])

                option_a = options[0] if len(options) > 0 else ""
                option_b = options[1] if len(options) > 1 else ""
                option_c = options[2] if len(options) > 2 else ""
                option_d = options[3] if len(options) > 3 else ""
                option_e = options[4] if len(options) > 4 else ""

                answer = item.get("answer", "")

                data.append(
                    {
                        "question": question,
                        "A": option_a,
                        "B": option_b,
                        "C": option_c,
                        "D": option_d,
                        "E": option_e,
                        "answer": answer,
                    }
                )

        return Dataset.from_list(data)
