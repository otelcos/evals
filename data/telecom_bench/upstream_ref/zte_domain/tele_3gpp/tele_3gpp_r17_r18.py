import json
import os

from datasets import Dataset

from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class Tele3gppR17(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        data = []
        file_path = os.path.join(path, f"{name}.jsonl")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)

                question = obj["问题"] if "问题" in obj else obj.get("question")
                options = obj["选项"] if "选项" in obj else obj.get("options", [])
                answer = obj["答案"] if "答案" in obj else obj.get("answer")

                item = {
                    "question": question,
                    "A": options[0] if len(options) > 0 else None,
                    "B": options[1] if len(options) > 1 else None,
                    "C": options[2] if len(options) > 2 else None,
                    "D": options[3] if len(options) > 3 else None,
                    "answer": answer,
                }
                data.append(item)
        return Dataset.from_list(data)
