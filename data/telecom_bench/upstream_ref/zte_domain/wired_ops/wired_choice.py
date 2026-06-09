import json
import os
from typing import Any, Dict, List

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class WiredChoiceDataset(BaseDataset):
    """Wired 运维选择题数据集（单选/多选）。

    读取 `path/{name}.jsonl`，并将每行的嵌套结构展开为标准字段：
    - question: str
    - options: list（保留原始 list，与原脚本一致）
    - type: str
    - answer: str
    """

    @staticmethod
    def load(path: str, name: str):
        data: List[Dict[str, Any]] = []
        file_path = os.path.join(path, f"{name}.jsonl")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                # 格式：{"多项选择题": {"问题": "...", "选项": [...], "答案": "A, C"}}
                qtype, content = next(iter(obj.items()))
                data.append(
                    dict(
                        question=content.get("问题", ""),
                        options=content.get("选项", []),
                        type=qtype,
                        answer=content.get("答案", ""),
                    )
                )

        return Dataset.from_list(data)
