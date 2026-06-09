import itertools
import json
import os
from collections import Counter, defaultdict
from typing import List, Union, Sequence

import numpy as np
from datasets import Dataset

from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


def remove_space(text: str) -> str:
    return ' '.join(text.replace('\n', ' ').split())


def ran_ap_keyword_postprocess(text: str) -> str:
    prefixes = ["关键词：", "。"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    return remove_space(text)


@LOAD_DATASET.register_module()
class RANAPQASDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        with open(os.path.join(path, f"{name}.json"), 'r', encoding='utf-8') as f:
            data = json.load(f)
            qas = data['qas']
            # 将字典转换为列表格式
            data_list = []
            for qa_id, qa in qas.items():
                qa['id'] = qa_id
                data_list.append(qa)
        return Dataset.from_list(data_list)


@LOAD_DATASET.register_module()
class RANAPQASDatasetV2(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        with open(os.path.join(path, f"{name}.json"), 'r', encoding='utf-8') as f:
            data_list = []
            # 如果文件包含多个JSON对象
            for item in json.load(f):
                data_list.append({
                    'id': item['doc_id'],
                    'doc': item['doc'],
                    'question': item['qas']['question'],
                    'A': item['qas']['A'],
                    'B': item['qas']['B'],
                    'C': item['qas']['C'],
                    'D': item['qas']['D'],
                    'answer': item['qas']['answer']
                })
        return Dataset.from_list(data_list)



@LOAD_DATASET.register_module()
class RANAPQASGPassKDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str, num_repeats: int = 1):
        """加载数据集，并支持重复n次以进行pass@k评估

        Args:
            path: 数据路径
            name: 数据集名称
            num_repeats: 每个问题重复的次数，用于pass@k评估
        """
        data_list = []
        file_path = os.path.join(path, f"{name}.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'File {file_path} does not exist, please check the '
                f'path and try again.')

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            qas = data['qas']

            for qa_id, qa in qas.items():
                qa['id'] = qa_id
                # 为每个问题创建num_repeats个副本
                for i in range(num_repeats):
                    qa_copy = qa.copy()
                    qa_copy['repeat_idx'] = i  # 添加重复索引
                    data_list.append(qa_copy)

        return Dataset.from_list(data_list)


@LOAD_DATASET.register_module()
class RANAPSummaryDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            summaries = data['summaries']
            relevant_docs = data['relevant_docs']
            qa_pairs = data['qa_pairs']

            data_list = []
            for summary_id, doc_id in qa_pairs.items():
                if summary_id in summaries and doc_id in relevant_docs:
                    item = {
                        'id': summary_id,
                        'content': relevant_docs[doc_id],
                        'answer': summaries[summary_id]
                    }
                    data_list.append(item)

        return Dataset.from_list(data_list)


@ICL_EVALUATORS.register_module()
class RANAPF1SoreEvaluator(BaseEvaluator):
    def f1_score(self, prediction, ground_truth):
        prediction_tokens = prediction.split('，')
        ground_truth_tokens = remove_space(ground_truth).split('，')
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0

        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def score(self, predictions, references):
        """计算所有预测的平均F1分数"""
        if len(predictions) != len(references):
            return {
                'error': 'predictions和references长度不一致'
            }

        total_f1 = 0
        total = len(predictions)

        for prediction, reference in zip(predictions, references):
            f1 = self.f1_score(prediction, reference)
            total_f1 += f1

        avg_f1 = 100.0 * total_f1 / total

        return {'f1score': avg_f1}


@ICL_EVALUATORS.register_module()
class ChoiceGPassKEvaluator(BaseEvaluator):
    """G-passK评估器，用于计算在n次评测中至少k次正确的概率。

    Args:
        k (Union[int, Sequence[int]]): 需要计算的k值列表，例如k=[1,2]表示计算pass@1和pass@2。
            如果只传入一个整数，会被转换为单元素列表。
        n (int): 每个问题的评测次数，默认为5。
    """

    def __init__(self, k: Union[int, Sequence[int]] = (1, 2), n: int = 5) -> None:
        if not isinstance(k, Sequence):
            k = (k,)
        self.k = k
        self.n = n
        super().__init__()

    def preprocess(self, predictions: List, references: List, test_set: Dataset) -> List[int]:
        """预处理预测结果，计算每个问题的正确性。

        Args:
            predictions: 预测结果列表
            references: 参考答案列表
            test_set: 测试数据集

        Returns:
            List[int]: 每个预测的正确性（1表示正确，0表示错误）
        """
        if len(predictions) != len(references):
            raise ValueError('predictions和references长度不一致')

        # 按问题ID和重复索引分组
        grouped_predictions = defaultdict(list)
        grouped_references = defaultdict(list)

        for pred, ref, example in zip(predictions, references, test_set):
            q_id = example['id']
            grouped_predictions[q_id].append(pred)
            grouped_references[q_id].append(ref)

        # 计算每个问题的正确次数
        correct_counts = []
        for q_id in grouped_predictions:
            preds = grouped_predictions[q_id]
            refs = grouped_references[q_id]
            correct = sum(1 for p, r in zip(preds, refs) if p == r)
            correct_counts.append(correct)

        return correct_counts

    def score(self, predictions: List, references: List, test_set: Dataset) -> dict:
        """计算所有预测的pass@k分数。

        Args:
            predictions: 预测结果列表
            references: 参考答案列表
            test_set: 测试数据集

        Returns:
            dict: 包含不同k值下的pass@k分数
        """
        correct_counts = self.preprocess(predictions, references, test_set)

        # 计算不同k值下的pass@k
        pass_at_k = {}
        for k in self.k:
            if k <= self.n:
                pass_at_k[f'pass@{k}'] = sum(1 for c in correct_counts if c >= k) / len(correct_counts) * 100

        return pass_at_k
