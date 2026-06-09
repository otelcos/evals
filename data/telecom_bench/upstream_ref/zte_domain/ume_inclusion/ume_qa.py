import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils.text_postprocessors import general_postprocess

import re

import pandas as pd
from rouge import Rouge
import jieba

#from zte.feedback.util.logutil import log

pd.options.mode.chained_assignment = None

from opencompass.datasets.base import BaseDataset


@LOAD_DATASET.register_module()
class UmeDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path) as f:
            reader = csv.reader(f)
            raw_data = []
            next(reader)
            for row in reader:
                assert len(row) == 3
                question = row[0]
                context = row[1]
                ground_truths = row[2]
                raw_data.append({'question': question, 'context': context, 'ground_truths':ground_truths})
            dataset['test'] = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module()
class UmeEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        total_recall_score = 0.0
        total_precision_score = 0.0
        for rouge_source_txt, rouge_target_txt in zip(predictions, references):
            score = self.get_rouge_scores(rouge_source_txt, rouge_target_txt)
            print(f"rouge score:{score[0]['rouge-l']}")
            print(f"rouge score:{score[0]['rouge-l']['r']}")
            total_recall_score += float(score[0]['rouge-l']['r'])
            total_precision_score += float(score[0]['rouge-l']['p'])

        return {"rouge-l-r":total_recall_score*100/len(predictions), "rouge-l-p":total_precision_score*100/len(predictions)}

    def get_rouge_scores(self, rouge_source_txt, rouge_target_txt):
        # source = llm_answer.replace(" ", "")  # 大模型答案
        # target = related_content.replace(" ", "")  # 分母，参考答案
        hyps = ' '.join(jieba.cut(rouge_source_txt.replace(" ", "")))   # 大模型答案
        refs = ' '.join(jieba.cut(rouge_target_txt.replace(" ", "")))  # 分母，参考答案
        # if len(source.strip(".")) == 0 or len(target.strip(".")) == 0:
        #     return [{'rouge-l': {'r': 0.0}}]
        rouge_temp = Rouge(['rouge-l'])
        try:
            return rouge_temp.get_scores(hyps, refs)
        except Exception as ex:
            #log.exception(ex)
            # rouge_l_r = slide_window_rouge(source, target)
            return [{'rouge-l': {'r': 0.0}}]

#     def get_rouge_scores(rouge_source_txt, rouge_target_txt):
#         # source = llm_answer.replace(" ", "")  # 大模型答案
#         # target = related_content.replace(" ", "")  # 分母，参考答案
#         hyps = ' '.join(jieba.cut(rouge_source_txt.replace(" ", "")))   # 大模型答案
#         refs = ' '.join(jieba.cut(rouge_target_txt.replace(" ", "")))  # 分母，参考答案
#         # if len(source.strip(".")) == 0 or len(target.strip(".")) == 0:
#         #     return [{'rouge-l': {'r': 0.0}}]
#         rouge_temp = Rouge(['rouge-l'])
#         try:
#             return rouge_temp.get_scores(hyps, refs)
#         except Exception as ex:
#             log.exception(ex)

#     def get_eval_value(self, rouge_source_txt, rouge_target_txt):
#         pattern = r'\n?\d+\s?[.,，、 ]'
#         index_list = [(m.start(), m.end()) for m in re.finditer(pattern, rouge_target_txt)]
#         no_str_list = []
#         for index in index_list:
#             no_str = rouge_target_txt[index[0]:index[1]]
#             no_str_list.append(no_str)

#         for no_str in no_str_list:
#             rouge_target_txt = rouge_target_txt.replace(no_str, "")

#         rouge_target_txt = rouge_target_txt.replace("\\n", "").replace("\n", "").replace(" ", "").replace("*", "")

#         index_list = [(m.start(), m.end()) for m in re.finditer(pattern, rouge_source_txt)]
#         no_str_list = []
#         for index in index_list:
#             no_str = rouge_source_txt[index[0]:index[1]]
#             no_str_list.append(no_str)

#         for no_str in no_str_list:
#             rouge_source_txt = rouge_source_txt.replace(no_str, "")

#         rouge_source_txt = rouge_source_txt.replace("\\n", "").replace("\n", "").replace(" ", "").replace("*", "")

#         result = get_rouge_scores(rouge_source_txt, rouge_target_txt)
#         try:
#             rouge_l_r = result[0]['rouge-l']
#             log.info("result:" + str(result))
#             return rouge_l_r
#         except Exception as e:
#             log.warning("Error: Unable to retrieve 'rouge-l' r value.")
#             log.exception(e)
