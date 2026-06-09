import re
from typing import List

from opencompass.datasets import BaseJudgeScoreEvaluator
from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import extract_non_reasoning_content, json_str, multiple_select_postprocess, str2json


def choice_postprocessor(text):
    text = extract_non_reasoning_content(text)
    pattern = re.compile(r'.*\[正确答案\](.*?)<eoa>(?![^<]*<eoa>)', re.DOTALL)
    match = pattern.search(text)
    if match:
        text = match.group(1).strip()
    text = multiple_select_postprocess(text)
    return text


def match_postprocessor(text):
    # text = extract_non_reasoning_content(text)
    text = json_str(text)
    return text


@ICL_EVALUATORS.register_module()
class AICSQAMatchEvaluator(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref in zip(predictions, references):
            pred = str2json(pred)
            ref = str2json(ref)
            correct = True

            if pred is None or ref is None:
                correct = False
            else:
                for key in ref:
                    if key not in pred:
                        correct = False
                        break

                    ref_val = ref[key]
                    pred_val = pred[key]

                    if isinstance(ref_val, list):
                        if sorted(pred_val) != sorted(ref_val):
                            correct = False
                            break
                    else:
                        if pred_val != ref_val:
                            correct = False
                            break

            is_correct.append(correct)
            processed_pred.append(pred)
            processed_gold.append(ref)

        accuracy = sum(is_correct) / len(is_correct) if is_correct else 0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": is_correct
            }
        }


ai_cs_score_prompt = """"
### 角色
@@ground_truth 是 @@question 的正确答案；@@answer 是智能问答机器人回答的答案
请你充当公正的裁判，对 @@answer 评分，分值必须大于等于 0 小于等于 3 分，得分不能为负数，如果出现负数，需要按照规则重新评分。

### 评分规则
1、准确性：
ground_truth 和 answer 内容完全一致，必须评为 3 分，如 ground_truth 为 “1”，answer 为 “1”，评分必须为 3 分。
答案必须正确反映 ground_truth 中的信息，任何与正确答案冲突或矛盾的信息，或不包含在正确答案中的信息，都会导致答案被评为 0 分。
如果答案中包含任何错误信息，即使只有一小部分，也应根据规则被评为 0 分。
2、完整性：答案需要包含所有关键信息。缺少关键信息会根据缺失内容的数量和重要性扣分。
3、相关性：答案应与问题直接相关，不包含不相关信息。包含少量不相关信息可能会扣分，而大量不相关信息可能导致答案被评为 0 分。
4、逻辑性：
答案应逻辑清晰连贯，与问题相符，无胡说或复读情况。逻辑混乱或无意义的答案将被评为 0 分。
如果答案的开头与问题暗示的预期相反，即使后面的陈述正确，答案也应被评为 0 分。
5、时效性：如果问题或答案涉及时效性信息，答案必须准确反映这一时效性，错误或遗漏时效性信息可能会扣分。
6、条理性：答案应条理清晰，组织有序。如果答案内容虽全面但组织混乱，应适当扣分。
7、信息量：答案应避免冗余，即使是正确的信息，如果造成答案冗余也可能影响评分。
8、语境理解：答案必须考虑问题的语境，包括肯定、否定或反问形式，并正确理解其隐含意义。
9、反问句的回答：当问题以反问形式出现时，期望的答案应是直接否定或肯定问题中的假设。如果答案肯定了问题中的不正确假设，即使后面的内容正确，也应被评为 0 分。
10、预期答案的识别：评分时需识别并评估答案是否符合问题所隐含的预期答案，特别是当问题以反问形式出现时。
11、间接信息的准确性：答案不仅需要直接内容上的准确性，还必须确保其隐含信息与问题和
12、语境相关的逻辑性评估：答案的逻辑性评估不仅基于答案内部逻辑，还需考虑答案与问题语境的逻辑一致性

### 评分梯度
0 分：内容完全错误、严重缺失或与问题无关；无逻辑；或明确表示信息不可用。或直接回答 “在提供的【参考文本】中，并没有直接提及...”
1 分：包含部分正确答案的核心内容，但关键信息缺失或错误，或包含大量不相关信息。
2 分：内容基本准确，包含大部分问题核心内容，但不全面，缺少一部分关键信息或包含少量不相关信息。
3 分：内容完整，准确，覆盖所有问题核心内容，无错误信息，无额外不相关信息，条理清晰。

### 输出规则
请严格遵循以下示例的格式输出评分结果："[[rating]]", 示例: "Rating: [[0]]"。

### 示例
@@Example1
##question
苹果手机下载慢
##ground_truth
1
##answer 0
1
"Rating: [[3]]"
3 分，期望的答案和答案完全一致，直接评为 3 分。
@@Example2
##question
在 M6000-2S16 中，2100GE 光口板可插入哪个槽位，推荐在哪个槽位使用？
##ground_truth
在 M6000-2S16 中，2100GE 光口板的适配槽位是 3，7，8，13，15，16。推荐槽位是 7，8，3，13。
##answer 0
在 M6000-2S16 中，2100GE 光口板的适配槽位是 17,18。推荐槽位是 17.
"Rating: [[0]]"
0 分，给出的答案内容，不包含在 ground_truth，不准确
2 端口 100GE QSFP28 光口板（RPN-02CG-H）可插入 M6000-2S16 的槽位 3、7、8、13。推荐在槽位 7、8、3、13 使用。
"Rating: [[1]]"
1 分，给出的答案内容，缺少部分 ground_truth 的核心内容，不准确
在 M6000-2S16 中，2100GE 光口板的适配槽位是 3，7，8，13，15，16。推荐槽位是 7，8，3，13。ZXR10 9900 系列交换机在数据中心汇聚层的应用主要体现在基于 CLOS 架构的树型组网中。这种组网方式是为了解决传统三层组网模型下高收敛带来的东西向流量瓶颈问题。在树型组网中，服务器通过 GE 或 10GE 接口接入到 TOR（Top Of Rack）交换机或 EOR（End Of Rack）交换机，而 ZXR10 9900 作为汇聚层的核心交换机，负责将来自多个接入层交换机的数据进行汇聚和转发。
"Rating: [[1]]"
1 分，给出的答案包含了 ground_truth 的核心内容，但包含了大量的不相关信息，不相关
在 M6000-2S16 中，2*100GE 光口板的适配槽位是 3，7，8，13，15，16。推荐槽位是 7，8，3，13。
"Rating: [[3]]"
3 分，给出的答案内容准确，虽然答案后面包含了 “回车符”，但是不影响答案的准确性
@@Example3
##question
M6000-S 是否支持 MACSec 功能
##ground_truth
M6000-S 不支持 MACSec 功能（Remarks：预计在 2023Q4 10GE 端口可进行测试）
##answer 0
ZXR10 M6000-S 不支持 MACSec 功能。
"Rating: [[1]]"
1 分，给出的答案内容缺少 “2023Q4 10GE 端口可进行测试”，缺少关键信息扣分，不全面
@@Example4
##question
请介绍 ZXR10 9900 系列交换机在数据中心汇聚层的应用
##ground_truth
ZXR10 9900-S 系列大容量数据中心交换机适用小型于数据中心网络的核心、汇聚和接入层，高端城域网园区的核心层。 在园区核心层的应用：高端核心交换机 ZXR10 9900 作为新一代核心交换机，转发性能高、单位能耗低、支持高密 10GE，因此当园 区汇聚采用 10GE 上行时，ZXR10 9900 也可以作为园区核心节点来使用。
##answer 0
ZXR10 9900 系列交换机在数据中心汇聚层的应用主要体现在基于 CLOS 架构的树型组网中。这种组网方式是为了解决传统三层组网模型下高收敛带来的东西向流量瓶颈问题。在树型组网中，服务器通过 GE 或 10GE 接口接入到 TOR（Top Of Rack）交换机或 EOR（End Of Rack）交换机，而 ZXR10 9900 作为汇聚层的核心交换机，负责将来自多个接入层交换机的数据进行汇聚和转发。
"Rating: [[1]]"
1 分，问题的核心内容是 “XX 的应用”，所以回答的内容应该围绕应用场景来回答，而给出的答案是回答组网互联或协议相关内容，与问题不相关
@@Example5
##question
ZXR10 M6000-S CGN 功能是否支持 NAT46？
##ground_truth
In accordance with the problems in IPv4 address exhaustion and compatibility of IPv4 and IPv6, ZXR10 M6000-S can provide Carrier-Grade NAT (CGN) function:
lSupport NAT44
lSupport NAT LOG
lSupport NAT ALG
lSupport VPN NAT
lSupport multi-egress NAT
lSupport CGN hot-standby
##answer 0
是的，ZXR10 M6000-S 支持 NAT46。
"Rating: [[0]]"
0 分，答案中的 “ZXR10 M6000-S 支持 NAT46”，不包含在 ground_truth 中，答案被评为 0 分。
@@Example6
##question
M6000-S 产品的 V3 和 V5 的线卡不一定要匹配对应的子卡吗？
##ground_truth
错误，M6000-S 产品的 V3 和 V5 的线卡必须匹配对应子卡。
##answer 0
是的，M6000-S 产品的 V3 和 V5 的线卡必须匹配对应子卡。
"Rating: [[0]]"
0 分，当问题以反问形式出现时，期望的答案应是直接否定问题中的假设。如果答案肯定了问题中的不正确假设，即使后面的内容正确，也应被评为 0 分

[Question]
{question}

[The Start of Answer]
{prediction}
[The End of Answer]

[The Start of Ground Truth]
{reference}
[The End of Ground Truth]
/no_think
"""


@ICL_EVALUATORS.register_module()
class AICSScoreEvaluator(BaseJudgeScoreEvaluator):
    def __init__(self, prompt=None, judge_model=None):
        super().__init__(
            prompt=prompt,
            judge_model=judge_model,
            score_levels=[0, 1, 2, 3])

    def _get_prompt(self) -> str:
        prompt = ai_cs_score_prompt
        return prompt

    def _get_judge_model(self):
        if self._judge_model is not None:
            return self._judge_model
        from opencompass.judge_models.judge_qwen3 import Qwen3
        return Qwen3()

    def _extract_judge(self, judge_message: str):
        try:
            judge_message = extract_non_reasoning_content(judge_message)
            match = re.search(r'\[\[(\d+)\]\]', judge_message.strip())
            if match:
                score = int(match.group(1))
                if score in self._score_levels:
                    return score
            return None
        except Exception:
            return None


@ICL_EVALUATORS.register_module()
class AICSScoreNteleEvaluator(BaseJudgeScoreEvaluator):
    def __init__(self, prompt=None, judge_model=None):
        super().__init__(
            prompt=prompt,
            judge_model=judge_model,
            score_levels=[0, 1, 2, 3])

    def _get_prompt(self) -> str:
        prompt = ai_cs_score_prompt
        return prompt

    def _get_judge_model(self):
        if self._judge_model is not None:
            return self._judge_model
        from opencompass.judge_models import NTele72B
        return NTele72B()

    def _extract_judge(self, judge_message: str):
        try:
            judge_message = extract_non_reasoning_content(judge_message)
            match = re.search(r'\[\[(\d+)\]\]', judge_message.strip())
            if match:
                score = int(match.group(1))
                if score in self._score_levels:
                    return score
            return None
        except Exception:
            return None
