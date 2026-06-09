import ast
import json
import re
from typing import Callable, Optional, Union

from opencompass.registry import TEXT_POSTPROCESSORS
from opencompass.utils.clean_jsonstr import clean_str_to_json


@TEXT_POSTPROCESSORS.register_module("general")
def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    if not isinstance(text, str) or not text.strip():
        return ""
    truncated_text = re.split(r"[\n,.]", text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r"[^\w\s]", "", truncated_text)

    # Remove article
    no_articles = re.sub(r"\b(a|an|the)\b", "", no_punctuation, flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r"\s+", " ", no_articles).strip()
    return cleaned_text


# 截取第一行文本，删除重复空格，保留字母、数字和特定符号
@TEXT_POSTPROCESSORS.register_module("general-en")
def general_en_postprocess(text: str) -> str:
    truncated_text = re.split(r"[\n]", text, 1)[0]

    no_punctuation = re.sub(r"[^\w\s\-(){}<>\[\]]", " ", truncated_text)

    cleaned_text = re.sub(r"\s+", " ", no_punctuation).strip()

    return cleaned_text


# 保留更多数学题常用符号
@TEXT_POSTPROCESSORS.register_module("general-math")
def general_math_postprocess(text: str) -> str:
    truncated_text = re.split(r"[\n]", text, 1)[0]

    no_punctuation = re.sub(r"[^\w\s\-(){}!.,+=^<>\[\]\\]", " ", truncated_text)

    cleaned_text = re.sub(r"\s+", " ", no_punctuation).strip()

    return cleaned_text


@TEXT_POSTPROCESSORS.register_module("general_cn")
def general_cn_postprocess(text: str) -> str:
    truncated_text = re.split(r"[\n.,]", text, 1)[0]

    no_punctuation = re.sub(r"[^\w\s]", "", truncated_text)

    no_articles = re.sub(r"\b(a|an|the)\b", "", no_punctuation, flags=re.IGNORECASE)

    cleaned_text = re.sub(r"\s+", " ", no_articles).strip()
    import jieba

    cleaned_text = " ".join(jieba.cut(text))
    return cleaned_text

#xxxxxxxx
# 将模型输出的 JSON 字段名映射为评估所需的标签名
# mapping 格式：{"模型输出字段名": "目标标签名"}
@TEXT_POSTPROCESSORS.register_module("field-mapping-json")
def field_mapping_json_postprocess(text: str, mapping: dict = None) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    if not mapping:
        return text
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return text
        mapped_data = {}
        for k, v in data.items():
            mapped_k = mapping.get(k, k)
            mapped_data[mapped_k] = v
        return json.dumps(mapped_data, ensure_ascii=False)
    except Exception:
        return text


# 返回第一个大写字母
@TEXT_POSTPROCESSORS.register_module("first-capital")
def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return ""


# 返回最后一个大写字母
@TEXT_POSTPROCESSORS.register_module("last-capital")
def last_capital_postprocess(text: str) -> str:
    for t in text[::-1]:
        if t.isupper():
            return t
    return ""


# 返回第一个匹配下面正则的内容
@TEXT_POSTPROCESSORS.register_module("first-option")
def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""
    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'答案是?\s?([{options}])',
        f'答案是?\s?：([{options}])',
        f'答案是?\s?:([{options}])',
        f'答案应该?是\s?([{options}])',
        f'答案应该?选\s?([{options}])',
        f'答案为\s?([{options}])',
        f'答案选\s?([{options}])',
        f'选择?\s?([{options}])',
        f'故选?\s?([{options}])'
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'[Tt]he answer is \(?([{options}])\)?',
        f'[Tt]he answer is option \(?([{options}])\)?',
        f'[Tt]he correct answer is \(?([{options}])\)?',
        f'[Tt]he correct answer is option \(?([{options}])\)?',
        f'[Tt]he answer to the question is \(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'(\s|^)[{options}](\s|$)',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'[{options}]',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ""


# 返回第一个连续的ABCD子串
@TEXT_POSTPROCESSORS.register_module("first-capital-multi")
def first_capital_postprocess_multi(text: str) -> str:
    match = re.search(r"([A-D]+)", text)
    if match:
        return match.group(1)
    return ""


# 匹配指定字母，按顺序组成字符串
@TEXT_POSTPROCESSORS.register_module("specified-options")
def extract_specified_options(text: str) -> str:
    options = re.findall(r"[A-G]", text)
    return "".join(sorted(options))


# 匹配全部option，返回最后一个
def last_option_postprocess(text: str, options: str) -> str:
    match = re.findall(rf"([{options}])", text)
    if match:
        return match[-1]
    return ""


# 返回第一个数字，包括负数、小数
def first_number_postprocess(text: str) -> float:
    """Return the first number in a string."""
    # regex pattern to match numbers (both integers and decimals)
    pattern = r"(-?\d*\.?\d+)"

    # search the string for the pattern
    match = re.search(pattern, text)

    # if a match is found, return it. Otherwise, return None.
    return float(match.group(1)) if match else None


# 提取回答中的所有大写字母，直接拼接
@TEXT_POSTPROCESSORS.register_module("multiple-select")
def multiple_select_postprocess(text: str) -> str:
    ret = set([t for t in text if t.isupper()])
    return "".join(sorted(ret))


@TEXT_POSTPROCESSORS.register_module("specific-xml-tag")
def xml_tag_postprocessor(text, tag):
    """Extracts content enclosed within a specified XML-style tag from a
    string.

    Args:
        texts: The input string containing XML-style tags.
        tag: The XML-style tag to extract content from (e.g., "<conclude>").  Must include the angle brackets.

    Returns:
        The content enclosed within the specified tag, or None if the tag is not found.
    """

    # Use a regular expression to find the content within the specified tag.  This handles cases where the tag might appear multiple times.
    matches = re.findall(
        rf"{tag}(.*?)</{tag[1:-1]}>", text, re.DOTALL
    )  # re.DOTALL allows . to match newline characters

    if matches:
        # Only keep the last one
        output = matches[
            -1
        ].strip()  # Extract the content and remove leading/trailing whitespace
    else:
        output = "NO ANSWER FOUND"

    return output


def general_eval_wrapper_postprocess(
    text: str, postprocess: Optional[Union[str, Callable]] = None, **kwargs
) -> str:
    """Wrapper for eval text repr. Especially for chatglmpro.

    Args:
        text(str): Text to be postprocessed.
        postprocess(Callable, optional): Original post processing function.
            Defaults to None.
        **kwargs: Other necessary kwargs for post processing function.
    """
    try:
        text = eval(text)
    except Exception:
        # in case empty input or other error, skip eval
        pass

    if postprocess:
        if isinstance(postprocess, str):
            postprocess = TEXT_POSTPROCESSORS.get(postprocess)
        return postprocess(text, **kwargs)
    else:
        return text


# 返回第一个匹配正则 answer_pattern 的文本
def match_answer_pattern(response_text: str, answer_pattern: str):
    match = re.search(answer_pattern, response_text)
    extracted_answer = match.group(1) if match else ""
    return extracted_answer


# 去除 </think>
def extract_non_reasoning_content(text):
    """
    Remove content within <think>...</think> tags and retain only the content after </think>.
    """
    # Use regular expression to find the closing </think> tag and keep content after it
    result = re.split(r"</think>", text, maxsplit=1)
    if len(result) > 1:
        return result[1].strip()  # Return content after </think>
    return text  # If </think> is not found, return the original text


@TEXT_POSTPROCESSORS.register_module("double-select")
def text2sql_remove_double_select(text: str) -> str:
    result_query = "SELECT " + text.replace("\n", " ")
    result_query = result_query.replace("SELECT SELECT", "SELECT")
    return result_query


def process_latex(text: str):
    text = text.replace("dfrac", "frac")
    return text


# 返回最后一个 "boxed{}" 里的内容
def extract_boxed_content(text: str) -> str:
    # 从字符串末尾向前查找最后一个 boxed{
    start = text.rfind(r"boxed{")
    if start == -1:
        return ""

    i = start + len(r"boxed{")  # 移动到内容起始位置
    stack = 1  # 栈初始化为1（已匹配开头的{）
    content = []

    while i < len(text) and stack > 0:
        if text[i] == "\\" and i + 1 < len(text):
            # 处理转义字符：保留反斜杠及下一个字符
            content.append(text[i])
            content.append(text[i + 1])
            i += 2  # 跳过已处理的下一个字符
            continue
        elif text[i] == "{":
            stack += 1  # 遇到{，栈深度增加
        elif text[i] == "}":
            stack -= 1  # 遇到}，栈深度减少
            if stack == 0:
                break  # 栈归零时立即终止
        # 只有当栈>0时才记录字符（避免记录最外层闭合括号）
        if stack > 0:
            content.append(text[i])
        i += 1

    return "".join(content).strip()


def latex_math_postprocess(text: str) -> str:
    text = extract_non_reasoning_content(text)
    boxed = extract_boxed_content(text)
    if boxed:
        return general_math_postprocess(boxed)
    phase = ["answer:", "answer is:", "final answer", "option"]
    pattern = "|".join(re.escape(p) for p in phase)
    parts = re.split(pattern, text.strip(), flags=re.IGNORECASE)
    last_part = parts[-1]
    ans_part = last_part.split(".", 1)[0]
    return general_math_postprocess(ans_part)


# 返回最后一个 \boxed{} 里的特定字母
def latex_last_option(text: str, options=None) -> str:
    if options is None:
        options = "ABCDabcd"

    matches = extract_boxed_content(text)
    if matches:
        content = matches
        # content 中第一个在 options 里的字符
        for char in content:
            if char in options:
                return char
    return first_option_postprocess(text)


# 返回最后一个 \boxed{} 里的英文内容
def latex_last_en(text: str) -> str:
    matches = extract_boxed_content(text)
    if matches:
        return general_en_postprocess(matches)
    return ""


# 返回最后一个 \boxed{} 里的第一个大写字母 / 分割可能的答案，返回第一个大写字母
def latex_last_scq(text: str) -> str:
    text = extract_non_reasoning_content(text)
    latex_matches = extract_boxed_content(text)
    if latex_matches:
        return first_capital_postprocess(latex_matches)
    else:
        phase = ["answer is", "final answer", "answer:"]
        pattern = re.compile("|".join(re.escape(p) for p in phase), re.IGNORECASE)
        parts = pattern.split(text)

        last_part = parts[-1] if parts else text
        return first_capital_postprocess(last_part)


# 返回最后一个 \boxed{} 里的全部大写字母 / 分割可能的答案，返回extract_specified_options
def latex_last_mcq(text: str) -> str:
    text = extract_non_reasoning_content(text)
    latex_matches = extract_boxed_content(text)
    if latex_matches:
        return extract_specified_options(latex_matches)
    phase = ["answer:", "answer is:", "final answer", "option"]
    pattern = "|".join(re.escape(p) for p in phase)
    parts = re.split(pattern, text.strip(), flags=re.IGNORECASE)
    last_part = parts[-1]
    ans_part = last_part.split(".", 1)[0]
    return extract_specified_options(ans_part)


def extract_last_boxed_robust(text):
    # 找到所有 \boxed{ 的位置
    start_positions = [m.start() for m in re.finditer(r"\\boxed\{", text)]

    if not start_positions:
        return None

    # 从最后一个开始处理
    last_start = start_positions[-1]
    start_index = last_start + len(r"\boxed{")

    # 找到最后一个 } 的位置
    last_brace_pos = text.rfind("}")

    if last_brace_pos == -1 or last_brace_pos <= start_index:
        return None

    # 返回最后一个 \boxed{ 和最后一个 } 之间的所有内容
    return text[start_index:last_brace_pos]


# 返回最后一个 boxed{} 里的内容 -> 分割，返回最后一个part的内容
def bbh_latex_freeform(text: str) -> str:
    text = extract_non_reasoning_content(text)
    boxed = extract_last_boxed_robust(text)
    if boxed:
        return general_en_postprocess(boxed)
    phase = ["answer is", "list is", "answer:", "list:"]
    pattern = "|".join(re.escape(p) for p in phase)
    parts = re.split(pattern, text.strip(), flags=re.IGNORECASE)
    last_part = parts[-1]
    return general_en_postprocess(last_part)


@TEXT_POSTPROCESSORS.register_module("extract-json-str")
def extract_json_str(text: str) -> str:
    result = text.replace("\n", "").replace("`", "")
    json_pattern = re.compile(r"\{.*\}")
    match = json_pattern.search(result)
    if match:
        result = match.group()
    return result

# 识别可转换成 dict 的 str，输出 str 格式的标准 json
def json_str(text: str):
    if not isinstance(text, str):
        text = str(text)

    replacements = {
        "\\n": "\n",
        "'": '"',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r'(?i)"(none|null)"', "null", text)
    text = re.sub(r"\s+", " ", text)

    # 查找JSON对象边界
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        return None

    json_str = text[start : end + 1]

    try:
        # 使用ast.literal_eval解析dict，并转换回JSON字符串
        parsed = ast.literal_eval(json_str)
        return json.dumps(parsed, ensure_ascii=False)
    except (ValueError, SyntaxError, json.JSONDecodeError):
        return json_str


# 识别可转换成 dict 的 str，输出标准格式的 json
def str2json(text):
    return clean_str_to_json(text)


def eoa_tag_postprocessor(text):
    text = extract_non_reasoning_content(text)
    pattern = re.compile(r'.*\[正确答案\](.*?)<eoa>(?![^<]*<eoa>)', re.DOTALL)
    match = pattern.search(text)
    if match:
        text = match.group(1).strip()
    text = multiple_select_postprocess(text)
    return text
