import ast
import json
import re
from typing import Any, List, Optional


def _extract_json_candidates(text: str) -> List[str]:
    if not text:
        return []
    candidates = []
    stack: List[str] = []
    start_idx: Optional[int] = None
    in_string = False
    string_quote = ""
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == string_quote:
                in_string = False
                string_quote = ""
            continue

        if ch in ("\"", "'"):
            in_string = True
            string_quote = ch
            continue

        if ch in ("{", "["):
            if not stack:
                start_idx = i
            stack.append(ch)
            continue

        if ch in ("}", "]") and stack:
            top = stack[-1]
            if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                stack.pop()
                if not stack and start_idx is not None:
                    candidates.append(text[start_idx : i + 1])
                    start_idx = None
            continue
    return candidates


def _strip_wrappers(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def clean_str_to_json(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return None

    candidates = _extract_json_candidates(text)
    parsed: List[Any] = []
    for candidate in candidates:
        try:
            parsed.append(json.loads(candidate))
            continue
        except json.JSONDecodeError:
            pass

        try:
            parsed.append(ast.literal_eval(candidate))
        except (ValueError, SyntaxError):
            continue

    if parsed:
        return parsed[-1]

    cleaned = _strip_wrappers(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return None
