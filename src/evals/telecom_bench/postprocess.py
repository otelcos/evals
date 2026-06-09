"""Verbatim-faithful ports of ZTE TeleCom-Bench text postprocessors.

Sources (vendored under data/telecom_bench/upstream_ref/):
  utils/text_postprocessors.py: multiple_select_postprocess, extract_non_reasoning_content, str2json
  utils/clean_jsonstr.py:       clean_str_to_json (str2json delegates here)
  zte_domain/ai_cs/alarm_nodes.py: are_json_equal
"""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from typing import Any


def multiple_select_postprocess(text: str) -> str:
    """Return sorted unique uppercase letters (the selected MCQ options)."""
    return "".join(sorted({t for t in text if t.isupper()}))


def extract_non_reasoning_content(text: str) -> str:
    """Drop a leading <think>...</think> block; keep content after </think>."""
    parts = re.split(r"</think>", text, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else text


def _extract_json_candidates(text: str) -> list[str]:
    """Scan for top-level {...}/[...] spans, ignoring brackets inside strings."""
    if not text:
        return []
    candidates: list[str] = []
    stack: list[str] = []
    start_idx: int | None = None
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

        if ch in ('"', "'"):
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
    """Strip <think>...</think> blocks and ```json fences, then trim."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def str2json(text: Any) -> Any | None:
    """Faithful port of clean_str_to_json: return the LAST parseable JSON value."""
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return None

    candidates = _extract_json_candidates(text)
    parsed: list[Any] = []
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


def are_json_equal(a: Any, b: Any) -> bool:
    """Order-insensitive deep equality (port of ai_cs/alarm_nodes.are_json_equal)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(are_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        if all(isinstance(x, dict) for x in a) and all(isinstance(x, dict) for x in b):
            try:
                sa = sorted(a, key=lambda d: tuple(sorted(d.items())))
                sb = sorted(b, key=lambda d: tuple(sorted(d.items())))
                return sa == sb
            except TypeError:
                return all(are_json_equal(x, y) for x, y in zip(a, b))
        return all(are_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def normalize_zh(text: str) -> str:
    """NFKC normalize (folds full-width to half-width) and strip."""
    return unicodedata.normalize("NFKC", text or "").strip()
