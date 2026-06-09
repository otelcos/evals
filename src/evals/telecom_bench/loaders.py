"""Generic JSON loading helpers for telecom_bench sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def render_mcq(record: dict, letters: tuple[str, ...] = ("A", "B", "C", "D")) -> str:
    """Render an MCQ record's stem + options into a single prompt string."""
    stem = record["question"]
    lines = [
        f"{ltr}. {record[ltr]}" for ltr in letters if ltr in record and record[ltr]
    ]
    return stem + "\n" + "\n".join(lines)
