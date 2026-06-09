# telecom_bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recreate ZTE TeleCom-Bench (arXiv 2605.18025) as a native Inspect AI benchmark `telecom_bench` with 12 evaluation sets, faithful static scoring, and a tri-expert configurable LLM-judge panel.

**Architecture:** A shared library (postprocessors + three reusable scorers) plus 12 thin `@task` modules that load released example JSON and score offline. No agent, no sandbox. Phase 0 builds the shared library and one reference task by hand (TDD); Phase 1 fans out a `Workflow` of 11 agents that clone the reference for the remaining sets; Phase 2 registers, runs, and reports.

**Tech Stack:** Python 3.10+, Inspect AI (`inspect_ai`), pytest, uv. Design: `docs/plans/2026-06-08-telecom-bench-design.md`.

---

## File structure

```
src/evals/telecom_bench/
  __init__.py
  config.py                 # paths + dataset locations (single source of truth)
  _types.py                 # TypedDicts for raw records
  postprocess.py            # multiple_select_postprocess, extract_non_reasoning_content, str2json, are_json_equal, normalize_zh
  loaders.py                # load_json + generic helpers
  scorers/
    __init__.py
    multiselect_f1.py        # Macro-F1 + exact-set accuracy (MCQ)
    structured_em.py         # str2json + are_json_equal (JSON) | normalized string EM
    judge_panel.py           # tri-expert configurable Likert panel
  application/
    __init__.py
    intent_recognition.py    # Phase 0 reference template
    entity_extraction.py     # Phase 1
    tool_invocation.py        # Phase 1
    event_verification.py     # Phase 1
    root_cause_diagnosis.py   # Phase 1
    solution_generation.py    # Phase 1
  comprehension/
    __init__.py
    basic_knowledge.py        # Phase 1
    network_5g.py             # Phase 1
    protocols_3gpp.py         # Phase 1
    wireless_network.py       # Phase 1
    wired_network.py          # Phase 1
    core_network.py           # Phase 1
  SETS.md                    # per-set spec sheet consumed by the Phase 1 swarm
  evaluation/
    REPORT.md                # Phase 2
    IMPLEMENTATION.md         # Phase 2 (fidelity-decision log)
data/telecom_bench/
  datasets/                  # vendored released example JSON
  upstream_ref/              # vendored ZTE evaluator files (fidelity reference)
src/tests/telecom_bench/
  test_postprocess.py
  test_multiselect_f1.py
  test_structured_em.py
  test_judge_panel.py
  test_intent_recognition.py
  test_<set>.py              # one per Phase 1 set
```

Registry wiring (`src/evals/_registry.py`, `src/evals/__init__.py`) is centralized in Task 11, never touched by swarm agents.

---

## Phase 0: scaffold + reference task (solo, TDD)

### Task 0: Branch, dirs, vendor data + upstream reference

**Files:**
- Create: directory tree above (empty `__init__.py` files)
- Create: `data/telecom_bench/datasets/`, `data/telecom_bench/upstream_ref/`

- [ ] **Step 1: Confirm branch + enable pre-commit**

```bash
cd /Users/emolero/Documents/GitHub/ot/evals
git branch --show-current   # expect: feat/telecom-bench
uv run pre-commit install
```
Expected: `pre-commit installed at .git/hooks/pre-commit`

- [ ] **Step 2: Create package directories**

```bash
mkdir -p src/evals/telecom_bench/scorers src/evals/telecom_bench/application \
  src/evals/telecom_bench/comprehension src/evals/telecom_bench/evaluation \
  src/tests/telecom_bench data/telecom_bench/datasets data/telecom_bench/upstream_ref
touch src/evals/telecom_bench/__init__.py src/evals/telecom_bench/scorers/__init__.py \
  src/evals/telecom_bench/application/__init__.py src/evals/telecom_bench/comprehension/__init__.py \
  src/tests/telecom_bench/__init__.py
```

- [ ] **Step 3: Vendor released example data and ZTE evaluator reference**

Clone upstream (ephemeral) and copy the pieces we depend on into the repo so the build is reproducible without `/tmp`.

```bash
git clone --depth 1 https://github.com/ZTE-AICloud/TeleCom-Bench.git /tmp/TeleCom-Bench 2>/dev/null || true
cp -R /tmp/TeleCom-Bench/datasets/* data/telecom_bench/datasets/
mkdir -p data/telecom_bench/upstream_ref/utils data/telecom_bench/upstream_ref/zte_domain
cp /tmp/TeleCom-Bench/code/opencompass/utils/text_postprocessors.py data/telecom_bench/upstream_ref/utils/
cp /tmp/TeleCom-Bench/code/opencompass/utils/clean_jsonstr.py data/telecom_bench/upstream_ref/utils/
cp -R /tmp/TeleCom-Bench/code/opencompass/datasets/zte_domain/* data/telecom_bench/upstream_ref/zte_domain/
```

- [ ] **Step 4: Commit scaffold**

```bash
git add src/evals/telecom_bench data/telecom_bench src/tests/telecom_bench
git commit -m "chore(telecom_bench): scaffold package, vendor example data + upstream eval ref"
```

---

### Task 1: Port ZTE postprocessors (`postprocess.py`)

**Files:**
- Create: `src/evals/telecom_bench/postprocess.py`
- Test: `src/tests/telecom_bench/test_postprocess.py`

- [ ] **Step 1: Write the failing tests**

```python
# src/tests/telecom_bench/test_postprocess.py
from evals.telecom_bench.postprocess import (
    multiple_select_postprocess,
    extract_non_reasoning_content,
    str2json,
    are_json_equal,
    normalize_zh,
)


def test_multiple_select_extracts_sorted_unique_uppercase():
    assert multiple_select_postprocess("the answer is C and A") == "AC"


def test_extract_non_reasoning_strips_think():
    assert extract_non_reasoning_content("<think>x</think>final") == "final"
    assert extract_non_reasoning_content("no tags") == "no tags"


def test_str2json_parses_embedded_object():
    assert str2json('blah {"a": 1} tail') == {"a": 1}


def test_str2json_returns_last_candidate():
    assert str2json('{"a":1} then {"b":2}') == {"b": 2}


def test_str2json_none_on_garbage():
    assert str2json("not json at all") is None


def test_are_json_equal_order_insensitive_list_of_dicts():
    a = [{"x": 1}, {"y": 2}]
    b = [{"y": 2}, {"x": 1}]
    assert are_json_equal(a, b) is True


def test_are_json_equal_detects_difference():
    assert are_json_equal({"a": 1}, {"a": 2}) is False


def test_normalize_zh_folds_fullwidth():
    assert normalize_zh("ＡＢＣ　") == "ABC"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest src/tests/telecom_bench/test_postprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: evals.telecom_bench.postprocess`

- [ ] **Step 3: Implement `postprocess.py`**

```python
# src/evals/telecom_bench/postprocess.py
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

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OPENERS = "{["
_CLOSERS = {"}": "{", "]": "["}


def multiple_select_postprocess(text: str) -> str:
    """Return sorted unique uppercase letters (the selected MCQ options)."""
    return "".join(sorted({t for t in text if t.isupper()}))


def extract_non_reasoning_content(text: str) -> str:
    """Drop a leading <think>...</think> block; keep content after </think>."""
    parts = re.split(r"</think>", text, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else text


def _json_candidates(text: str) -> list[str]:
    candidates = [m.group(1).strip() for m in _FENCE.finditer(text)]
    stack: list[str] = []
    start: int | None = None
    for i, ch in enumerate(text):
        if ch in _OPENERS:
            if not stack:
                start = i
            stack.append(ch)
        elif ch in _CLOSERS and stack:
            if stack[-1] == _CLOSERS[ch]:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start : i + 1])
                    start = None
            else:
                stack.clear()
                start = None
    return candidates


def str2json(text: Any) -> Any | None:
    """Faithful port of clean_str_to_json: return the LAST parseable JSON value."""
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return None
    parsed: list[Any] = []
    for candidate in _json_candidates(text):
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
    cleaned = text.strip().strip("`").strip()
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest src/tests/telecom_bench/test_postprocess.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/evals/telecom_bench/postprocess.py src/tests/telecom_bench/test_postprocess.py
git commit -m "feat(telecom_bench): port ZTE postprocessors with tests"
```

---

### Task 2: Config + types

**Files:**
- Create: `src/evals/telecom_bench/config.py`
- Create: `src/evals/telecom_bench/_types.py`

- [ ] **Step 1: Implement `config.py`**

```python
# src/evals/telecom_bench/config.py
"""telecom_bench paths and dataset locations. Read this to understand the eval."""

from pathlib import Path

# repo_root/src/evals/telecom_bench/config.py -> parents[3] == repo root
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "telecom_bench" / "datasets"
UPSTREAM_REF = Path(__file__).resolve().parents[3] / "data" / "telecom_bench" / "upstream_ref"

KC = DATA_DIR / "Knowledge_Comprehension"
KA = DATA_DIR / "Knowledge_Application"

# Default judge panel: three calls to the active model. Override per-run.
DEFAULT_JUDGES: list[str | None] = [None, None, None]
```

- [ ] **Step 2: Implement `_types.py`**

```python
# src/evals/telecom_bench/_types.py
"""Shared record shapes for telecom_bench loaders."""

from __future__ import annotations

from typing import TypedDict


class MCQRecord(TypedDict, total=False):
    id: int
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: str


class QARecord(TypedDict, total=False):
    question: str
    answer: str
    best_answer: str
```

- [ ] **Step 3: Commit**

```bash
git add src/evals/telecom_bench/config.py src/evals/telecom_bench/_types.py
git commit -m "feat(telecom_bench): add config paths and record types"
```

---

### Task 3: `multiselect_f1` scorer

**Files:**
- Create: `src/evals/telecom_bench/scorers/multiselect_f1.py`
- Test: `src/tests/telecom_bench/test_multiselect_f1.py`

- [ ] **Step 1: Write failing tests**

```python
# src/tests/telecom_bench/test_multiselect_f1.py
from evals.telecom_bench.scorers.multiselect_f1 import f1, options_of


def test_options_of_extracts_letters():
    assert options_of("AC") == {"A", "C"}


def test_f1_perfect():
    assert f1({"A", "C"}, {"A", "C"}) == 1.0


def test_f1_partial():
    # pred {A,B} vs gold {A,C}: p=0.5, r=0.5 -> F1=0.5
    assert f1({"A", "B"}, {"A", "C"}) == 0.5


def test_f1_no_overlap():
    assert f1({"B"}, {"A", "C"}) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest src/tests/telecom_bench/test_multiselect_f1.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scorer**

```python
# src/evals/telecom_bench/scorers/multiselect_f1.py
"""Macro-F1 + exact-set accuracy for multi-select MCQ (faithful to ZTE)."""

from __future__ import annotations

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Target,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from evals.telecom_bench.postprocess import (
    extract_non_reasoning_content,
    multiple_select_postprocess,
)


def options_of(text: str) -> set[str]:
    return set(multiple_select_postprocess(text))


def f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)


@metric
def macro_f1() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        vals = [float(s.score.metadata.get("f1", 0.0)) for s in scores]
        return sum(vals) / len(vals) if vals else 0.0

    return compute


@scorer(metrics=[accuracy(), stderr(), macro_f1()])
def multiselect_f1_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        pred = options_of(extract_non_reasoning_content(state.output.completion))
        gold = options_of(target.text)
        exact = pred == gold
        return Score(
            value=CORRECT if exact else INCORRECT,
            answer="".join(sorted(pred)),
            metadata={"f1": f1(pred, gold), "pred": sorted(pred), "gold": sorted(gold)},
        )

    return score
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest src/tests/telecom_bench/test_multiselect_f1.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/evals/telecom_bench/scorers/multiselect_f1.py src/tests/telecom_bench/test_multiselect_f1.py
git commit -m "feat(telecom_bench): multiselect Macro-F1 scorer"
```

---

### Task 4: `structured_em` scorer

**Files:**
- Create: `src/evals/telecom_bench/scorers/structured_em.py`
- Test: `src/tests/telecom_bench/test_structured_em.py`

- [ ] **Step 1: Write failing tests**

```python
# src/tests/telecom_bench/test_structured_em.py
from evals.telecom_bench.scorers.structured_em import judge_correct


def test_json_mode_equal():
    assert judge_correct('{"a": 1}', '{"a": 1}', mode="json") is True


def test_json_mode_unequal():
    assert judge_correct('{"a": 2}', '{"a": 1}', mode="json") is False


def test_exact_mode_normalizes():
    assert judge_correct("ＤＯＮＥ", "DONE", mode="exact") is True


def test_pre_callable_applied():
    pred = "Thought: x\nOutput: DONE\nThought: y"
    assert judge_correct(
        pred, "DONE", mode="exact",
        pre=lambda t: t.split("Output:")[-1].split("\nThought:")[0],
    ) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest src/tests/telecom_bench/test_structured_em.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scorer**

```python
# src/evals/telecom_bench/scorers/structured_em.py
"""Structured Exact Match: JSON equality (str2json + are_json_equal) or string EM."""

from __future__ import annotations

from collections.abc import Callable

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from evals.telecom_bench.postprocess import (
    are_json_equal,
    extract_non_reasoning_content,
    normalize_zh,
    str2json,
)


def judge_correct(
    completion: str,
    target: str,
    *,
    mode: str = "json",
    pre: Callable[[str], str] | None = None,
) -> bool:
    raw = extract_non_reasoning_content(completion)
    if pre is not None:
        raw = pre(raw)
    if mode == "json":
        pred = str2json(raw)
        gold = str2json(target)
        return pred is not None and gold is not None and are_json_equal(pred, gold)
    return normalize_zh(raw) == normalize_zh(target)


@scorer(metrics=[accuracy(), stderr()])
def structured_em_scorer(mode: str = "json", pre: Callable[[str], str] | None = None):
    async def score(state: TaskState, target: Target) -> Score:
        correct = judge_correct(state.output.completion, target.text, mode=mode, pre=pre)
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=extract_non_reasoning_content(state.output.completion)[:500],
            metadata={"mode": mode, "correct": correct},
        )

    return score
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest src/tests/telecom_bench/test_structured_em.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/evals/telecom_bench/scorers/structured_em.py src/tests/telecom_bench/test_structured_em.py
git commit -m "feat(telecom_bench): structured exact-match scorer"
```

---

### Task 5: `judge_panel` scorer (tri-expert, configurable)

**Files:**
- Create: `src/evals/telecom_bench/scorers/judge_panel.py`
- Test: `src/tests/telecom_bench/test_judge_panel.py`

- [ ] **Step 1: Write failing tests**

```python
# src/tests/telecom_bench/test_judge_panel.py
from evals.telecom_bench.scorers.judge_panel import parse_likert, aggregate


def test_parse_likert_finds_digit():
    assert parse_likert("评分：4") == 4
    assert parse_likert("the score is 5/5") == 5


def test_parse_likert_none_when_absent():
    assert parse_likert("no number here") is None


def test_aggregate_mean_and_spread():
    norm, mean_likert, spread = aggregate([5, 3, 4])
    assert mean_likert == 4.0
    assert norm == 0.75  # (4-1)/4
    assert spread == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest src/tests/telecom_bench/test_judge_panel.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scorer**

```python
# src/evals/telecom_bench/scorers/judge_panel.py
"""Tri-expert configurable LLM-judge panel (5-point Likert), faithful to ZTE.

Mirrors BaseJudgeACCEvaluator: each judge scores 1-5 against the reference;
we report the normalized mean as the Score value and the raw Likert mean +
inter-judge spread as metrics.
"""

from __future__ import annotations

import re

from inspect_ai.model import get_model
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Target,
    mean,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from evals.telecom_bench.config import DEFAULT_JUDGES

LIKERT_RUBRIC = """你是通信领域的资深专家评审。请根据[参考答案]评估[模型回答]的质量。
评分标准（5分制）：
5 = 完全正确、完整、专业；
4 = 基本正确，少量遗漏；
3 = 部分正确，有明显遗漏或错误；
2 = 大部分错误；
1 = 完全错误或答非所问。
[问题]
{question}
[参考答案]
{reference}
[模型回答]
{answer}
请只输出一个1到5之间的整数分数，不要输出任何其他内容。"""


def parse_likert(text: str) -> int | None:
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None


def aggregate(scores: list[int]) -> tuple[float, float, int]:
    mean_likert = sum(scores) / len(scores)
    norm = (mean_likert - 1) / 4
    spread = max(scores) - min(scores)
    return norm, mean_likert, spread


@metric
def mean_likert_metric() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        vals = [
            float(s.score.metadata["likert_mean"])
            for s in scores
            if s.score.metadata.get("likert_mean") is not None
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    return compute


@scorer(metrics=[mean(), stderr(), mean_likert_metric()])
def judge_panel_scorer(judges: list[str | None] | None = None, single: bool = False):
    panel = list(judges) if judges is not None else list(DEFAULT_JUDGES)
    if single:
        panel = panel[:1]

    async def score(state: TaskState, target: Target) -> Score:
        prompt = LIKERT_RUBRIC.format(
            question=state.input_text,
            reference=target.text,
            answer=state.output.completion,
        )
        raw: list[int] = []
        for judge in panel:
            out = await get_model(judge).generate(prompt)
            parsed = parse_likert(out.completion)
            if parsed is not None:
                raw.append(parsed)
        if not raw:
            return Score(value=0.0, answer="no judge score", metadata={"likert_mean": None})
        norm, mean_likert, spread = aggregate(raw)
        return Score(
            value=norm,
            answer=str(mean_likert),
            metadata={"likert_mean": mean_likert, "panel": raw, "spread": spread},
        )

    return score
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest src/tests/telecom_bench/test_judge_panel.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/evals/telecom_bench/scorers/judge_panel.py src/tests/telecom_bench/test_judge_panel.py
git commit -m "feat(telecom_bench): tri-expert configurable judge panel scorer"
```

---

### Task 6: Loaders helper

**Files:**
- Create: `src/evals/telecom_bench/loaders.py`

- [ ] **Step 1: Implement `loaders.py`**

```python
# src/evals/telecom_bench/loaders.py
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
    lines = [f"{ltr}. {record[ltr]}" for ltr in letters if ltr in record and record[ltr]]
    return stem + "\n" + "\n".join(lines)
```

- [ ] **Step 2: Commit**

```bash
git add src/evals/telecom_bench/loaders.py
git commit -m "feat(telecom_bench): json loading helpers"
```

---

### Task 7: Reference task `intent_recognition` (end-to-end)

**Files:**
- Create: `src/evals/telecom_bench/application/intent_recognition.py`
- Test: `src/tests/telecom_bench/test_intent_recognition.py`
- Modify: `src/evals/_registry.py`, `src/evals/__init__.py`

Released data: `data/telecom_bench/datasets/Knowledge_Application/Intent_Recognition/intent_recognition.json` (list of `{id, summary, input, ...}`; the gold label lives in a field the implementer must confirm by inspecting the file). ZTE scorer: `upstream_ref/zte_domain/IDA/intent_recognize.py` (splits on `Output:` / `\nThought:`, then `str2json` + dict equality; classes DONE/UNDONE/ORDER/NO).

- [ ] **Step 1: Inspect the data to confirm the gold field**

Run: `uv run python -c "import json; d=json.load(open('data/telecom_bench/datasets/Knowledge_Application/Intent_Recognition/intent_recognition.json')); print(list(d[0].keys())); print(d[0])"`
Expected: prints the record keys; identify the answer/label key (e.g. `answer` or `output`). Use that key name as `GOLD_KEY` below.

- [ ] **Step 2: Write the failing tests**

```python
# src/tests/telecom_bench/test_intent_recognition.py
from evals.telecom_bench.application.intent_recognition import record_to_sample, INTENT_PRE


def test_record_to_sample_input_and_target():
    rec = {"id": "q_0000", "input": "请改善黄家庄村的高负荷问题", "answer": "DONE"}
    s = record_to_sample(rec)
    assert "黄家庄村" in s.input
    assert s.target == "DONE"


def test_intent_pre_extracts_output_segment():
    assert INTENT_PRE("Thought: a\nOutput: NO\nThought: b") == "NO"
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest src/tests/telecom_bench/test_intent_recognition.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the task**

Replace `GOLD_KEY` with the key confirmed in Step 1 (the plan assumes `"answer"`).

```python
# src/evals/telecom_bench/application/intent_recognition.py
"""TeleCom-Bench Knowledge Application: Intent Recognition (faithful static).

Reference: upstream_ref/zte_domain/IDA/intent_recognize.py
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from evals.telecom_bench.config import KA
from evals.telecom_bench.loaders import load_json
from evals.telecom_bench.scorers.structured_em import structured_em_scorer

GOLD_KEY = "answer"
DATA_FILE = KA / "Intent_Recognition" / "intent_recognition.json"


def INTENT_PRE(text: str) -> str:
    """ZTE preprocessing: keep the Output: segment before the next Thought:."""
    return text.split("Output:")[-1].split("\nThought:")[0].strip()


def record_to_sample(record: dict) -> Sample:
    return Sample(
        id=str(record.get("id", "")),
        input=record["input"],
        target=str(record[GOLD_KEY]),
        metadata={"set": "intent_recognition", "raw": record},
    )


def load_dataset() -> list[Sample]:
    raw = load_json(DATA_FILE)
    records = raw if isinstance(raw, list) else raw.get("questions", [])
    return [record_to_sample(r) for r in records if GOLD_KEY in r]


@task
def telecom_bench_intent_recognition() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=structured_em_scorer(mode="exact", pre=INTENT_PRE),
    )
```

- [ ] **Step 5: Run to verify tests pass**

Run: `uv run pytest src/tests/telecom_bench/test_intent_recognition.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Register the task**

In `src/evals/_registry.py` add the import (keep alphabetical-ish grouping):

```python
from evals.telecom_bench.application.intent_recognition import telecom_bench_intent_recognition
```
and add `"telecom_bench_intent_recognition"` to its `__all__`.

In `src/evals/__init__.py` add `"telecom_bench"` to `__all__` (lazy-import list) if not present.

- [ ] **Step 7: Verify the task is discoverable and runs on the examples**

Run: `uv run inspect eval evals/telecom_bench_intent_recognition --model mockllm/model --limit 5`
Expected: completes; produces a log with accuracy reported (value depends on the mock).

- [ ] **Step 8: Commit**

```bash
git add src/evals/telecom_bench/application/intent_recognition.py \
  src/tests/telecom_bench/test_intent_recognition.py src/evals/_registry.py src/evals/__init__.py
git commit -m "feat(telecom_bench): intent_recognition reference task + registry"
```

---

## Phase 1: swarm the remaining 11 sets

### Task 8: Write the per-set spec sheet `SETS.md`

**Files:**
- Create: `src/evals/telecom_bench/SETS.md`

- [ ] **Step 1: Write `SETS.md`** with this exact content (the swarm consumes it):

````markdown
# telecom_bench set specifications

Each agent builds ONE set by cloning `application/intent_recognition.py`. Reuse the shared
scorers in `scorers/`. Do NOT edit `_registry.py` (Task 11 handles registration). Confirm the
gold field and record shape by inspecting the data file before coding. Write a test mirroring
`src/tests/telecom_bench/test_intent_recognition.py` (golden record scores correct; wrong scores 0).

| module | data file (under data/telecom_bench/datasets/) | record shape | input | target | scorer | upstream ref |
|---|---|---|---|---|---|---|
| application/entity_extraction.py | Knowledge_Application/Entity_Extraction/entity_extraction.json | list `{id,question,answer(JSON str),...}` | `question` | `answer` | `structured_em_scorer(mode="json")` | zte_domain/IDA/parameter_extract.py |
| application/event_verification.py | Knowledge_Application/Event_Verification/event_verification.json | dict `{question,best_answer}` (wrap as 1-item list) | `question` | `best_answer` | `judge_panel_scorer()` | zte_domain/ai_cs/ai_cs.py |
| application/root_cause_diagnosis.py | Knowledge_Application/Root_Cause_Diagnosis/input.json + label.json | input `{nodes,edges}`, label `{nodes:[...]}` (single sample) | rendered `json.dumps(input, ensure_ascii=False)` | `json.dumps(label, ensure_ascii=False)` | `structured_em_scorer(mode="json")` | zte_domain/ai_cs/alarm_nodes.py |
| application/tool_invocation.py | Knowledge_Application/Tool_Invocation/tool_invocation.json | dict `{conversations:[{role,content}...]}` | concat system+first user turn | the reference assistant `\boxed{}` conclusions joined | `structured_em_scorer(mode="exact")` with a `pre` that extracts `\boxed{...}` via the regex in telco_challenge/track_a/config.py ANSWER_PATTERN | lagent / datasets/teval |
| application/solution_generation.py | Knowledge_Application/Solution_Generation/solution_generation.json | list `{question,...}` (gold step sequence field confirmed on inspect) | `question` | gold steps | `structured_em_scorer(mode="exact")` for step EM; ALSO expose a `*_judged` task variant using `judge_panel_scorer()` | zte_domain/ume_exclusion/solution.py |
| comprehension/basic_knowledge.py | Knowledge_Comprehension/Basic Theory/Basic_Knowledge/basic_knowledge.json | dict `{total_sampled,questions:[{question,A,B,C,D,answer,...}]}` | `render_mcq(record)` | `answer` (letters) | `multiselect_f1_scorer()` | utils multiple_select_postprocess |
| comprehension/network_5g.py | Knowledge_Comprehension/Basic Theory/5G_Network/5G_network.json | same MCQ shape | `render_mcq(record)` | `answer` | `multiselect_f1_scorer()` | utils multiple_select_postprocess |
| comprehension/protocols_3gpp.py | Knowledge_Comprehension/Basic Theory/3GPP_Protocols/3GPP_protocols.json | inspect: MCQ if A/B/C/D present else subjective | `render_mcq` or `question` | `answer` | `multiselect_f1_scorer()` if MCQ else `judge_panel_scorer()` | zte_domain/tele_3gpp/ |
| comprehension/wireless_network.py | Knowledge_Comprehension/Product Knowledge/Wireless_Network/fault_maintenance.json + network_optimization.json | subjective QA (confirm) | `question` | answer field | `judge_panel_scorer()` | zte_domain/network_optimize/, ume_inclusion/ |
| comprehension/wired_network.py | Knowledge_Comprehension/Product Knowledge/Wired_Nerwork/wired_network.json | inspect MCQ vs QA | `render_mcq` or `question` | answer field | per shape (see protocols_3gpp) | zte_domain/wired_ops/ |
| comprehension/core_network.py | Knowledge_Comprehension/Product Knowledge/Core_Network/core_network.json | subjective QA (confirm) | `question` | answer field | `judge_panel_scorer()` | zte_domain/ume_inclusion/ |

All `@task` functions are named `telecom_bench_<module_stem>` (e.g. `telecom_bench_entity_extraction`).
Records lacking the gold field get `metadata.skip_score=True` and are excluded from the dataset
(filter in `load_dataset`, log a warning). Multi-file sets (RCA, wireless) merge their files in the loader.
````

- [ ] **Step 2: Commit**

```bash
git add src/evals/telecom_bench/SETS.md
git commit -m "docs(telecom_bench): per-set spec sheet for the build swarm"
```

---

### Task 9: Author and run the build `Workflow`

**Files:**
- Uses: the `Workflow` tool (no file to commit for the orchestration itself; agents commit their own files)

- [ ] **Step 1: Run this Workflow** (paste as the `script` arg to the Workflow tool)

```javascript
export const meta = {
  name: 'telecom-bench-swarm',
  description: 'Build the 11 remaining telecom_bench sets by cloning the reference task',
  phases: [{ title: 'Implement' }, { title: 'Review' }],
}

const REPO = '/Users/emolero/Documents/GitHub/ot/evals'
const SETS = [
  'application/entity_extraction.py',
  'application/event_verification.py',
  'application/root_cause_diagnosis.py',
  'application/tool_invocation.py',
  'application/solution_generation.py',
  'comprehension/basic_knowledge.py',
  'comprehension/network_5g.py',
  'comprehension/protocols_3gpp.py',
  'comprehension/wireless_network.py',
  'comprehension/wired_network.py',
  'comprehension/core_network.py',
]

const READ_FIRST = `Read first, in ${REPO}:
- src/evals/telecom_bench/SETS.md (find YOUR row)
- src/evals/telecom_bench/application/intent_recognition.py (the template to clone)
- src/evals/telecom_bench/scorers/{multiselect_f1,structured_em,judge_panel}.py
- src/evals/telecom_bench/loaders.py, config.py, postprocess.py
- src/tests/telecom_bench/test_intent_recognition.py (the test pattern)
- the vendored upstream ref named in your SETS.md row, under data/telecom_bench/upstream_ref/`

const IMPLEMENT = (mod) => `${READ_FIRST}

Implement the telecom_bench set whose module is "src/evals/telecom_bench/${mod}".
1. Inspect its data file (the path is in your SETS.md row) to confirm the exact record shape and gold field.
2. Create src/evals/telecom_bench/${mod} with record_to_sample(), load_dataset() (filter records
   missing the gold field, set metadata.skip_score and log a warning), and an @task named
   telecom_bench_<module_stem> using solver=generate() and the scorer named in your SETS.md row.
3. Create the matching test src/tests/telecom_bench/test_<module_stem>.py: a golden-record test
   (feeding the gold value scores correct) and a known-wrong test (scores 0). Mock get_model for
   judge-panel sets so tests are offline.
4. Run: cd ${REPO} && uv run pytest src/tests/telecom_bench/test_<module_stem>.py -v  (must pass).
5. Do NOT modify _registry.py or __init__.py.
6. git add only your two files and commit: "feat(telecom_bench): <set> task + tests".
Return a JSON object: {module, task_name, gold_key, scorer, tests_passed: bool, notes}.`

const REVIEW = (mod, impl) => `${READ_FIRST}

Adversarially verify the implementation of "src/evals/telecom_bench/${mod}" for FIDELITY to ZTE.
The implementer reported: ${JSON.stringify(impl)}.
Check: (a) record_to_sample maps the correct gold field; (b) the scorer matches the SETS.md row and
the upstream evaluator's logic; (c) load_dataset filters unlabeled records; (d) the @task name is
telecom_bench_<module_stem>; (e) tests actually assert correct=1.0 and wrong=0.0; (f) re-run
cd ${REPO} && uv run pytest src/tests/telecom_bench/test_<stem>.py -v.
If you find a defect, FIX it, re-run the test, and amend the commit.
Return JSON: {module, fidelity_ok: bool, fixed: bool, remaining_issues: string}.`

const SCHEMA_IMPL = { type: 'object', properties: {
  module: {type:'string'}, task_name:{type:'string'}, gold_key:{type:'string'},
  scorer:{type:'string'}, tests_passed:{type:'boolean'}, notes:{type:'string'} },
  required:['module','task_name','tests_passed'] }
const SCHEMA_REVIEW = { type:'object', properties: {
  module:{type:'string'}, fidelity_ok:{type:'boolean'}, fixed:{type:'boolean'},
  remaining_issues:{type:'string'} }, required:['module','fidelity_ok'] }

const results = await pipeline(
  SETS,
  (mod) => agent(IMPLEMENT(mod), {label: `impl:${mod}`, phase: 'Implement', schema: SCHEMA_IMPL}),
  (impl, mod) => agent(REVIEW(mod, impl), {label: `review:${mod}`, phase: 'Review', schema: SCHEMA_REVIEW}),
)

const failures = results.filter(Boolean).filter(r => r && r.fidelity_ok === false)
log(`built ${results.filter(Boolean).length}/${SETS.length} sets; ${failures.length} with open issues`)
return { results: results.filter(Boolean), failures }
```

- [ ] **Step 2: Review the Workflow result.** Read the returned `failures`; for any set with `fidelity_ok=false` or open issues, fix manually following the same TDD loop (inspect data, fix loader/scorer, run its test, commit). Confirm 11 new task modules + 11 new test files exist:

Run: `ls src/evals/telecom_bench/application src/evals/telecom_bench/comprehension && ls src/tests/telecom_bench`
Expected: 6 application modules, 6 comprehension modules, and a test per set.

---

## Phase 2: register, run, report (solo)

### Task 10: Register all 12 tasks

**Files:**
- Modify: `src/evals/_registry.py`, `src/evals/__init__.py`

- [ ] **Step 1: Add all 12 imports + `__all__` entries to `_registry.py`**

```python
# in src/evals/_registry.py, add:
from evals.telecom_bench.application.entity_extraction import telecom_bench_entity_extraction
from evals.telecom_bench.application.event_verification import telecom_bench_event_verification
from evals.telecom_bench.application.intent_recognition import telecom_bench_intent_recognition
from evals.telecom_bench.application.root_cause_diagnosis import telecom_bench_root_cause_diagnosis
from evals.telecom_bench.application.solution_generation import telecom_bench_solution_generation
from evals.telecom_bench.application.tool_invocation import telecom_bench_tool_invocation
from evals.telecom_bench.comprehension.basic_knowledge import telecom_bench_basic_knowledge
from evals.telecom_bench.comprehension.core_network import telecom_bench_core_network
from evals.telecom_bench.comprehension.network_5g import telecom_bench_network_5g
from evals.telecom_bench.comprehension.protocols_3gpp import telecom_bench_protocols_3gpp
from evals.telecom_bench.comprehension.wired_network import telecom_bench_wired_network
from evals.telecom_bench.comprehension.wireless_network import telecom_bench_wireless_network
```
Add all 12 names to `_registry.py`'s `__all__`.

- [ ] **Step 2: Verify discovery**

Run: `uv run inspect list tasks 2>/dev/null | grep telecom_bench`
Expected: 12 `telecom_bench_*` tasks listed.

- [ ] **Step 3: Run the full test suite + lint**

Run: `uv run pytest src/tests/telecom_bench -v && uv run ruff check src/evals/telecom_bench && uv run mypy src/evals/telecom_bench`
Expected: all tests pass; ruff clean; mypy clean (fix any reported issues before continuing).

- [ ] **Step 4: Commit**

```bash
git add src/evals/_registry.py src/evals/__init__.py
git commit -m "feat(telecom_bench): register all 12 evaluation sets"
```

---

### Task 11: Smoke-run across two models + report

**Files:**
- Create: `src/evals/telecom_bench/evaluation/REPORT.md`
- Create: `src/evals/telecom_bench/evaluation/IMPLEMENTATION.md`

- [ ] **Step 1: Run every task on the released examples across two models**

```bash
cd /Users/emolero/Documents/GitHub/ot/evals
for t in $(uv run inspect list tasks 2>/dev/null | grep -o 'telecom_bench_[a-z_0-9]*' | sort -u); do
  uv run inspect eval "evals/$t" --model openai/gpt-4o-mini --limit 10
done
# repeat with a second model, e.g. --model anthropic/claude-3-5-sonnet-latest
```
Expected: each run completes and writes a log under `logs/`. Judge-panel tasks require a judge model available; pass `-T judges='["openai/gpt-4o-mini"]'` or `-T single=true` for cheap runs.

- [ ] **Step 2: Write `REPORT.md`** summarizing, per set: example-subset n, the metric(s), the two models' scores, and a column noting the paper's reported number for context with the caveat "example subset, not paper-scale; numbers not comparable." Mirror the structure of `src/evals/telco_challenge/evaluation/REPORT.md`.

- [ ] **Step 3: Write `IMPLEMENTATION.md`** documenting each fidelity decision: postprocessors ported verbatim, intent `Output:`-split, JSON `are_json_equal` order-insensitivity, judge-panel normalization `(mean-1)/4`, and every place we diverged from upstream (and why).

- [ ] **Step 4: Commit**

```bash
git add src/evals/telecom_bench/evaluation
git commit -m "docs(telecom_bench): evaluation report + implementation notes"
```

- [ ] **Step 5: Update repo docs**

Add `telecom_bench` to `docs/eval-list.md` (one row per the repo's existing format). Commit:

```bash
git add docs/eval-list.md
git commit -m "docs: list telecom_bench in eval-list"
```

---

## Self-review (completed by plan author)

- **Spec coverage:** all 12 sets have a module + test (Task 7 + Task 8/9 spec rows + Task 10 registration); the three scorers (Tasks 3-5) cover Macro-F1, structured EM, and the tri-expert judge; postprocessors ported verbatim (Task 1); no-sandbox honored (every task uses `solver=generate()`); missing-label handling is in `load_dataset` filters (Task 7 + SETS.md); report + implementation log (Task 11) satisfy the deliverables.
- **Placeholder scan:** `GOLD_KEY` and per-set gold fields are explicitly confirmed by an inspection step before coding, not left vague; the Workflow script and SETS.md table are concrete content, not "implement appropriately."
- **Type consistency:** scorer factory names (`multiselect_f1_scorer`, `structured_em_scorer`, `judge_panel_scorer`), helper names (`options_of`, `f1`, `judge_correct`, `parse_likert`, `aggregate`, `render_mcq`, `load_json`, `INTENT_PRE`), and task naming (`telecom_bench_<stem>`) are used consistently across tasks.
- **Known risk:** `tool_invocation` and `solution_generation` have the loosest upstream scoring; their reviewer stage in Task 9 must confirm the `\boxed{}` extraction and step-sequence EM against the vendored `lagent`/`ume_exclusion` references; flagged in SETS.md.
