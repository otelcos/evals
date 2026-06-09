# telecom_bench set specifications (verified against the released example data)

Each agent builds ONE set by cloning `application/intent_recognition.py` (the reference template).
Reuse the shared scorers in `scorers/`. **Do NOT edit `_registry.py` or `__init__.py`** (Task 10
handles registration centrally).

The record shapes, gold fields, and sample counts below were **verified by inspecting the actual
data files** (not guessed). Still, re-inspect your file before coding and confirm the gold field
matches what is stated here. Gold field names are NOT uniform across sets (some use `answer`, some
`best_answer`, some `output`, some Chinese keys like `答案`/`correct_answers`) — use the exact one
named for your set.

## Conventions every agent follows

- `@task` function name: `telecom_bench_<module_stem>` (e.g. `application/entity_extraction.py` ->
  `telecom_bench_entity_extraction`). The module file is `src/evals/telecom_bench/<module>`.
- Use `solver=generate()`. No agent, no sandbox.
- `load_dataset()` filters records missing the gold field; if any are skipped, `logger.warning(...)`
  with the count (mirror the reference template's pattern).
- Write `src/tests/telecom_bench/test_<module_stem>.py` mirroring `test_intent_recognition.py`:
  a golden-record test (the gold value scores correct) and a known-wrong test (scores 0). For
  EM-based sets, call the scorer's pure helper directly (`judge_correct(...)` from `structured_em`,
  or `f1(...)`/`options_of(...)` from `multiselect_f1`). For judge-panel sets, test `parse_likert`/
  `aggregate` and mock `inspect_ai.model.get_model` so tests are fully offline.
- Multi-file sets (root_cause_diagnosis, wireless_network) merge their files inside the loader.
- Render helpers: `render_mcq(record)` in `loaders.py` works for records with flat `A/B/C/D` keys.
  Sets with a different option shape (a list, a dict, or a nested key) need a small local renderer
  in the module (see each set below).
- After implementing: `uv run pytest src/tests/telecom_bench/test_<module_stem>.py -v` must pass;
  `uv run ruff check` and `uv run mypy` on your module must be clean. Then `git add` ONLY your two
  files and commit `feat(telecom_bench): <set> task + tests`.

Data root: `data/telecom_bench/datasets/` (referenced below as `<DATA>`). `KA`/`KC` in `config.py`
already point at `<DATA>/Knowledge_Application` and `<DATA>/Knowledge_Comprehension`.

---

## Knowledge Application

### application/entity_extraction.py  →  `telecom_bench_entity_extraction`
- File: `KA/Entity_Extraction/entity_extraction.json` — top-level **list**, n=10.
- Record keys: `id, question, 意图, 构造方式, 测试维度, answer, 机房名称, 专业, type`.
- Input: `record["question"]`.
- Gold: `record["answer"]` — a **JSON string** e.g. `{"机房名称":"郑州金水机房","专业":"网管网"}` (str2json-parseable).
- Scorer: `structured_em_scorer(mode="json")` (str2json + are_json_equal on both sides).
- Upstream ref: `zte_domain/IDA/parameter_extract.py`.

### application/event_verification.py  →  `telecom_bench_event_verification`
- File: `KA/Event_Verification/event_verification.json` — top-level **dict** `{question, best_answer}`. Single sample.
- `best_answer` is a **list with one structured dict** (`source_ishighloadcell`, `highload_time`,
  `target`, `load_unbalance_result`).
- Input: `record["question"]` (a long flow-analysis prompt over graph data).
- Gold/target: `json.dumps(best_answer, ensure_ascii=False)`.
- Scorer: `judge_panel_scorer()` (open-ended analysis task; per design doc + upstream judge).
  Inspect `zte_domain/ai_cs/ai_cs.py`: if upstream does deterministic dict comparison rather than an
  LLM judge, prefer `structured_em_scorer(mode="json")` and say so in your report.
- Wrap the single dict as a 1-item dataset: `load_dataset()` returns one Sample.

### application/root_cause_diagnosis.py  →  `telecom_bench_root_cause_diagnosis`
- Files: `KA/Root_Cause_Diagnosis/input.json` (dict `{nodes:[15], edges}`) + `label.json`
  (dict `{nodes:[2]}`, each node `{label, properties, @rid}`). **Single sample**, merge both files.
- Input: `json.dumps(input_data, ensure_ascii=False)` (the graph).
- Gold/target: `json.dumps(label_data, ensure_ascii=False)`.
- Scorer: `structured_em_scorer(mode="json")` (are_json_equal is the port of this set's own
  `zte_domain/ai_cs/alarm_nodes.are_json_equal`, so this is the most faithful set).
- Upstream ref: `zte_domain/ai_cs/alarm_nodes.py`.

### application/tool_invocation.py  →  `telecom_bench_tool_invocation`  (loosest scoring — flag for review)
- File: `KA/Tool_Invocation/tool_invocation.json` — top-level **dict** `{conversations, extra_info}`. Single sample.
- `conversations` = 7 turns: `system, user, assistant, user, assistant, user, assistant`. The
  three **assistant** turns each contain a `\boxed{...}` conclusion. `extra_info` =
  `{事件核查结果, 一级根因, 二级根因}` whose values equal those three boxed conclusions.
- Input: system turn content + the first user turn content (the task setup).
- Gold/target: the three `\boxed{...}` conclusions joined (e.g. with `"|"`), or equivalently the
  three `extra_info` values joined in the same order.
- Scorer: `structured_em_scorer(mode="exact", pre=<boxed-extractor>)` where the `pre` extracts all
  `\boxed{...}` contents from the model output and joins them. Use the boxed regex pattern from
  `src/evals/telco_challenge/track_a/config.py` (ANSWER_PATTERN) for fidelity.
- Single sample. Upstream ref: lagent / teval boxed-answer convention. **Reviewer must verify the
  boxed extraction and target construction carefully — this set has the loosest upstream scoring.**

### application/solution_generation.py  →  `telecom_bench_solution_generation` (+ `_judged` variant)
- File: `KA/Solution_Generation/solution_generation.json` — top-level **list**, n=5.
- Record keys: `question, best_answer`. Gold field is **`best_answer`** (NOT `answer`).
- `best_answer` = a step sequence string, e.g. `step1.使用[IQ碎片清理]执行&defragmentIQ&命令step2...`.
- Input: `record["question"]`.
- Scorer: `structured_em_scorer(mode="exact")` for step-sequence EM. **Also** expose a second
  `@task` `telecom_bench_solution_generation_judged` using `judge_panel_scorer()` (same dataset).
- Upstream ref: `zte_domain/ume_exclusion/solution.py`.

---

## Knowledge Comprehension

### comprehension/basic_knowledge.py  →  `telecom_bench_basic_knowledge`
- File: `KC/Basic Theory/Basic_Knowledge/basic_knowledge.json` — top-level **dict**
  `{total_sampled, questions:[...]}`, n=23. (Note the space in `Basic Theory`; pathlib handles it.)
- Record: flat `id, question, A, B, C, D, answer, ...(many metadata fields)`. `answer` is a letter
  (single or multi).
- Input: `render_mcq(record)`. Gold: `record["answer"]`.
- Scorer: `multiselect_f1_scorer()`.

### comprehension/network_5g.py  →  `telecom_bench_network_5g`  (mixed types — needs T/F mapping)
- File: `KC/Basic Theory/5G_Network/5G_network.json` — dict `{total_sampled, questions:[...]}`, n=23.
- Record: `id, source_file, question, A, B, C, D, answer`.
- **Mixed question types** (from `source_file`): 单选题 (single, answer a letter), 多选题 (multi,
  answer like `ABCD`), 判断题 (true/false). For 判断题, `A="正确"`, `B="错误"`, `C/D="None"` (string),
  and `answer` is **`"T"` or `"F"`** (T=正确, F=错误).
- **Required handling:** in the loader, normalize true/false gold to the matching option letter:
  `"T" -> "A"`, `"F" -> "B"` (confirm `A=="正确"` and `B=="错误"` for those records; if not, map by
  matching the 正确/错误 text). Without this, `multiselect_f1` compares `{F}` vs the model's
  `{A|B}` and always misses. Skip option values equal to the string `"None"` when rendering.
- Input: `render_mcq(record)` (drop `"None"` options). Gold: normalized `answer`.
- Scorer: `multiselect_f1_scorer()`. Document the T/F normalization in your report.

### comprehension/protocols_3gpp.py  →  `telecom_bench_protocols_3gpp`  (all MCQ)
- File: `KC/Basic Theory/3GPP_Protocols/3GPP_protocols.json` — dict
  `{total_sampled, stratify_by, questions:[...]}`, n=36 (多选题×24, 单选题×12 — **all MCQ**, none subjective).
- Record: `id, 题型, question, answer, A, B, C, D, difficulty, prompt`. `answer` is
  **comma-separated letters** e.g. `"A,B,C,D"` (multiple_select_postprocess ignores commas, so
  `options_of` yields the right set). Every record has a pre-rendered **`prompt`** field.
- Input: `record["prompt"]` (use the ready-made prompt; fall back to `render_mcq` only if absent).
- Gold: `record["answer"]`. Scorer: `multiselect_f1_scorer()`.
- Upstream ref: `zte_domain/tele_3gpp/`.

### comprehension/wireless_network.py  →  `telecom_bench_wireless_network`  (MCQ; merge two differently-shaped files)
- Files (merge both): `KC/Product Knowledge/Wireless_Network/fault_maintenance.json` and
  `.../network_optimization.json`. **Both are MCQ** (the plan mislabeled this set "subjective").
  - `fault_maintenance.json` — top-level **list**, n=33. Record: `id, question, options (LIST of
    "A. ..." strings), answer (letter), explanation, metainfo, type`. Render: `question + "\n" +
    "\n".join(options)`. Gold: `answer`.
  - `network_optimization.json` — top-level **list**, n=33. Record: `id, question_type, type, stem,
    options (DICT {"A":..,"B":..}), correct_answers (LIST e.g. ["B"]), knowledge, capability`.
    Render: `stem + "\n" + "\n".join(f"{k}. {v}" for k,v in options.items())`. Gold:
    `"".join(correct_answers)` (or join — `options_of` only keeps letters).
- The loader normalizes each file's records to `(input_text, gold_letters)` then concatenates (66 samples).
- Scorer: `multiselect_f1_scorer()`.
- Upstream ref: `zte_domain/network_optimize/`, `zte_domain/ume_inclusion/`.

### comprehension/wired_network.py  →  `telecom_bench_wired_network`  (MCQ nested under Chinese keys)
- File: `KC/Product Knowledge/Wired_Nerwork/wired_network.json` (note the upstream typo
  `Wired_Nerwork`) — dict `{total_sampled, questions:[...]}`, n=30.
- Record: `{id, <nested>}` where `<nested>` key is **either `单项选择题` (single, ×12) or
  `多项选择题` (multiple, ×18)**. The nested dict is `{问题, 选项 (LIST of "A. ..." strings), 答案}`.
- Input: `问题 + "\n" + "\n".join(选项)`. Gold: `答案` (a letter or multi-letter string).
- Handle both nested keys in the loader. Scorer: `multiselect_f1_scorer()`.
- Upstream ref: `zte_domain/wired_ops/`.

### comprehension/core_network.py  →  `telecom_bench_core_network`  (subjective QA)
- File: `KC/Product Knowledge/Core_Network/core_network.json` — dict
  `{total_sampled, products, questions:[...]}`, n=10.
- Record (Chinese keys): `难度, 大类, 题目, 答案, product, id`. `题目`=question, `答案`=**free-text**
  answer (e.g. `"SMF网元和AMF网元"`).
- Input: `record["题目"]`. Gold/target: `record["答案"]`.
- Scorer: `judge_panel_scorer()`. Upstream ref: `zte_domain/ume_inclusion/`.

---

## Summary table

| module | task name | samples | input | gold | scorer |
|---|---|---|---|---|---|
| application/entity_extraction.py | telecom_bench_entity_extraction | 10 | question | answer (JSON str) | structured_em(json) |
| application/event_verification.py | telecom_bench_event_verification | 1 | question | json.dumps(best_answer) | judge_panel |
| application/root_cause_diagnosis.py | telecom_bench_root_cause_diagnosis | 1 | json.dumps(input) | json.dumps(label) | structured_em(json) |
| application/tool_invocation.py | telecom_bench_tool_invocation | 1 | system+first user | boxed conclusions joined | structured_em(exact, boxed pre) |
| application/solution_generation.py | telecom_bench_solution_generation(+_judged) | 5 | question | best_answer | structured_em(exact) (+judge_panel) |
| comprehension/basic_knowledge.py | telecom_bench_basic_knowledge | 23 | render_mcq | answer | multiselect_f1 |
| comprehension/network_5g.py | telecom_bench_network_5g | 23 | render_mcq | answer (T→A,F→B) | multiselect_f1 |
| comprehension/protocols_3gpp.py | telecom_bench_protocols_3gpp | 36 | prompt | answer | multiselect_f1 |
| comprehension/wireless_network.py | telecom_bench_wireless_network | 66 | per-file render | answer/correct_answers | multiselect_f1 |
| comprehension/wired_network.py | telecom_bench_wired_network | 30 | 问题+选项 | 答案 | multiselect_f1 |
| comprehension/core_network.py | telecom_bench_core_network | 10 | 题目 | 答案 | judge_panel |
