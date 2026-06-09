# telecom_bench: implementation and fidelity notes

This benchmark is a native Inspect AI port of ZTE TeleCom-Bench (arXiv 2605.18025). The goal is
faithful reproduction of ZTE's scoring methodology over the released example data, with no agent and
no sandbox: every task is `solver=generate()` plus an offline scorer.

ZTE's evaluators and postprocessors are vendored under `data/telecom_bench/upstream_ref/` and were
the reference for every decision below. Where the released opencompass *code* and the *paper* differ,
the choice taken is stated explicitly.

## Shared components

### postprocess.py (ported verbatim from upstream)
- `multiple_select_postprocess`, `extract_non_reasoning_content`, `are_json_equal`, `normalize_zh`
  are line-faithful ports of `utils/text_postprocessors.py` and `zte_domain/ai_cs/alarm_nodes.py`.
- `str2json` is a near-literal port of upstream `utils/clean_jsonstr.py:clean_str_to_json`. The first
  draft was a simplified bracket scanner; it was replaced with the full upstream implementation,
  which has (a) a string-state machine in `_extract_json_candidates` so `{`/`[` inside JSON string
  values are ignored, and (b) a `_strip_wrappers` fallback that strips `<think>...</think>` blocks
  (DOTALL, case-insensitive) and ```json fences before retrying `json.loads`/`ast.literal_eval`.
  This matters because every JSON-scored set depends on it.

### Scorers
- `multiselect_f1` (MCQ): primary value is **exact-set accuracy** (`CORRECT` iff the predicted option
  set equals the gold set), which is faithful to upstream's exact-match MCQ scoring. Macro-F1 is added
  as a **supplementary** partial-credit metric (an enhancement, not present upstream).
- `structured_em`: `mode="json"` applies `str2json` to both prediction and gold then `are_json_equal`
  (order-insensitive for lists of dicts); `mode="exact"` compares `normalize_zh`-normalized strings.
  An optional `pre` callable runs ZTE-style preprocessing before comparison.
- `judge_panel`: a configurable tri-expert 5-point Likert panel. This follows the **paper's** described
  methodology, not the released code. The code uses several different per-subdomain judge scales
  (0-3 for AI customer service, 0-5 and 0-10 for CCN, binary for 3GPP-subjective, a 4-dimension rubric
  for UME). Standardizing on a single configurable 5-point panel is the design's deliberate choice.
  The headline `Score` is `(mean_likert - 1) / 4` (min-max of a 1-5 scale to [0,1]); raw `likert_mean`
  and inter-judge `spread` are reported as metrics. `DEFAULT_JUDGES = [None, None, None]` means three
  calls to the active model; pass `-T judges='[...]'` or `-T single=true` to override.

## Per-set decisions

| set | scorer | gold field | notes / fidelity |
|---|---|---|---|
| intent_recognition | structured_em(exact, INTENT_PRE) | `output` | Upstream applies `str2json` to both sides then `==`, but the gold labels are bare class strings (DONE/UNDONE/ORDER/NO) for which `str2json` returns `None` (it would score 0% on this data). Normalized exact-string match is the faithful equivalent; `INTENT_PRE` replicates upstream's `Output:`/`Thought:` split verbatim. The plan assumed gold field `answer`; the real field is `output`. |
| entity_extraction | structured_em(json) | `answer` | `answer` is a JSON string (e.g. `{"机房名称":...}`); `str2json` + `are_json_equal`. **Conservative divergence:** upstream `IDA/parameter_extract.py:_check_dict` iterates only the gold's keys, so a prediction with the correct values plus *extra* keys passes upstream but fails `are_json_equal` (which requires equal key sets). This can only under-credit, never over-credit, and makes no difference on the example data (the 10 example golds have clean key sets and no `time` field, so upstream's `format_time_to_custom` path is also irrelevant). We keep full-key equality rather than fork the scorer, because the shared `are_json_equal` must stay strict for root_cause_diagnosis (its own evaluator). |
| event_verification | judge_panel | `best_answer` (1-item list of a structured dict) | Open-ended graph-analysis task; single sample. Target is `json.dumps(best_answer)`. Upstream `ai_cs/ai_cs.py` is an LLM judge. |
| root_cause_diagnosis | structured_em(json) | `label.json` | Merges `input.json` (graph) + `label.json`; single sample. Most faithful set: `are_json_equal` is the direct port of this set's own `alarm_nodes.are_json_equal`. |
| tool_invocation | structured_em(exact, boxed pre) | 3 `\boxed{}` conclusions joined | Loosest upstream scoring. `conversations` has three assistant `\boxed{...}` conclusions equal to the `extra_info` values; target joins them. The `pre` extracts `\boxed{...}` from the model output. Single sample. |
| solution_generation | structured_em(exact, tool-step pre) + `_judged` (judge_panel) | `best_answer` | Gold field is `best_answer`, not `answer`. The plain task reproduces upstream `UMESolutionEvaluator`'s binary `tool_step_accuracy`: it extracts the bracketed tool steps `re.findall(r"\[(.*?)\]")` from BOTH the model output (via the scorer `pre`) and the gold (the dataset target is the extracted-step sequence) and requires equality — NOT a full-prose match, which would never match the irregular gold strings. The `_judged` variant keeps the full prose gold and uses the judge panel as a proxy for upstream's ROUGE metrics. |
| basic_knowledge | multiselect_f1 | `answer` | Flat A/B/C/D MCQ, `render_mcq`. |
| network_5g | multiselect_f1 | `answer` (T/F mapped) | Mixed single/multi/true-false. True-false records carry `answer` `"T"`/`"F"` with `A=正确`, `B=错误`; the loader maps `T->A`, `F->B` (otherwise the option-letter comparison can never match) and drops `"None"` option values. |
| protocols_3gpp | multiselect_f1 | `answer` | All MCQ (24 multi, 12 single). `answer` is comma-separated letters. Uses the record's pre-rendered `prompt` field as input. |
| wireless_network | multiselect_f1 | `answer` / `correct_answers` | **Both source files are MCQ** (the plan mislabeled this set as subjective QA). Merges `fault_maintenance.json` (options as a list of "A. ..." strings, gold `answer`) and `network_optimization.json` (options as a dict, gold `correct_answers` list); normalized to (input, gold-letters), 66 samples. |
| wired_network | multiselect_f1 | `答案` | MCQ nested under Chinese keys `单项选择题` (single) / `多项选择题` (multiple), each `{问题, 选项, 答案}`; both handled. |
| core_network | judge_panel | `答案` | Subjective QA with Chinese keys `题目`/`答案` (free-text answers). |

## Divergences from upstream, summarized

1. `multiselect_f1` adds Macro-F1 as a supplementary metric; exact-set accuracy remains the faithful primary.
2. `are_json_equal` is order-insensitive for lists of dicts; faithful for root_cause_diagnosis (its own
   evaluator), marginally more permissive than the raw `==` used by entity_extraction's upstream.
3. `structured_em(exact)` normalizes with NFKC (`normalize_zh`); upstream intent/parameter comparisons
   use raw `==`. This is a no-op on ASCII labels and only helps with full-width Chinese variants.
4. intent_recognition uses exact-string match instead of upstream's `str2json`+`==` (which returns
   `None` on bare class labels and would score 0%).
5. judge_panel follows the paper's single 5-point tri-expert panel rather than the code's many
   per-subdomain judge scales.
6. solution_generation (plain) scores on extracted bracketed tool-step sequences (upstream's
   `tool_step_accuracy`), not full-prose exact match; the `_judged` variant proxies upstream ROUGE.
7. entity_extraction requires equal key sets (`are_json_equal`), stricter than upstream's gold-key-only
   check; conservative (can only under-credit) and a no-op on the example data. See the table above.
8. The MCQ scorer (`multiselect_f1`) runs `multiple_select_postprocess` over the whole completion, so
   stray uppercase letters in a model's reasoning (NR, KPI, OFDM, or option labels E-Z) can pollute the
   predicted option set and flip exact-set accuracy. This is faithful to upstream's bare postprocessor;
   protocols_3gpp's `prompt` already instructs "只输出正确答案的选项". For the other unconstrained MCQ
   sets, prefer running with an instruction to emit only option letters, or extracting an answer region
   before scoring, if you see reasoning-driven false negatives.

## Missing-label handling

Every `load_dataset()` filters records lacking the set's gold field and emits a `logger.warning` with
the skipped count, so partially-labeled releases degrade gracefully rather than crashing.
