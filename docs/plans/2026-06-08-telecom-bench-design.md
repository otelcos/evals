# Design: `telecom_bench` — faithful Inspect AI port of ZTE TeleCom-Bench

Date: 2026-06-08
Status: approved (design); pending implementation plan
Upstream: ZTE-AICloud/TeleCom-Bench (paper: arXiv 2605.18025, "TeleCom-Bench: How Far Are Large Language Models from Industrial Telecommunication Applications?")

## 1. Goal and success criteria

Recreate ZTE's TeleCom-Bench as a native Inspect AI benchmark inside `gsma-labs/evals`, named `telecom_bench`.

"Done" means:

1. All 12 evaluation sets are implemented as Inspect `@task`s.
2. Each task uses ZTE's exact scoring methodology (Exact Match, Macro-F1, tri-expert LLM-judge), with their postprocessors ported verbatim.
3. The suite runs end-to-end on the released example subsets.
4. The harness is structured so the full datasets drop in unchanged if obtained later.
5. A report compares our example-subset numbers against the paper's methodology across at least two models.

### Hard constraint: the full data is not public

ZTE released only tiny example subsets to prevent evaluation leakage. The cloned repo ships, per application set: Intent Recognition 10, Entity Extraction 10, Tool Invocation 2, Event Verification 1, Root Cause Diagnosis 2, Solution Generation 5; comprehension sets ship `{total_sampled, questions:[...]}` stubs (for example Basic Knowledge n=23). The paper's full counts are 22,678 samples across 12 sets. Reproducing the paper's reported numbers is therefore out of scope; reproducing the task definitions, harness, and scoring methodology is the goal.

### Other constraints

- All content is zh-CN (Mandarin) telecom-operations text; prompts, rubrics, tool libraries, and labels stay in Chinese (faithful).
- No Docker or Inspect sandbox. TeleCom-Bench as released scores everything offline; even the two "execution" tasks (Tool Invocation, Solution Generation) compare a generated tool-call sequence to a reference sequence by Exact Match, with nothing executed in a container. The paper's "live network / KPI validation" is how ZTE collected ground truth, not something a runner reproduces.

## 2. Package layout

Follows the repo convention (each benchmark is a package under `src/evals/<name>/`; `@task`s are imported in `src/evals/_registry.py` and exposed via the `inspect_ai` entry point).

```
src/evals/telecom_bench/
  __init__.py
  _types.py                 # shared TypedDicts/dataclasses for records
  loaders.py                # JSON -> Inspect Sample adapters (per format)
  postprocess.py            # ports of multiple_select_postprocess, str2json, boxed extraction, zh normalization
  scorers/
    __init__.py
    multiselect_f1.py        # Macro-F1 + accuracy for MCQ
    structured_em.py         # JSON/structured Exact Match (intent, entity, event, RCA, tool, solution)
    judge_panel.py           # tri-expert configurable LLM-judge, 5-pt Likert + inter-rater agreement
  comprehension/            # 6 tasks
    __init__.py
    basic_knowledge.py
    network_5g.py
    protocols_3gpp.py
    wireless_network.py
    wired_network.py
    core_network.py
  application/              # 6 tasks
    __init__.py
    intent_recognition.py    # reference template, built first
    entity_extraction.py
    tool_invocation.py
    event_verification.py
    root_cause_diagnosis.py
    solution_generation.py
data/telecom_bench/         # released example JSONs (vendored or HF-pulled)
src/tests/telecom_bench/    # one test module per set + shared scorer tests
```

All 12 `@task`s are registered in `src/evals/_registry.py` and `src/evals/__init__.py`, named `telecom_bench_<set>` (for example `telecom_bench_intent_recognition`). They remain 12 independent tasks; aggregation into the paper's Comprehension/Application tables happens in the report, not via a combined task.

## 3. The 12 sets and their faithful metrics

| Level | Set | Released n | Output | Scorer |
|---|---|---|---|---|
| Comprehension | Basic Knowledge | 23 | MCQ (1-4 correct) | `multiselect_f1` (Macro-F1 + accuracy) |
| Comprehension | 5G Network | sample | MCQ | `multiselect_f1` |
| Comprehension | 3GPP Protocols | sample | MCQ and subjective | `multiselect_f1` / `judge_panel` |
| Comprehension | Wireless Network (fault + optimization) | sample | subjective QA | `judge_panel` |
| Comprehension | Wired Network | sample | MCQ and QA | `multiselect_f1` / `judge_panel` |
| Comprehension | Core Network | sample | subjective QA | `judge_panel` |
| Application | Intent Recognition | 10 | class in {DONE, UNDONE, ORDER, NO} | `structured_em` (accuracy) |
| Application | Entity Extraction | 10 | JSON entities | `structured_em` (JSON-normalized EM) |
| Application | Tool Invocation | 2 | multi-step tool-call sequence + `\boxed{}` conclusions | `structured_em` (sequence + boxed match) |
| Application | Event Verification | 1 | boolean + justification | `structured_em` / `judge_panel` |
| Application | Root Cause Diagnosis | 2 | root-cause node(s) from a fault graph | `structured_em` (alarm-node match) |
| Application | Solution Generation | 5 | `[tool]&arg&` step sequence | `structured_em` (steps) + `judge_panel` |

The MCQ-vs-judge split for the comprehension Product-Knowledge and 3GPP/Wired sets is decided per file by the implementing agent, by inspecting record structure (presence of `A`/`B`/`C`/`D` + `answer` indicates MCQ; free-text `answer`/`best_answer` indicates subjective).

Mapping to ZTE's own evaluator code (used as the fidelity reference per set):

- Intent Recognition: `datasets/zte_domain/IDA/intent_recognize.py`
- Entity Extraction: `datasets/zte_domain/IDA/parameter_extract.py`
- Tool Invocation: `lagent` / `datasets/teval` agent-trajectory evaluators
- Event Verification, Root Cause Diagnosis: `datasets/zte_domain/ai_cs/ai_cs.py`, `ai_cs/alarm_nodes.py`
- Solution Generation: `datasets/zte_domain/ume_exclusion/solution.py`
- 3GPP / subjective comprehension: `datasets/zte_domain/tele_3gpp/tele_3gpp_subjective.py` (`BaseJudgeACCEvaluator`)
- MCQ comprehension: `multiple_select_postprocess` + accuracy pattern shared across loaders

## 4. Data flow (uniform across tasks)

1. `loaders.py` reads the set's JSON and emits `Sample(input=rendered prompt, target=normalized gold, metadata={set, raw, skip_score})`.
2. The task uses a plain `generate()` solver; no agent and no sandbox (faithful static scoring).
3. The scorer applies the set's postprocessor, then its metric.
4. Per-set aggregate metrics surface in the Inspect log; the report rolls them into the paper's two-level tables.

## 5. Shared scorers

- `multiselect_f1`: normalizes the model's selected options with the ported `multiple_select_postprocess`, computes Macro-F1 and exact accuracy against the gold option set.
- `structured_em`: applies the per-task normalizer (JSON canonicalization via ported `str2json`, boxed-content extraction, alarm-node extraction, or class-label match) then Exact Match.
- `judge_panel`: configurable tri-expert panel. Three judges score on the paper's 5-point Likert rubric; reports the mean score plus inter-rater agreement. Judge model(s) configurable via a `grader_model`-style param; a single-judge flag is available for cheap dev runs. Mirrors `BaseJudgeACCEvaluator`'s rubric and prompt structure, preserving zh-CN content.

## 6. Fidelity guarantees

ZTE's postprocessors are ported verbatim into `postprocess.py` (`multiple_select_postprocess`, `str2json`, `extract_non_reasoning_content`, boxed-content extraction) so Exact Match normalization matches theirs. The judge panel reproduces the 5-point rubric and prompt structure. Every divergence from the original is logged in `IMPLEMENTATION.md`.

## 7. Error handling and edge cases

- Missing answer keys: some released sets ship inputs without labels. Those samples get `metadata.skip_score=True` and are excluded from the metric with a logged warning, rather than scored against nothing.
- Tiny subsets (n=1-2): the run completes, and the report flags "example subset, not paper-scale."
- Chinese normalization: NFKC plus full-width/half-width folding before Exact Match.
- Quirky JSON shapes (for example Tool Invocation stored as `{"conversations": [...]}`, RCA split across `input.json` and `label.json`): handled in per-set loader adapters, isolated from the scorers.

## 8. Testing

Per-set module `src/tests/telecom_bench/test_<set>.py`, mirroring `src/tests/ttac_ipnet/`:

- Golden-record test: feeding the gold answer as the model output scores perfect.
- Known-wrong test: an incorrect output scores zero.

Shared scorer unit tests cover Macro-F1, JSON Exact Match normalization, and judge-panel aggregation (mocked judges).

## 9. Build orchestration (scaffold-once, then swarm)

- Phase 0 (solo, inline): build `_types`, `loaders`, `postprocess`, the three scorers, registry wiring, and `intent_recognition.py` end-to-end as the canonical template, with its tests green.
- Phase 1 (swarm `Workflow`): fan out 11 agents, one per remaining set. Each agent inspects its released JSON and the matching `zte_domain` evaluator, clones the template, implements its loader adapter + `@task` + scorer wiring + tests, then a per-task adversarial review stage verifies fidelity against ZTE's scoring. Pipelined, so each set verifies as soon as it is built.
- Phase 2 (verify, solo): register all tasks, run the suite on the released examples across two models, generate the report.

## 10. Deliverables

- 12 registered `@task`s under `src/evals/telecom_bench/`.
- Shared scorer and postprocessor library.
- Per-set tests under `src/tests/telecom_bench/`.
- Released example data under `data/telecom_bench/`.
- `evaluation/REPORT.md` comparing example-subset numbers to the paper's methodology.
- `evaluation/IMPLEMENTATION.md` documenting each fidelity decision.
