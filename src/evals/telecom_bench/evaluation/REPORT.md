# TeleCom-Bench: Evaluation Report

## Benchmark Overview

TeleCom-Bench is a Chinese-language telecom domain benchmark from ZTE (arXiv 2605.18025), released as
[ZTE-AICloud/TeleCom-Bench](https://github.com/ZTE-AICloud/TeleCom-Bench). This is a native Inspect AI
port: 12 evaluation sets across two categories, scored offline (no agent, no sandbox) using ports of
ZTE's own postprocessors and evaluators. Fidelity decisions are documented in `IMPLEMENTATION.md`.

- **Knowledge Application** (6 sets): intent recognition, entity extraction, event verification,
  root cause diagnosis, tool invocation, solution generation.
- **Knowledge Comprehension** (6 sets): basic knowledge, 5G network, 3GPP protocols, wireless network,
  wired network, core network.

`solution_generation` is exposed as two tasks (deterministic step EM and an LLM-judged variant), so
the suite registers **13 tasks**.

## Configuration

| Parameter | Value |
|-----------|-------|
| Data | Released example subsets vendored under `data/telecom_bench/datasets/` |
| Solver | `generate()` (single turn, no tools, no sandbox) |
| MCQ scoring | Exact-set accuracy (primary) + Macro-F1 (supplementary) |
| Structured scoring | JSON equality (`str2json` + `are_json_equal`) or NFKC-normalized exact match |
| Subjective scoring | Tri-expert 5-point Likert judge panel; headline value `(mean-1)/4` |
| Judge model(s) | `-T judges='[...]'` or `-T single=true`; default = three calls to the active model |

## Sets and example-subset sizes

| set (task name) | category | n (examples) | scorer | reported metrics |
|---|---|---:|---|---|
| telecom_bench_intent_recognition | Application | 10 | structured_em(exact) | accuracy, stderr |
| telecom_bench_entity_extraction | Application | 10 | structured_em(json) | accuracy, stderr |
| telecom_bench_event_verification | Application | 1 | judge_panel | mean, stderr, mean_likert |
| telecom_bench_root_cause_diagnosis | Application | 1 | structured_em(json) | accuracy, stderr |
| telecom_bench_tool_invocation | Application | 1 | structured_em(exact, boxed) | accuracy, stderr |
| telecom_bench_solution_generation | Application | 5 | structured_em(exact) | accuracy, stderr |
| telecom_bench_solution_generation_judged | Application | 5 | judge_panel | mean, stderr, mean_likert |
| telecom_bench_basic_knowledge | Comprehension | 23 | multiselect_f1 | accuracy, macro_f1, stderr |
| telecom_bench_network_5g | Comprehension | 23 | multiselect_f1 | accuracy, macro_f1, stderr |
| telecom_bench_protocols_3gpp | Comprehension | 36 | multiselect_f1 | accuracy, macro_f1, stderr |
| telecom_bench_wireless_network | Comprehension | 66 | multiselect_f1 | accuracy, macro_f1, stderr |
| telecom_bench_wired_network | Comprehension | 30 | multiselect_f1 | accuracy, macro_f1, stderr |
| telecom_bench_core_network | Comprehension | 10 | judge_panel | mean, stderr, mean_likert |

Total: 216 example samples across 13 tasks.

> **Caveat:** these are the upstream *example* subsets, not the paper-scale evaluation sets. Absolute
> numbers here are **not comparable** to the figures reported in the ZTE paper; this report exists to
> validate the port end-to-end and to provide a reproducible harness.

## Verification status (offline, no API)

- All 13 tasks are discoverable via `inspect list tasks`.
- All datasets load with the exact expected sample counts (table above).
- 119 unit tests pass (`uv run pytest src/tests/telecom_bench`), including a golden-record (scores
  correct) and known-wrong (scores 0) test per set. ruff and mypy are clean across the package.
- Every task executes end-to-end under `mockllm/model` (`--limit 1`).

## Results (n = full example subset per set)

Run on 2026-06-09 with `openrouter/deepseek/deepseek-v4-flash` (single turn, `generate()`,
`--max-connections 8`). Judge-panel sets used the default panel (three calls to the same model).
Reproduce with:

```bash
for t in $(uv run inspect list tasks 2>/dev/null | grep -o 'telecom_bench_[a-z0-9_]*' | sort -u); do
  uv run inspect eval "evals/$t" --model openrouter/deepseek/deepseek-v4-flash
done
```

| set | n | scorer | headline | secondary |
|---|---:|---|---:|---|
| intent_recognition | 10 | structured_em(exact) | acc 0.800 | stderr 0.133 |
| entity_extraction | 10 | structured_em(json) | acc 0.000 | stderr 0.000 |
| event_verification | 1 | judge_panel | mean 1.000 | likert 5.00 |
| root_cause_diagnosis | 1 | structured_em(json) | acc 0.000 | stderr 0.000 |
| tool_invocation | 1 | structured_em(exact, boxed) | acc 0.000 | stderr 0.000 |
| solution_generation | 5 | structured_em(exact) | acc 0.400 | stderr 0.245 |
| solution_generation_judged | 5 | judge_panel | mean 0.733 | likert 3.93 |
| basic_knowledge | 23 | multiselect_f1 | acc 0.348 | macro_f1 0.577 |
| network_5g | 23 | multiselect_f1 | acc 0.217 | macro_f1 0.411 |
| protocols_3gpp | 36 | multiselect_f1 | acc 0.556 | macro_f1 0.796 |
| wireless_network | 66 | multiselect_f1 | acc 0.076 | macro_f1 0.283 |
| wired_network | 30 | multiselect_f1 | acc 0.000 | macro_f1 0.324 |
| core_network | 10 | judge_panel | mean 0.642 | likert 3.57 |

Notes:
- Judge-panel headline is the normalized `(mean_likert-1)/4`; the raw 1-5 mean is the `likert` column.
- The strict exact-match sets (entity_extraction, root_cause_diagnosis, tool_invocation, wired_network)
  score 0 exact-set, while the multiselect sets still earn partial macro-F1 (e.g. wired_network
  macro_f1 0.324 with 0 exact-set), confirming the scorer credits partial overlap. Three Application
  sets have n=1, so a single miss yields 0.000; absolute zeros on this small Chinese example subset
  reflect a weak/strict pairing, not scorer bugs (each scorer is covered by golden + known-wrong
  unit tests).
