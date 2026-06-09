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
- 115 unit tests pass (`uv run pytest src/tests/telecom_bench`), including a golden-record (scores
  correct) and known-wrong (scores 0) test per set. ruff and mypy are clean across the package.
- Every task executes end-to-end under `mockllm/model` (`--limit 1`).

## Results (n = full example subset per set)

Pending. The requested model run (`openrouter/deepseek/deepseek-v4-flash`) is blocked: the
`OPENROUTER_API_KEY` in `.env` returns `401 "User not found"` (revoked/expired). Once a valid
OpenRouter key is in place, reproduce with:

```bash
for t in $(uv run inspect list tasks 2>/dev/null | grep -o 'telecom_bench_[a-z0-9_]*' | sort -u); do
  uv run inspect eval "evals/$t" --model openrouter/deepseek/deepseek-v4-flash
done
```

| set | deepseek-v4-flash |
|---|---|
| intent_recognition | _pending_ |
| entity_extraction | _pending_ |
| event_verification | _pending_ |
| root_cause_diagnosis | _pending_ |
| tool_invocation | _pending_ |
| solution_generation | _pending_ |
| solution_generation_judged | _pending_ |
| basic_knowledge | _pending_ |
| network_5g | _pending_ |
| protocols_3gpp | _pending_ |
| wireless_network | _pending_ |
| wired_network | _pending_ |
| core_network | _pending_ |
