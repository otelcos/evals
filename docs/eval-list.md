# List of Evaluations

This page lists all available benchmarks in Open Telco. Each benchmark tests different aspects of AI capabilities in telecommunications.

## Quick Reference

| Benchmark | Category | Difficulty | Best For |
|-----------|----------|------------|----------|
| [TeleQnA](#teleqna) | Knowledge | Easy | First evaluation, baseline testing |
| [TeleTables](#teletables) | Knowledge | Medium | Table interpretation tasks |
| [TeleMath](#telemath) | Math Reasoning | Hard | Mathematical/analytical tasks |
| [TeleLogs](#telelogs) | Operations | Medium | Network diagnostics use cases |
| [6G-Bench](#6g-bench) | Standards | Hard | 6G-native telecom reasoning |
| [3GPP TSG](#3gpp-tsg) | Standards | Medium | Standards document work |
| [ORANBench](#oranbench) | Standards | Medium | O-RAN architecture and specs |
| [srsRANBench](#srsranbench) | Code Understanding | Medium | srsRAN codebase comprehension |
| [TeleCom-Bench](#telecom-bench) | Knowledge (zh) | Hard | Chinese telecom application + comprehension |
| TeleYAML | Configuration | Hard | Network automation (coming soon) |

**Recommended starting point**: TeleQnA - fastest to run and provides good baseline metrics.

---

## Knowledge & QA

### TeleQnA

**[TeleQnA](../src/evals/teleqna/)**: Benchmark Dataset to Assess Large Language Models for Telecommunications

A benchmark dataset of 10,000 question-answer pairs sourced from telecommunications standards and research articles. Evaluates LLMs' knowledge across general telecom inquiries and complex standards-related questions.

```bash
uv run inspect eval src/evals/teleqna/teleqna.py --model <model>
```

[Paper](https://arxiv.org/abs/2310.15051) | [Dataset](https://huggingface.co/datasets/netop/TeleQnA)

### TeleTables

**[TeleTables](../src/evals/teletables/)**: Evaluating LLM Interpretation of Tables in 3GPP Specifications

A curated set of 100 of the hardest questions from the TeleTables dataset, testing LLM ability to interpret technical tables from 3GPP standards covering signal processing, channel configurations, power parameters, and modulation schemes.

```bash
uv run inspect eval src/evals/teletables/teletables.py --model <model>
```

[Paper](https://arxiv.org/abs/2601.04202) | [Dataset](https://huggingface.co/datasets/netop/TeleTables)

### TeleCom-Bench

**[TeleCom-Bench](../src/evals/telecom_bench/)**: A Chinese-language telecom domain benchmark from ZTE

A native Inspect AI port of ZTE TeleCom-Bench: 12 evaluation sets (13 tasks) across Knowledge Application (intent recognition, entity extraction, event verification, root cause diagnosis, tool invocation, solution generation) and Knowledge Comprehension (basic knowledge, 5G network, 3GPP protocols, wireless network, wired network, core network). Scored offline with ports of ZTE's own postprocessors and evaluators: exact-set/Macro-F1 for MCQ, JSON exact match for structured outputs, and a configurable tri-expert 5-point Likert judge panel for subjective answers. See `src/evals/telecom_bench/evaluation/` for the report and fidelity notes.

```bash
uv run inspect eval evals/telecom_bench_basic_knowledge --model <model>
```

[Paper](https://arxiv.org/abs/2605.18025) | [Upstream](https://github.com/ZTE-AICloud/TeleCom-Bench)

## Mathematical Reasoning

### TeleMath

**[TeleMath](../src/evals/telemath/)**: Evaluating Mathematical Reasoning in Telecom Domain

500 mathematically intensive problems covering signal processing, network optimization, and performance analysis. Implemented as a ReAct agent using bash and python tools to solve domain-specific mathematical computations.

```bash
uv run inspect eval src/evals/telemath/telemath.py --model <model>
```

[Paper](https://arxiv.org/abs/2506.10674) | [Dataset](https://huggingface.co/datasets/netop/TeleMath)

## Network Operations & Diagnostics

### TeleLogs

**[TeleLogs](../src/evals/telelogs/)**: Root Cause Analysis in 5G Networks

A synthetic dataset for root cause analysis (RCA) in 5G networks. Given network configuration parameters and user-plane data (throughput, RSRP, SINR), models must identify which of 8 predefined root causes explain throughput degradation below 600 Mbps.

```bash
uv run inspect eval src/evals/telelogs/telelogs.py --model <model>
```

[Paper](https://arxiv.org/abs/2507.21974) | [Dataset](https://huggingface.co/datasets/netop/TeleLogs)

## Network Configuration

**TeleYaml: 5G Network Configuration Generation** *(In Progress)*

Evaluates the capability of LLMs to generate standard-compliant YAML configurations for 5G core network tasks: AMF Configuration, Network Slicing, and UE Provisioning. This benchmark is currently being revamped.

[Dataset](https://huggingface.co/datasets/otellm/gsma-sample-data)

## Standardization

### 6G-Bench

**[6G-Bench](../src/evals/sixg_bench/sixg_bench.py)**: AI-native 6G network reasoning benchmark

```bash
uv run inspect eval src/evals/sixg_bench/sixg_bench.py --model <model>

uv run inspect eval src/evals/sixg_bench/sixg_bench.py --model <model> -T task_id=T1
```

[Paper](https://arxiv.org/abs/2602.08675) | [Source](https://github.com/maferrag/6G-Bench)

### 3GPP TSG

**[3GPP TSG](../src/evals/three_gpp/)**: Technical Specification Group Classification

Classifies 3GPP technical documents according to their working group. Models must identify the correct group for a given technical text.

```bash
uv run inspect eval src/evals/three_gpp/three_gpp.py --model <model>
```

[Dataset](https://huggingface.co/datasets/eaguaida/gsma_sample)

### ORANBench

**[ORANBench](../src/evals/oranbench/)**: Assessing LLM Understanding of Open Radio Access Networks

1,500 multiple-choice questions derived from 116 O-RAN Alliance specification documents, stratified across 3 difficulty levels (500 easy, 500 medium, 500 hard). A curated subset of the ORAN-Bench-13K dataset covering O-RAN architecture components (O-RU, O-DU, O-CU, RIC), fronthaul interfaces, E2 node configurations, and management/orchestration.

```bash
uv run inspect eval src/evals/oranbench/oranbench.py --model <model>

# Filter by difficulty
uv run inspect eval src/evals/oranbench/oranbench.py --model <model> -T difficulty=hard
```

[Paper](https://arxiv.org/abs/2407.06245) | [Dataset](https://huggingface.co/datasets/prnshv/ORANBench)

### srsRANBench

**[srsRANBench](../src/evals/srsranbench/)**: Assessing LLM Understanding of the srsRAN 5G Codebase

1,502 multiple-choice questions testing LLM comprehension of the srsRAN Project — an open-source 5G RAN implementation in C++. Questions cover classes, functions, libraries, configurations, and 3GPP specification details as implemented in srsRAN.

```bash
uv run inspect eval src/evals/srsranbench/srsranbench.py --model <model>
```

[Paper](https://arxiv.org/abs/2407.06245) | [Dataset](https://huggingface.co/datasets/prnshv/srsRANBench)
