---
id: benchmarks
title: Benchmarks
sidebar_label: Benchmarks
sidebar_position: 4
---

import BenchmarkCard from '@site/tabs/research/components/BenchmarkCard';

# Benchmarks

Open Telco provides a suite of benchmarks testing different AI capabilities in telecommunications.

## Quick Reference

| Benchmark | Category | Difficulty | Best For |
|-----------|----------|------------|----------|
| TeleQnA | Knowledge | Easy | First evaluation, baseline testing |
| TeleMath | Math Reasoning | Hard | Mathematical/analytical tasks |
| TeleLogs | Operations | Medium | Network diagnostics use cases |
| 3GPP TSG | Standards | Medium | Standards document work |
| TeleYAML | Configuration | Hard | Network automation (coming soon) |

## All Benchmarks

<div className="benchmark-grid">

<BenchmarkCard
  title="TeleQnA"
  description="A benchmark dataset of 10,000 question-answer pairs sourced from telecommunications standards and research articles. Evaluates LLMs' knowledge across general telecom inquiries and complex standards-related questions."
  sampleCount="10K samples"
  difficulty="easy"
  paperLink="https://arxiv.org/abs/2310.15051"
  datasetLink="https://huggingface.co/datasets/netop/TeleQnA"
/>

<BenchmarkCard
  title="TeleMath"
  description="500 mathematically intensive problems covering signal processing, network optimization, and performance analysis. Implemented as a ReAct agent using bash and python tools to solve domain-specific mathematical computations."
  sampleCount="500 samples"
  difficulty="hard"
  paperLink="https://arxiv.org/abs/2506.10674"
  datasetLink="https://huggingface.co/datasets/netop/TeleMath"
/>

<BenchmarkCard
  title="TeleLogs"
  description="A synthetic dataset for root cause analysis (RCA) in 5G networks. Given network configuration parameters and user-plane data (throughput, RSRP, SINR), models must identify which of 8 predefined root causes explain throughput degradation."
  sampleCount="Synthetic"
  difficulty="medium"
  paperLink="https://arxiv.org/abs/2507.21974"
  datasetLink="https://huggingface.co/datasets/netop/TeleLogs"
/>

<BenchmarkCard
  title="3GPP TSG"
  description="Classifies 3GPP technical documents according to their working group. Models must identify the correct technical specification group for a given document text."
  sampleCount="Variable"
  difficulty="medium"
  datasetLink="https://huggingface.co/datasets/eaguaida/gsma_sample"
/>

<BenchmarkCard
  title="TeleYAML"
  description="Evaluates the capability of LLMs to generate standard-compliant YAML configurations for 5G core network tasks: AMF Configuration, Network Slicing, and UE Provisioning."
  sampleCount="In Progress"
  difficulty="hard"
  datasetLink="https://huggingface.co/datasets/otellm/gsma-sample-data"
/>

</div>

## Running Benchmarks

### Basic Usage

```bash
uv run inspect eval src/open_telco/<benchmark>/<benchmark>.py --model <model>
```

### Examples

```bash
# TeleQnA - Knowledge evaluation
uv run inspect eval src/open_telco/teleqna/teleqna.py --model openai/gpt-4o

# TeleMath - Mathematical reasoning
uv run inspect eval src/open_telco/telemath/telemath.py --model openai/gpt-4o

# TeleLogs - Root cause analysis
uv run inspect eval src/open_telco/telelogs/telelogs.py --model openai/gpt-4o

# 3GPP TSG - Standards classification
uv run inspect eval src/open_telco/three_gpp/three_gpp.py --model openai/gpt-4o
```

### Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--limit N` | Run only N samples | `--limit 50` |
| `--epochs N` | Run N times for consistency | `--epochs 3` |
| `--log-dir` | Custom log directory | `--log-dir logs/my-run` |

## Benchmark Details

### TeleQnA

**What it tests:** General telecommunications knowledge from standards and research

**Question types:**
- Multiple choice questions
- True/false questions
- Fill-in-the-blank questions

**Topics covered:**
- 3GPP specifications
- Network protocols
- Wireless technologies
- Telecom standards

**Recommended starting point** for new users.

---

### TeleMath

**What it tests:** Mathematical reasoning in telecom contexts

**Problem types:**
- Signal processing calculations
- Network optimization problems
- Performance analysis
- Resource allocation

**Implementation:** Uses a ReAct agent with bash and Python tools

**Note:** This is the most challenging benchmark, expect lower scores.

---

### TeleLogs

**What it tests:** Root cause analysis capabilities

**Scenario:** Given 5G network data (throughput, RSRP, SINR) and configuration parameters, identify the root cause of throughput degradation.

**Root causes:**
1. High interference
2. Poor coverage
3. Capacity issues
4. Handover problems
5. Backhaul constraints
6. Configuration errors
7. Hardware faults
8. External interference

---

### 3GPP TSG

**What it tests:** Understanding of 3GPP organizational structure

**Task:** Classify technical documents by their originating working group

**Groups:** RAN, SA, CT (Core and Terminals)

---

### TeleYAML (Coming Soon)

**What it tests:** Configuration generation capabilities

**Tasks:**
- AMF Configuration
- Network Slicing setup
- UE Provisioning

**Status:** Currently being revamped

## Interpreting Results

### Accuracy Scores

| Benchmark | Typical Range | Notes |
|-----------|---------------|-------|
| TeleQnA | 40-70% | Multiple choice, tests breadth |
| TeleLogs | 30-60% | Requires reasoning about network state |
| TeleMath | 20-50% | Complex multi-step calculations |
| 3GPP TSG | 50-80% | Document classification |

### Comparing Models

For fair comparisons:
- Use the same number of samples (`--limit`)
- Run multiple epochs (`--epochs 3`)
- Use the same benchmark version

## Next Steps

- [Quickstart](/docs/quickstart) - Run your first evaluation
- [FAQ](/docs/faq) - Common questions answered
- [Research Dashboard](/research/dashboard) - View model rankings
