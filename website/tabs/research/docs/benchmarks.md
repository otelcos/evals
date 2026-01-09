---
id: benchmarks
title: Benchmarks
sidebar_label: Benchmarks
hide_table_of_contents: true
---

import BenchmarkCard from '@site/tabs/research/components/BenchmarkCard';

<div className="research-tabs">
  <a href="/open_telco/research/dashboard" className="research-tab">Dashboard</a>
  <a href="/open_telco/research/benchmarks" className="research-tab active">Benchmarks</a>
  <a href="/open_telco/research/models" className="research-tab">Models</a>
</div>

# Benchmarks

Explore evaluations across 5 distinct benchmarks, covering telecommunications knowledge, network operations, mathematical reasoning, and standards comprehension.

<div className="benchmark-grid">

<BenchmarkCard
  title="TeleQnA"
  description="10,000 multiple-choice questions testing LLM telecom knowledge across 5 categories: Lexicon, Research Overview, Research Publications, Standards Overview, and Standards Specifications."
  paperLink="https://arxiv.org/abs/2310.15051"
  datasetLink="https://huggingface.co/datasets/netop/TeleQnA"
/>

<BenchmarkCard
  title="TeleLogs"
  description="A synthetic dataset for root cause analysis (RCA) in 5G networks. Given network configuration parameters and user-plane data, models must identify which of 8 predefined root causes explain throughput degradation."
  paperLink="https://arxiv.org/abs/2507.21974"
  datasetLink="https://huggingface.co/datasets/netop/TeleLogs"
/>

<BenchmarkCard
  title="TeleMath"
  description="500 mathematically intensive problems covering signal processing, network optimization, and performance analysis. Models must solve domain-specific mathematical computations step-by-step."
  paperLink="https://arxiv.org/abs/2506.10674"
  datasetLink="https://huggingface.co/datasets/netop/TeleMath"
/>

<BenchmarkCard
  title="3GPP-TSG"
  description="Classifies 3GPP technical documents by working group. Models must identify the correct TSG group (e.g., RAN1, SA2, CT4) for a given technical text from specifications."
  datasetLink="https://huggingface.co/datasets/eaguaida/gsma_sample"
/>

<BenchmarkCard
  title="TeleYAML"
  description="Evaluates LLM capability to generate valid YAML configurations for telecommunications network elements. Tests understanding of 5G core network parameters and configuration syntax."
/>

</div>

## Running Benchmarks

```bash
# Run all benchmarks
uv run inspect eval open_telco/teleqna open_telco/telemath \
  --model your/model \
  --log-dir ./results

# Run a specific benchmark
uv run inspect eval src/open_telco/teleqna/teleqna.py --model openai/gpt-4o

# View results
uv run inspect view --log-dir ./results
```

## Benchmark Details

### TeleQnA Categories

| Category | Questions | Description |
|----------|-----------|-------------|
| Lexicon | 500 | General telecom terminology |
| Research Overview | 2,000 | Broad telecom research topics |
| Research Publications | 4,500 | Detailed multi-disciplinary research |
| Standards Overview | 1,000 | 3GPP/IEEE standards summaries |
| Standards Specifications | 2,000 | Technical specs and implementations |

### Scoring Methods

- **TeleQnA**: Multiple choice accuracy with exact match
- **TeleLogs**: Pattern extraction with soft/hard evaluation modes
- **TeleMath**: Numerical answer extraction from `\boxed{...}` format
- **3GPP-TSG**: Pattern matching for working group codes
- **TeleYAML**: LLM-as-judge evaluation for configuration validity
