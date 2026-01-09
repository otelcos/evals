---
id: welcome
title: Open Telco
sidebar_label: Welcome
sidebar_position: 1
slug: /
---

import BenchmarkCard from '@site/tabs/research/components/BenchmarkCard';
import QuickStartSteps from '@site/tabs/user-guide/components/QuickStartSteps';

# Open Telco

A suite of telco-specific benchmarks for evaluating AI models on telecommunications tasks.

<div className="welcome-hero">
  <p className="welcome-tagline">
    Built on <a href="https://inspect.aisi.org.uk/">Inspect AI</a>, Open Telco provides standardized evaluations for knowledge, reasoning, and operational capabilities in the telecom domain.
  </p>
</div>

## What You Can Do

<div className="feature-grid">
  <div className="feature-card">
    <div className="feature-icon">📊</div>
    <h3>Benchmark AI Models</h3>
    <p>Evaluate LLMs on telecom-specific tasks including knowledge, math reasoning, and network diagnostics.</p>
  </div>
  <div className="feature-card">
    <div className="feature-icon">🔬</div>
    <h3>Access Research Datasets</h3>
    <p>Use curated datasets from GSMA and partners, available on <a href="https://huggingface.co/datasets/GSMA/open_telco">Hugging Face</a>.</p>
  </div>
  <div className="feature-card">
    <div className="feature-icon">🛠️</div>
    <h3>Build Custom Evals</h3>
    <p>Create new benchmarks using the Inspect AI framework with network simulation support.</p>
  </div>
  <div className="feature-card">
    <div className="feature-icon">🤖</div>
    <h3>Deploy Telco Agents</h3>
    <p>Access specialized agents trained for telecommunications workflows.</p>
  </div>
</div>

## Quick Start

Get running in under 5 minutes:

<QuickStartSteps steps={[
  {
    title: "Clone the repository",
    description: "Get the Open Telco codebase from GitHub.",
    code: "git clone https://github.com/gsma-research/open_telco.git\ncd open_telco"
  },
  {
    title: "Install dependencies",
    description: "Use uv for fast, reliable package management.",
    code: "pip install uv\nuv sync"
  },
  {
    title: "Configure your environment",
    description: "Add your API keys to a .env file.",
    code: "# Create .env with:\nHF_TOKEN=your_huggingface_token\nOPENAI_API_KEY=your_openai_key"
  },
  {
    title: "Run your first evaluation",
    description: "Test with TeleQnA benchmark.",
    code: "uv run inspect eval src/open_telco/teleqna/teleqna.py --model openai/gpt-4o --limit 20"
  }
]} />

For detailed setup instructions, see the [Installation Guide](/docs/installation).

## Available Benchmarks

<div className="benchmark-grid">
  <BenchmarkCard
    title="TeleQnA"
    description="10,000 Q&A pairs testing telecom knowledge from standards and research."
    sampleCount="10K samples"
    difficulty="easy"
    paperLink="https://arxiv.org/abs/2310.15051"
    datasetLink="https://huggingface.co/datasets/netop/TeleQnA"
  />
  <BenchmarkCard
    title="TeleMath"
    description="Mathematical reasoning in signal processing and network optimization."
    sampleCount="500 samples"
    difficulty="hard"
    paperLink="https://arxiv.org/abs/2506.10674"
    datasetLink="https://huggingface.co/datasets/netop/TeleMath"
  />
  <BenchmarkCard
    title="TeleLogs"
    description="Root cause analysis for 5G network throughput issues."
    sampleCount="Synthetic"
    difficulty="medium"
    paperLink="https://arxiv.org/abs/2507.21974"
    datasetLink="https://huggingface.co/datasets/netop/TeleLogs"
  />
</div>

See all benchmarks on the [Benchmarks](/docs/benchmarks) page.

## Next Steps

| Guide | Description |
|-------|-------------|
| [Installation](/docs/installation) | Detailed setup with prerequisites and troubleshooting |
| [Quickstart](/docs/quickstart) | Run your first evaluation step-by-step |
| [Benchmarks](/docs/benchmarks) | Explore all available evaluations |
| [FAQ](/docs/faq) | Common questions answered |
| [Contributing](/docs/contributing) | How to contribute to Open Telco |

## Resources

- **Repository:** [github.com/gsma-research/open_telco](https://github.com/gsma-research/open_telco)
- **Datasets:** [huggingface.co/datasets/GSMA/open_telco](https://huggingface.co/datasets/GSMA/open_telco)
- **Inspect AI:** [inspect.aisi.org.uk](https://inspect.aisi.org.uk/)
