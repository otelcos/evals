---
id: installation
title: Installation
sidebar_label: Installation
sidebar_position: 2
---

import InfoCard from '@site/tabs/user-guide/components/InfoCard';

# Installation

Complete setup guide for Open Telco with prerequisites and troubleshooting.

## Prerequisites

Before you begin, ensure you have:

<div className="prereq-grid">
  <div className="prereq-card">
    <div className="prereq-icon">🐍</div>
    <div className="prereq-content">
      <h4>Python 3.10-3.13</h4>
      <p>Python 3.14+ not yet supported</p>
    </div>
    <div className="prereq-check">Required</div>
  </div>
  <div className="prereq-card">
    <div className="prereq-icon">📦</div>
    <div className="prereq-content">
      <h4>uv package manager</h4>
      <p>Fast Python dependency management</p>
    </div>
    <div className="prereq-check">Required</div>
  </div>
  <div className="prereq-card">
    <div className="prereq-icon">🤗</div>
    <div className="prereq-content">
      <h4>HuggingFace account</h4>
      <p>For dataset access</p>
    </div>
    <div className="prereq-check">Required</div>
  </div>
  <div className="prereq-card">
    <div className="prereq-icon">🔑</div>
    <div className="prereq-content">
      <h4>Model API key</h4>
      <p>OpenAI, Anthropic, or OpenRouter</p>
    </div>
    <div className="prereq-check">At least one</div>
  </div>
</div>

## Step 1: Install uv

[uv](https://astral.sh/uv) is a fast Python package manager that handles dependencies reliably.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify the installation:

```bash
uv --version
```

## Step 2: Clone and Install

```bash
git clone https://github.com/gsma-research/open_telco.git
cd open_telco
uv sync
```

This installs all dependencies in an isolated environment.

## Step 3: Configure Environment

Create a `.env` file in the project root with your API keys:

```bash
# Required: HuggingFace token for dataset access
HF_TOKEN=your_huggingface_token_here

# Model API keys (add the ones you need)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# Optional: Default model for evaluations
INSPECT_EVAL_MODEL=anthropic/claude-sonnet-4-20250514
```

### Getting API Keys

| Provider | Where to get it |
|----------|-----------------|
| HuggingFace | [Settings > Access Tokens](https://huggingface.co/settings/tokens) |
| OpenAI | [API Keys](https://platform.openai.com/api-keys) |
| Anthropic | [API Keys](https://console.anthropic.com/settings/keys) |
| OpenRouter | [API Keys](https://openrouter.ai/keys) |

<InfoCard variant="tip" title="OpenRouter Tip">
  OpenRouter provides access to many models (GPT-4, Claude, Gemini, Llama, etc.) through a single API key.
</InfoCard>

## Step 4: Verify Installation

Run a quick test with 5 samples:

```bash
uv run inspect eval src/open_telco/teleqna/teleqna.py --model openai/gpt-4o --limit 5
```

If successful, you'll see evaluation progress and results in your terminal.

## Troubleshooting

### HuggingFace Token Issues

**"HF_TOKEN not set" or dataset access errors**

- Ensure your `.env` file is in the project root (not a subdirectory)
- Verify your HuggingFace token has read access
- Try loading the environment: `source .env` or restart your terminal

### Model Not Found

**"Model not found" errors**

- Check that the model name matches the [Inspect AI format](https://inspect.aisi.org.uk/models.html)
- Verify your API key is set for that provider
- Model format: `provider/model-name` (e.g., `openai/gpt-4o`)

### Python Version Issues

**Version compatibility problems**

```bash
# Check your version
python --version

# Pin a specific version with uv
uv python pin 3.12
uv sync
```

### Common Fixes

<InfoCard variant="warning" title="Environment Variables">
  Make sure your `.env` file doesn't have spaces around the `=` sign.

  Correct: `HF_TOKEN=abc123`

  Wrong: `HF_TOKEN = abc123`
</InfoCard>

## Supported Models

Open Telco supports any model available through [Inspect AI](https://inspect.aisi.org.uk/models.html):

| Provider | Example Models |
|----------|---------------|
| OpenAI | `openai/gpt-4o`, `openai/gpt-4-turbo`, `openai/gpt-4o-mini` |
| Anthropic | `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-3-5-haiku-latest` |
| OpenRouter | `openrouter/google/gemini-2.0-flash-001`, `openrouter/meta-llama/llama-3.1-70b-instruct` |
| Local | `ollama/llama3`, `vllm/mistral-7b` |

## Next Steps

- [Quickstart](/docs/quickstart) - Run your first full evaluation
- [Benchmarks](/docs/benchmarks) - Explore available evaluations
- [FAQ](/docs/faq) - Common questions answered
