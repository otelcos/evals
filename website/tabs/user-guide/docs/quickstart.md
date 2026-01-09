---
id: quickstart
title: Quickstart
sidebar_label: Quickstart
sidebar_position: 3
---

import InfoCard from '@site/tabs/user-guide/components/InfoCard';

# Quickstart

Get from zero to your first evaluation result in 5 minutes.

## 1. Install Dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/gsma-research/open_telco.git
cd open_telco
uv sync
```

## 2. Configure API Keys

Create a `.env` file in the project root:

```bash
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key
```

<InfoCard variant="tip" title="Need API keys?">
  See the [Installation Guide](/docs/installation#getting-api-keys) for links to get your tokens.
</InfoCard>

## 3. Run Your First Evaluation

Start with **TeleQnA** - it's the fastest benchmark and a good test of your setup:

```bash
uv run inspect eval src/open_telco/teleqna/teleqna.py --model openai/gpt-4o --limit 20
```

This runs 20 questions from the TeleQnA benchmark using GPT-4o.

### What You'll See

During evaluation:

```
teleqna: 100%|████████████████████| 20/20 [00:45<00:00]
```

After completion, you'll see a summary with:
- **Accuracy**: Percentage of correct answers
- **Samples**: Number of questions evaluated
- **Time**: Total evaluation duration

## 4. View Your Results

```bash
uv run inspect view
```

This opens a browser-based viewer showing:
- Overall accuracy score
- Per-question results
- Model responses and correct answers

## Try Different Benchmarks

Once TeleQnA works, try others:

```bash
# Network diagnostics (root cause analysis)
uv run inspect eval src/open_telco/telelogs/telelogs.py --model openai/gpt-4o --limit 10

# Standards classification
uv run inspect eval src/open_telco/three_gpp/three_gpp.py --model openai/gpt-4o --limit 10

# Mathematical reasoning (more challenging)
uv run inspect eval src/open_telco/telemath/telemath.py --model openai/gpt-4o --limit 5
```

## Try Different Models

```bash
# Anthropic Claude
uv run inspect eval src/open_telco/teleqna/teleqna.py \
  --model anthropic/claude-sonnet-4-20250514 --limit 20

# Via OpenRouter (access many models)
uv run inspect eval src/open_telco/teleqna/teleqna.py \
  --model openrouter/google/gemini-2.0-flash-001 --limit 20

# GPT-4o-mini (faster, cheaper)
uv run inspect eval src/open_telco/teleqna/teleqna.py \
  --model openai/gpt-4o-mini --limit 20
```

## Run Full Benchmarks

Remove the `--limit` flag to run the complete dataset:

```bash
# Full TeleQnA (10,000 questions)
uv run inspect eval src/open_telco/teleqna/teleqna.py --model openai/gpt-4o
```

<InfoCard variant="warning" title="API Costs">
  Full benchmarks can be expensive. TeleQnA with GPT-4o runs ~10K questions. Start with `--limit` to estimate costs.
</InfoCard>

## Compare Multiple Models

Run the same benchmark across models:

```bash
uv run inspect eval-set src/open_telco/teleqna/teleqna.py \
  --model openai/gpt-4o,anthropic/claude-sonnet-4-20250514 \
  --log-dir logs/comparison \
  --limit 50
```

## Common Issues

### Rate Limiting

If you hit rate limits, add retry options:

```bash
uv run inspect eval src/open_telco/teleqna/teleqna.py \
  --model openai/gpt-4o \
  --retry-attempts 5 \
  --retry-wait 60
```

### Slow Evaluations

- Use `--limit 10` for testing
- Try faster models like `gpt-4o-mini`
- Check your internet connection

### Inconsistent Results

Run multiple epochs for more stable results:

```bash
uv run inspect eval src/open_telco/teleqna/teleqna.py \
  --model openai/gpt-4o \
  --epochs 3
```

## Next Steps

- [Benchmarks](/docs/benchmarks) - Explore all available evaluations
- [FAQ](/docs/faq) - Common questions answered
- [Contributing](/docs/contributing) - Help improve Open Telco
