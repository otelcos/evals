# Quickstart Guide

Get from zero to your first evaluation result in 5 minutes.

## 1. Install Dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/otelcos/evals.git
cd evals
uv sync
```

## 2. Configure API Keys

Create a `.env` file in the project root:

```bash
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key
```

Need help getting tokens? See [Getting Your API Keys](getting-started.md#getting-your-api-keys).

## 3. Run Your First Evaluation

Start with **TeleQnA** - it's the fastest benchmark and a good test of your setup:

```bash
uv run inspect eval src/evals/teleqna/teleqna.py --model openai/gpt-4o --limit 20
```

This runs 20 questions from the TeleQnA benchmark using GPT-4o.

## 4. View Your Results

```bash
uv run inspect view
```

This opens a browser-based viewer showing:
- Overall accuracy score
- Per-question results
- Model responses and correct answers

## Understanding the Output

During evaluation, you'll see progress like:

```
teleqna: 100%|████████████████████| 20/20 [00:45<00:00]
```

Final results show:

| Metric | Description |
|--------|-------------|
| `accuracy` | Percentage of correct answers |
| `samples` | Number of questions evaluated |
| `time` | Total evaluation duration |

## Try Different Benchmarks

Once TeleQnA works, try others:

```bash
# Network diagnostics (root cause analysis)
uv run inspect eval src/evals/telelogs/telelogs.py --model openai/gpt-4o --limit 10

# Standards classification
uv run inspect eval src/evals/three_gpp/three_gpp.py --model openai/gpt-4o --limit 10
```

## Try Different Models

```bash
# Anthropic Claude
uv run inspect eval src/evals/teleqna/teleqna.py --model anthropic/claude-sonnet-4-20250514 --limit 20

# Via OpenRouter (access many models)
uv run inspect eval src/evals/teleqna/teleqna.py --model openrouter/google/gemini-2.0-flash-001 --limit 20
```

## Next Steps

- [List of Evaluations](eval-list.md) - All available benchmarks explained
- [Running Evaluations](running-evaluations.md) - Run full benchmarks, multiple models, and eval sets
- [FAQ](faq.md) - Common questions and troubleshooting
