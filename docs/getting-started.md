# Getting Started

Open Telco is a suite of telco-specific benchmarks for evaluating AI models on telecommunications tasks. Built on [Inspect AI](https://inspect.aisi.org.uk/), it provides standardized evaluations for knowledge, reasoning, and operational capabilities in the telecom domain.

## Prerequisites Checklist

Before you begin, ensure you have:

| Requirement | Details |
|-------------|---------|
| Python 3.10-3.13 | Python 3.14+ not yet supported |
| uv package manager | Fast Python package installer |
| HuggingFace account | For dataset access ([sign up](https://huggingface.co/join)) |
| Model API key | At least one: OpenAI, Anthropic, or OpenRouter |

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```bash
git clone https://github.com/gsma-research/open_telco.git
cd open_telco
uv sync
```

## Configuration

Create a `.env` file in the project root:

```bash
# Required: HuggingFace token for dataset access
HF_TOKEN=your_huggingface_token_here

# Model API keys (add the ones you need)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# Optional: Default model
INSPECT_EVAL_MODEL=anthropic/claude-sonnet-4-20250514
```

### Getting Your API Keys

| Provider | Where to get it |
|----------|-----------------|
| HuggingFace | [Settings > Access Tokens](https://huggingface.co/settings/tokens) |
| OpenAI | [API Keys](https://platform.openai.com/api-keys) |
| Anthropic | [API Keys](https://console.anthropic.com/settings/keys) |
| OpenRouter | [API Keys](https://openrouter.ai/keys) |

Full list of supported models: [Inspect AI Models](https://inspect.aisi.org.uk/models.html)

## Verify Your Setup

Run a quick test with 5 samples:

```bash
uv run inspect eval src/open_telco/teleqna/teleqna.py --model openai/gpt-4o --limit 5
```

If successful, you'll see evaluation progress and results in your terminal.

## Troubleshooting

**"HF_TOKEN not set" or dataset access errors**
- Ensure your `.env` file is in the project root (not a subdirectory)
- Verify your HuggingFace token has read access
- Run `source .env` or restart your terminal

**"Model not found" errors**
- Check that the model name matches the [Inspect AI format](https://inspect.aisi.org.uk/models.html)
- Verify your API key is set for that provider

**Python version issues**
- Run `python --version` to check your version
- Use `uv python pin 3.12` to set a specific version

## Next Steps

- [Quickstart Guide](quickstart.md) - Run your first full evaluation
- [List of Evaluations](eval-list.md) - Explore available benchmarks
- [Running Evaluations](running-evaluations.md) - Advanced usage and options
- [FAQ](faq.md) - Common questions answered
