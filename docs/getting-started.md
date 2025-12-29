# Getting Started

## Prerequisites

**Python 3.10-3.13**

This project requires Python 3.10, 3.11, 3.12, or 3.13. Python 3.14+ is not yet supported.

**uv package manager**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

Clone the repository, pin a supported Python version, and install dependencies:

```bash
git clone https://github.com/gsma/open_telco.git
cd open_telco
uv python pin 3.11  # Pin to a supported version (3.10-3.13)
uv sync
```

If you already have a supported Python version as default, you can skip the `uv python pin` step.

## Dataset Access

Before running evaluations, request access to the benchmark datasets on HuggingFace:

- [TeleQnA](https://huggingface.co/datasets/netop/TeleQnA)
- [TeleMath](https://huggingface.co/datasets/netop/TeleMath)
- [TeleLogs](https://huggingface.co/datasets/netop/TeleLogs)

Configure your HuggingFace token with read access to these repositories.

## Configuration

Create a `.env` file in the root folder:

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

Full list of supported models: https://inspect.aisi.org.uk/models.html
