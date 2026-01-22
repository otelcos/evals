```
████████╗███████╗██╗     ███████╗████████╗ █████╗ ██████╗ ██╗     ███████╗███████╗
╚══██╔══╝██╔════╝██║     ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║     ██╔════╝██╔════╝
   ██║   █████╗  ██║     █████╗     ██║   ███████║██████╔╝██║     █████╗  ███████╗
   ██║   ██╔══╝  ██║     ██╔══╝     ██║   ██╔══██║██╔══██╗██║     ██╔══╝  ╚════██║
   ██║   ███████╗███████╗███████╗   ██║   ██║  ██║██████╔╝███████╗███████╗███████║
   ╚═╝   ╚══════╝╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝╚══════╝
```

Evaluating LLM Interpretation of Tables in 3GPP Specifications.

## Overview

500 multiple-choice questions testing LLM ability to interpret technical tables from 3GPP standards, paired with 371 real tables from 13 key specifications.

| Capability Level | Description |
|------------------|-------------|
| Retrieval | Direct lookup from tables |
| Interpretation | Understanding table structure and content |
| Reasoning | Multi-step analysis and numerical aggregation |

**Topics covered:** Signal processing, channel configurations, power parameters, modulation schemes, and frequency reference channels.

**Difficulty:** ~50% of questions require multi-step reasoning.

## Usage

```bash
uv run inspect eval src/evals/teletables/teletables.py --model openai/gpt-4o
```

Run a specific sample by ID:
```bash
uv run inspect eval src/evals/teletables/teletables.py --model openai/gpt-4o --sample-id t_521_0
```

## Scoring

Each question has one correct answer from 5 choices. The model's response is compared against the target answer using exact match.

## Metrics

- **accuracy**: Fraction of correct answers
- **stderr**: Standard error of the accuracy estimate

## Dataset Structure

Each record contains:

| Field | Description |
|-------|-------------|
| `question` | Natural language question about table content |
| `choices` | List of 5 answer candidates |
| `answer` | Index (0-4) of correct answer |
| `explanation` | Justification for correct answer |
| `difficult` | Whether question requires multi-step reasoning |
| `table_id` | Identifier of source table |
| `document_id` | 3GPP document identifier |

## Links

- [Paper](https://arxiv.org/abs/2601.04202)
- [Dataset](https://huggingface.co/datasets/netop/TeleTables)
