```
████████╗███████╗██╗     ███████╗ ██████╗ ███╗   ██╗ █████╗
╚══██╔══╝██╔════╝██║     ██╔════╝██╔═══██╗████╗  ██║██╔══██╗
   ██║   █████╗  ██║     █████╗  ██║   ██║██╔██╗ ██║███████║
   ██║   ██╔══╝  ██║     ██╔══╝  ██║▄▄ ██║██║╚██╗██║██╔══██║
   ██║   ███████╗███████╗███████╗╚██████╔╝██║ ╚████║██║  ██║
   ╚═╝   ╚══════╝╚══════╝╚══════╝ ╚══▀▀═╝ ╚═╝  ╚═══╝╚═╝  ╚═╝
```

Evaluating LLM Understanding of Telco Knowledge and Standards.

## Overview

10,000 multiple-choice questions testing LLM telecom knowledge across 5 categories:

| Category | Questions | Description |
|----------|-----------|-------------|
| Lexicon | 500 | General telecom terminology |
| Research Overview | 2,000 | Broad telecom research topics |
| Research Publications | 4,500 | Detailed multi-disciplinary research |
| Standards Overview | 1,000 | 3GPP/IEEE standards summaries |
| Standards Specifications | 2,000 | Technical specs and implementations |

## Usage

```bash
uv run inspect eval src/open_telco/teleqna/teleqna.py --model openai/gpt-4o
```

Filter by category using the `subject` parameter:
```bash
uv run inspect eval src/open_telco/teleqna/teleqna.py --model openai/gpt-4o -T subject=lexicon
```

## Scoring

Each question has one correct answer from multiple choices. The model's response is compared against the target answer using exact match.

**Reference performance:** GPT-4 achieves ~87% on Lexicon but only ~64% on Standards questions, with 74% overall accuracy.

## Metrics

- **accuracy**: Fraction of correct answers
- **stderr**: Standard error of the accuracy estimate

## Links

- [Paper](https://arxiv.org/abs/2310.15051)
- [Dataset](https://huggingface.co/datasets/netop/TeleQnA)
- [GitHub](https://github.com/netop-team/TeleQnA)
