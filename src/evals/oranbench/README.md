```
 ██████╗ ██████╗  █████╗ ███╗   ██╗██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██╔═══██╗██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██║   ██║██████╔╝███████║██╔██╗ ██║██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██║   ██║██╔══██╗██╔══██║██║╚██╗██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
╚██████╔╝██║  ██║██║  ██║██║ ╚████║██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
 ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
```

Evaluating LLM Understanding of O-RAN Specifications and Architecture.

## Overview

1,500 multiple-choice questions derived from 116 O-RAN Alliance specification documents, stratified across 3 difficulty levels:

| Difficulty | Questions | Description |
|------------|-----------|-------------|
| Easy | 500 | Foundational O-RAN concepts and terminology |
| Medium | 500 | Intermediate architecture and protocol questions |
| Hard | 500 | Advanced specification details and functional splits |

## Usage

```bash
uv run inspect eval src/evals/oranbench/oranbench.py --model openai/gpt-4o
```

Filter by difficulty using the `difficulty` parameter:
```bash
uv run inspect eval src/evals/oranbench/oranbench.py --model openai/gpt-4o -T difficulty=hard
```

## Scoring

Each question has one correct answer from 4 choices. The model's response is compared against the target answer using exact match.

**Reference performance:** RAG-based ORANSight achieves ~78% on the full ORAN-Bench-13K; general-purpose LLMs score 21-23% lower.

## Metrics

- **accuracy**: Fraction of correct answers
- **stderr**: Standard error of the accuracy estimate

## Links

- [Paper](https://arxiv.org/abs/2407.06245)
- [Dataset](https://huggingface.co/datasets/prnshv/ORANBench)
- [GitHub](https://github.com/prnshv/ORAN-Bench-13K)
