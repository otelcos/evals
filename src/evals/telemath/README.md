```
████████╗███████╗██╗     ███████╗███╗   ███╗ █████╗ ████████╗██╗  ██╗
╚══██╔══╝██╔════╝██║     ██╔════╝████╗ ████║██╔══██╗╚══██╔══╝██║  ██║
   ██║   █████╗  ██║     █████╗  ██╔████╔██║███████║   ██║   ███████║
   ██║   ██╔══╝  ██║     ██╔══╝  ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║
   ██║   ███████╗███████╗███████╗██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║
   ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
```

Evaluating LLM Capability for Mathematical Reasoning in the Telecom Domain.

## Overview

500 mathematically intensive problems covering signal processing, network optimization, and performance analysis. The model is prompted to solve domain-specific mathematical computations step-by-step and provide the final numerical answer in LaTeX boxed format.

## Usage

```bash
uv run inspect eval src/evals/telemath/telemath.py --model openai/gpt-4o
```

## Scoring

The model must output its final numerical answer in `\boxed{...}` format. The scorer:
1. Extracts the last `\boxed{...}` value from the response
2. Normalizes both prediction and target (e.g., `5.0` → `5`)
3. Marks **CORRECT** if values match exactly or after normalization

## Metrics

- **accuracy**: Fraction of correct answers (pass@1)
- **stderr**: Standard error of the accuracy estimate

## Links

- [Paper](https://arxiv.org/abs/2506.10674)
- [Dataset](https://huggingface.co/datasets/netop/TeleMath)
