```
███████╗██████╗ ███████╗██████╗  █████╗ ███╗   ██╗██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██╔════╝██╔══██╗██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
███████╗██████╔╝███████╗██████╔╝███████║██╔██╗ ██║██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
╚════██║██╔══██╗╚════██║██╔══██╗██╔══██║██║╚██╗██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
███████║██║  ██║███████║██║  ██║██║  ██║██║ ╚████║██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
```

Evaluating LLM Understanding of the srsRAN 5G Open-Source Codebase.

## Overview

1,502 multiple-choice questions testing LLM comprehension of the srsRAN Project codebase — an open-source 5G RAN implementation. Questions cover C++ classes, functions, libraries, configurations, and 3GPP specification details as implemented in srsRAN.

## Usage

```bash
uv run inspect eval src/evals/srsranbench/srsranbench.py --model openai/gpt-4o
```

## Scoring

Each question has one correct answer from 4 choices. The model's response is compared against the target answer using exact match.

## Metrics

- **accuracy**: Fraction of correct answers
- **stderr**: Standard error of the accuracy estimate

## Links

- [Paper](https://arxiv.org/abs/2407.06245)
- [Dataset](https://huggingface.co/datasets/prnshv/srsRANBench)
- [srsRAN Project](https://github.com/srsran/srsRAN_Project)
