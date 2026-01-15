```
████████╗███████╗██╗     ███████╗██╗      ██████╗  ██████╗ ███████╗
╚══██╔══╝██╔════╝██║     ██╔════╝██║     ██╔═══██╗██╔════╝ ██╔════╝
   ██║   █████╗  ██║     █████╗  ██║     ██║   ██║██║  ███╗███████╗
   ██║   ██╔══╝  ██║     ██╔══╝  ██║     ██║   ██║██║   ██║╚════██║
   ██║   ███████╗███████╗███████╗███████╗╚██████╔╝╚██████╔╝███████║
   ╚═╝   ╚══════╝╚══════╝╚══════╝╚══════╝ ╚═════╝  ╚═════╝ ╚══════╝
```

Evaluating LLM Capability for Root Cause Analysis in 5G Networks.

## Overview

A synthetic dataset for root cause analysis (RCA) in 5G networks. Given network configuration parameters and user-plane data (throughput, RSRP, SINR), models must identify which of 8 predefined root causes explain throughput degradation below 600 Mbps.

## Usage

```bash
uv run inspect eval src/evals/telelogs/telelogs.py -T 4 --model openai/gpt-4o
```

Use `-T <N>` to specify epochs for majority voting.

## Scoring

Models output answers in `\boxed{N}` format where N is the root cause ID (1-8). The scorer extracts the integer and compares against ground truth.

- **Soft eval** (default): Matches first integer found
- **Hard eval**: Requires exact string match

## Metrics

- **pass@1**: Fraction of correct answers (averaged over N epochs)
- **maj@4**: Accuracy using majority voting across 4 epochs

## Links

- [Paper](https://arxiv.org/abs/2507.21974)
- [Dataset](https://huggingface.co/datasets/netop/TeleLogs)
