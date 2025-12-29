```
██████╗  ██████╗ ██████╗ ██████╗         ████████╗███████╗ ██████╗
╚════██╗██╔════╝ ██╔══██╗██╔══██╗        ╚══██╔══╝██╔════╝██╔════╝
 █████╔╝██║  ███╗██████╔╝██████╔╝ █████╗    ██║   ███████╗██║  ███╗
 ╚═══██╗██║   ██║██╔═══╝ ██╔═══╝  ╚════╝    ██║   ╚════██║██║   ██║
██████╔╝╚██████╔╝██║     ██║                ██║   ███████║╚██████╔╝
╚═════╝  ╚═════╝ ╚═╝     ╚═╝                ╚═╝   ╚══════╝ ╚═════╝
```

Evaluating LLM Capability for interpret and categorize complex 3GPP technical specifications.

## Overview

Classifies 3GPP technical documents by working group. Models must identify the correct TSG group (e.g., RAN1, SA2, CT4) for a given technical text.

## Usage

```bash
uv run inspect eval src/open_telco/three_gpp/three_gpp.py --model openai/gpt-4o
```

## Scoring

Uses **pattern matching** with regex `([A-Z]+\d+(?:-[A-Z]+)?)` to extract working group codes from model responses. A response is **CORRECT** if the extracted pattern matches the target group (case-insensitive).

| Response Example | Target | Result |
|------------------|--------|--------|
| "This belongs to RAN1" | RAN1 | CORRECT |
| "SA2-RAN handles this" | SA2-RAN | CORRECT |
| "I think CT3" | CT4 | INCORRECT |

## Metrics

- **accuracy**: Fraction of correct classifications
- **stderr**: Standard error of the accuracy estimate

## Links

- [Dataset](https://huggingface.co/datasets/eaguaida/gsma_sample)
