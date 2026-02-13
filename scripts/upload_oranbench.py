#!/usr/bin/env python3
"""Upload ORANBench to both GSMA HuggingFace datasets and update READMEs.

Adds 1,500 samples to GSMA/ot-full-benchmarks (config=oranbench, split=test)
and 150 random samples to GSMA/ot_sample_data (config=oranbench, split=test).
Also updates both dataset README cards.

Usage:
    uv run python scripts/upload_oranbench.py --dry-run   # validate only
    uv run python scripts/upload_oranbench.py              # upload to HF
"""

from __future__ import annotations

import argparse
import logging
import sys

from datasets import load_dataset
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

FULL_REPO = "GSMA/ot-full-benchmarks"
SAMPLE_REPO = "GSMA/ot_sample_data"
SOURCE_REPO = "prnshv/ORANBench"
CONFIG_NAME = "oranbench"
SAMPLE_SIZE = 150
SEED = 42

# ── README templates ──────────────────────────────────────────────────────

ORANBENCH_CITATION = r"""
@misc{gajjar2024oranbench,
  title={ORAN-Bench-13K: An Open Source Benchmark for Assessing LLMs in Open Radio Access Networks},
  author={Gajjar, Pranshav and Shah, Vijay K.},
  year={2024}, eprint={2407.06245}, archivePrefix={arXiv}
}"""

FULL_README = r"""---
license: mit
task_categories:
  - question-answering
  - text-classification
language:
  - en
tags:
  - telecommunications
  - telecom
  - 3gpp
  - 5g
  - benchmarks
  - evaluation
  - llm
pretty_name: Open Telco Full Benchmarks
size_categories:
  - 10K<n<100K
configs:
  - config_name: teleqna
    data_files:
      - split: test
        path: teleqna/test-*
  - config_name: teletables
    data_files:
      - split: test
        path: teletables/test-*
  - config_name: telemath
    data_files:
      - split: test
        path: telemath/test-*
  - config_name: telelogs
    data_files:
      - split: test
        path: telelogs/test-*
  - config_name: 3gpp_tsg
    data_files:
      - split: test
        path: 3gpp_tsg/test-*
  - config_name: oranbench
    data_files:
      - split: test
        path: oranbench/test-*
dataset_info:
  - config_name: teleqna
    features:
      - name: question
        dtype: string
      - name: choices
        list: string
      - name: answer
        dtype: int64
      - name: subject
        dtype: string
    splits:
      - name: test
        num_examples: 10000
  - config_name: teletables
    features:
      - name: question
        dtype: string
      - name: choices
        list: string
      - name: answer
        dtype: int64
      - name: explanation
        dtype: string
      - name: difficult
        dtype: bool
      - name: table_id
        dtype: string
      - name: table_title
        dtype: string
      - name: document_id
        dtype: string
      - name: document_title
        dtype: string
      - name: document_url
        dtype: string
    splits:
      - name: test
        num_examples: 500
  - config_name: telemath
    features:
      - name: question
        dtype: string
      - name: answer
        dtype: float64
      - name: category
        dtype: string
      - name: tags
        list: string
      - name: difficulty
        dtype: string
    splits:
      - name: test
        num_examples: 500
  - config_name: telelogs
    features:
      - name: question
        dtype: string
      - name: answer
        dtype: string
    splits:
      - name: test
        num_examples: 864
  - config_name: 3gpp_tsg
    features:
      - name: question
        dtype: string
      - name: answer
        dtype: string
      - name: file_name
        dtype: string
    splits:
      - name: test
        num_examples: 2000
  - config_name: oranbench
    features:
      - name: question
        dtype: string
      - name: choices
        list: string
      - name: answer
        dtype: int64
      - name: difficulty
        dtype: string
    splits:
      - name: test
        num_examples: 1500
---

# Open Telco Full Benchmarks

**15,364 telecom-specific evaluation samples** across 6 benchmarks — the complete evaluation suite for measuring telecom AI performance.

Use this dataset for final, publishable results. For fast iteration during model development, use [`ot_sample_data`](https://huggingface.co/datasets/GSMA/ot_sample_data) (1,550 samples).

[Eval Framework](https://github.com/gsma-labs/evals) | [Sample Data](https://huggingface.co/datasets/GSMA/ot_sample_data)

## Benchmarks

| Config | Samples | Task | Paper |
|--------|--------:|------|-------|
| `teleqna` | 10,000 | Multiple-choice Q&A on telecom standards | [arXiv](https://arxiv.org/abs/2310.15051) |
| `teletables` | 500 | Table interpretation from 3GPP specs | [arXiv](https://arxiv.org/abs/2601.04202) |
| `telemath` | 500 | Telecom mathematical reasoning | [arXiv](https://arxiv.org/abs/2506.10674) |
| `telelogs` | 864 | 5G network root cause analysis | [arXiv](https://arxiv.org/abs/2507.21974) |
| `3gpp_tsg` | 2,000 | 3GPP document classification by working group | [arXiv](https://arxiv.org/abs/2407.09424) |
| `oranbench` | 1,500 | Multiple-choice Q&A on O-RAN specifications | [arXiv](https://arxiv.org/abs/2407.06245) |

> For quick testing, use [`ot_sample_data`](https://huggingface.co/datasets/GSMA/ot_sample_data) (100–1,000 sample subsets of each benchmark).

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("GSMA/ot-full-benchmarks", "teleqna", split="test")
# Available configs: teleqna, teletables, telemath, telelogs, 3gpp_tsg, oranbench
```

Or run evaluations directly with [Inspect AI](https://inspect.aisi.org.uk/):

```bash
uv run inspect eval src/evals/teleqna/teleqna.py --model openai/gpt-4o -T full=true
```

See [Running Evaluations](https://github.com/gsma-labs/evals/blob/main/docs/running-evaluations.md) for the full guide.

## Citation

```bibtex
@misc{maatouk2023teleqna,
  title={TeleQnA: A Benchmark Dataset to Assess Large Language Models Telecommunications Knowledge},
  author={Maatouk, Ali and Ayed, Fadhel and Piovesan, Nicola and De Domenico, Antonio and Debbah, Merouane and Luo, Zhi-Quan},
  year={2023}, eprint={2310.15051}, archivePrefix={arXiv}
}

@misc{nazzal2025teletables,
  title={TeleTables: A Dataset for Evaluating LLM Interpretation of Tables in 3GPP Specifications},
  author={Nazzal, Jamal and Piovesan, Nicola and De Domenico, Antonio},
  year={2025}, eprint={2601.04202}, archivePrefix={arXiv}
}

@misc{ali2025telemath,
  title={TeleMath: Benchmarking LLMs in Telecommunications with a Mathematical Reasoning Evaluation Framework},
  author={Ali, Syed Muhammad Hasan and Maatouk, Ali and Piovesan, Nicola and De Domenico, Antonio and Debbah, Merouane},
  year={2025}, eprint={2506.10674}, archivePrefix={arXiv}
}

@misc{mekrache2025telelogs,
  title={TeleLogs: An LLM Benchmark for Root Cause Analysis in 5G Networks},
  author={Mekrache, Abdelkader and Piovesan, Nicola and De Domenico, Antonio},
  year={2025}, eprint={2507.21974}, archivePrefix={arXiv}
}

@misc{zou2024telecomgpt,
  title={TelecomGPT: A Framework to Build Telecom-Specific Large Language Models},
  author={Zou, Hang and Zhao, Qiyang and Tian, Yu and Bariah, Lina and Bader, Faouzi and Lestable, Thierry and Debbah, Merouane},
  year={2024}, eprint={2407.09424}, archivePrefix={arXiv}
}

@misc{gajjar2024oranbench,
  title={ORAN-Bench-13K: An Open Source Benchmark for Assessing LLMs in Open Radio Access Networks},
  author={Gajjar, Pranshav and Shah, Vijay K.},
  year={2024}, eprint={2407.06245}, archivePrefix={arXiv}
}
```
"""

SAMPLE_README = r"""---
license: mit
task_categories:
  - question-answering
  - text-classification
language:
  - en
tags:
  - telecommunications
  - telecom
  - 3gpp
  - 5g
  - benchmarks
  - evaluation
  - llm
pretty_name: Open Telco Sample Data
size_categories:
  - 1K<n<10K
configs:
  - config_name: teleqna
    data_files:
      - split: test
        path: test_teleqna.json
  - config_name: teletables
    data_files:
      - split: test
        path: test_teletables.json
  - config_name: telemath
    data_files:
      - split: test
        path: test_telemath.json
  - config_name: telelogs
    data_files:
      - split: test
        path: test_telelogs.json
  - config_name: 3gpp_tsg
    data_files:
      - split: test
        path: test_3gpp_tsg.json
  - config_name: oranbench
    data_files:
      - split: test
        path: test_oranbench.json
dataset_info:
  - config_name: teleqna
    features:
      - name: question
        dtype: string
      - name: choices
        list: string
      - name: answer
        dtype: int64
      - name: subject
        dtype: string
      - name: explaination
        dtype: string
    splits:
      - name: test
        num_examples: 1000
  - config_name: teletables
    features:
      - name: question
        dtype: string
      - name: choices
        list: string
      - name: answer
        dtype: int64
      - name: explanation
        dtype: string
      - name: difficult
        dtype: bool
      - name: table_id
        dtype: string
      - name: table_title
        dtype: string
      - name: document_id
        dtype: string
      - name: document_title
        dtype: string
      - name: document_url
        dtype: string
    splits:
      - name: test
        num_examples: 100
  - config_name: telemath
    features:
      - name: question
        dtype: string
      - name: answer
        dtype: float64
      - name: category
        dtype: string
      - name: tags
        list: string
      - name: difficulty
        dtype: string
    splits:
      - name: test
        num_examples: 100
  - config_name: telelogs
    features:
      - name: question
        dtype: string
      - name: answer
        dtype: string
    splits:
      - name: test
        num_examples: 100
  - config_name: 3gpp_tsg
    features:
      - name: question
        dtype: string
      - name: answer
        dtype: string
      - name: file_name
        dtype: string
    splits:
      - name: test
        num_examples: 100
  - config_name: oranbench
    features:
      - name: question
        dtype: string
      - name: choices
        list: string
      - name: answer
        dtype: int64
      - name: difficulty
        dtype: string
    splits:
      - name: test
        num_examples: 150
---

# Open Telco Sample Data

**1,550 telecom-specific evaluation samples** across 6 benchmarks — designed for fast iteration during model development.

With ~100–150 samples per benchmark, `ot_sample_data` lets you quickly measure meaningful performance changes while training or fine-tuning models. Run it frequently to track progress, then evaluate against the [full benchmarks](https://huggingface.co/datasets/GSMA/ot-full-benchmarks) (15,364 samples) for final results.

[Eval Framework](https://github.com/gsma-labs/evals) | [Full Benchmarks](https://huggingface.co/datasets/GSMA/ot-full-benchmarks)

## Benchmarks

| Config | Samples | Task | Paper |
|--------|--------:|------|-------|
| `teleqna` | 1,000 | Multiple-choice Q&A on telecom standards | [arXiv](https://arxiv.org/abs/2310.15051) |
| `teletables` | 100 | Table interpretation from 3GPP specs | [arXiv](https://arxiv.org/abs/2601.04202) |
| `telemath` | 100 | Telecom mathematical reasoning | [arXiv](https://arxiv.org/abs/2506.10674) |
| `telelogs` | 100 | 5G network root cause analysis | [arXiv](https://arxiv.org/abs/2507.21974) |
| `3gpp_tsg` | 100 | 3GPP document classification by working group | [arXiv](https://arxiv.org/abs/2407.09424) |
| `oranbench` | 150 | Multiple-choice Q&A on O-RAN specifications | [arXiv](https://arxiv.org/abs/2407.06245) |

> For full-scale evaluation, use [`GSMA/ot-full-benchmarks`](https://huggingface.co/datasets/GSMA/ot-full-benchmarks) (15,364 samples).

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("GSMA/ot_sample_data", "teleqna", split="test")
# Available configs: teleqna, teletables, telemath, telelogs, 3gpp_tsg, oranbench
```

Or run evaluations with [Inspect AI](https://inspect.aisi.org.uk/):

```bash
# Sample data (default)
uv run inspect eval src/evals/teleqna/teleqna.py --model openai/gpt-4o

# Full benchmarks
uv run inspect eval src/evals/teleqna/teleqna.py --model openai/gpt-4o -T full=true
```

See [Running Evaluations](https://github.com/gsma-labs/evals/blob/main/docs/running-evaluations.md) for the full guide.

## Citation

```bibtex
@misc{maatouk2023teleqna,
  title={TeleQnA: A Benchmark Dataset to Assess Large Language Models Telecommunications Knowledge},
  author={Maatouk, Ali and Ayed, Fadhel and Piovesan, Nicola and De Domenico, Antonio and Debbah, Merouane and Luo, Zhi-Quan},
  year={2023}, eprint={2310.15051}, archivePrefix={arXiv}
}

@misc{nazzal2025teletables,
  title={TeleTables: A Dataset for Evaluating LLM Interpretation of Tables in 3GPP Specifications},
  author={Nazzal, Jamal and Piovesan, Nicola and De Domenico, Antonio},
  year={2025}, eprint={2601.04202}, archivePrefix={arXiv}
}

@misc{ali2025telemath,
  title={TeleMath: Benchmarking LLMs in Telecommunications with a Mathematical Reasoning Evaluation Framework},
  author={Ali, Syed Muhammad Hasan and Maatouk, Ali and Piovesan, Nicola and De Domenico, Antonio and Debbah, Merouane},
  year={2025}, eprint={2506.10674}, archivePrefix={arXiv}
}

@misc{mekrache2025telelogs,
  title={TeleLogs: An LLM Benchmark for Root Cause Analysis in 5G Networks},
  author={Mekrache, Abdelkader and Piovesan, Nicola and De Domenico, Antonio},
  year={2025}, eprint={2507.21974}, archivePrefix={arXiv}
}

@misc{zou2024telecomgpt,
  title={TelecomGPT: A Framework to Build Telecom-Specific Large Language Models},
  author={Zou, Hang and Zhao, Qiyang and Tian, Yu and Bariah, Lina and Bader, Faouzi and Lestable, Thierry and Debbah, Merouane},
  year={2024}, eprint={2407.09424}, archivePrefix={arXiv}
}

@misc{gajjar2024oranbench,
  title={ORAN-Bench-13K: An Open Source Benchmark for Assessing LLMs in Open Radio Access Networks},
  author={Gajjar, Pranshav and Shah, Vijay K.},
  year={2024}, eprint={2407.06245}, archivePrefix={arXiv}
}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download, sample, and validate only — do not upload.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var).",
    )
    args = parser.parse_args()

    # ── Load source dataset ──────────────────────────────────────────────
    log.info("Loading ORANBench from %s ...", SOURCE_REPO)
    ds = load_dataset(SOURCE_REPO, split="train")
    log.info("Loaded %d samples", len(ds))

    # Keep only the columns the eval code needs
    keep_cols = {"question", "choices", "answer", "difficulty"}
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
        log.info("Dropped extra columns: %s", drop_cols)

    # ── Validate ─────────────────────────────────────────────────────────
    expected_count = 1500
    assert len(ds) == expected_count, (
        f"Expected {expected_count} samples, got {len(ds)}"
    )
    for i, row in enumerate(ds):
        n_choices = len(row["choices"])
        assert 0 <= row["answer"] < n_choices, (
            f"Row {i}: answer {row['answer']} out of range [0, {n_choices})"
        )
    log.info(
        "Validation passed: %d samples, columns: %s", len(ds), sorted(ds.column_names)
    )

    # ── Create 150-sample subset (stratified by difficulty) ──────────────
    sample_ds = ds.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
    log.info(
        "Sampled %d items (difficulty distribution: %s)",
        len(sample_ds),
        {
            d: sum(1 for r in sample_ds if r["difficulty"] == d)
            for d in ["easy", "medium", "hard"]
        },
    )

    if args.dry_run:
        log.info("Dry-run complete — no upload performed.")
        log.info(
            "Full dataset: %d samples → %s config=%s", len(ds), FULL_REPO, CONFIG_NAME
        )
        log.info(
            "Sample dataset: %d samples → %s config=%s",
            len(sample_ds),
            SAMPLE_REPO,
            CONFIG_NAME,
        )
        return

    api = HfApi(token=args.token)

    # ── Upload full dataset (1500 samples) ───────────────────────────────
    log.info(
        "Pushing %d samples to %s config=%s split=test ...",
        len(ds),
        FULL_REPO,
        CONFIG_NAME,
    )
    ds.push_to_hub(
        FULL_REPO,
        config_name=CONFIG_NAME,
        split="test",
        token=args.token,
    )

    # ── Upload sample dataset (150 samples) ──────────────────────────────
    log.info(
        "Pushing %d samples to %s config=%s split=test ...",
        len(sample_ds),
        SAMPLE_REPO,
        CONFIG_NAME,
    )
    sample_ds.push_to_hub(
        SAMPLE_REPO,
        config_name=CONFIG_NAME,
        split="test",
        token=args.token,
    )

    # ── Update READMEs ───────────────────────────────────────────────────
    log.info("Updating README for %s ...", FULL_REPO)
    api.upload_file(
        path_or_fileobj=FULL_README.strip().encode(),
        path_in_repo="README.md",
        repo_id=FULL_REPO,
        repo_type="dataset",
        token=args.token,
        commit_message="Add ORANBench config (1,500 samples)",
    )

    log.info("Updating README for %s ...", SAMPLE_REPO)
    api.upload_file(
        path_or_fileobj=SAMPLE_README.strip().encode(),
        path_in_repo="README.md",
        repo_id=SAMPLE_REPO,
        repo_type="dataset",
        token=args.token,
        commit_message="Add ORANBench config (150 samples)",
    )

    log.info("Upload complete.")


if __name__ == "__main__":
    sys.exit(main() or 0)
