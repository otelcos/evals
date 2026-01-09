---
id: datasets
title: Datasets & Methodology
sidebar_label: Datasets & Methodology
sidebar_position: 1
---

# Datasets & Methodology

Open Telco provides curated datasets specifically designed for evaluating language models in telecommunications domains.

## Hugging Face Collection

All our datasets are available on Hugging Face:

**[huggingface.co/datasets/GSMA/open_telco](https://huggingface.co/datasets/GSMA/open_telco)**

## Available Datasets

### TeleQnA

A comprehensive question-answering dataset covering telecommunications knowledge.

| Attribute | Value |
|-----------|-------|
| **Size** | 10,000 question-answer pairs |
| **Source** | Telecom standards and research papers |
| **Format** | Multiple choice |
| **Topics** | 5G, LTE, network protocols, standards |

**Usage:**
```python
from datasets import load_dataset

dataset = load_dataset("GSMA/open_telco", "teleqna")
```

### TeleMath

Mathematical reasoning problems in telecommunications contexts.

| Attribute | Value |
|-----------|-------|
| **Size** | 500 problems |
| **Difficulty** | Intermediate to Advanced |
| **Topics** | Signal processing, network optimization, performance analysis |

**Usage:**
```python
dataset = load_dataset("GSMA/open_telco", "telemath")
```

### TeleLogs

Synthetic network logs for root cause analysis tasks.

| Attribute | Value |
|-----------|-------|
| **Size** | Synthetic dataset |
| **Format** | Network logs with annotations |
| **Task** | Diagnose throughput degradation in 5G networks |

**Usage:**
```python
dataset = load_dataset("GSMA/open_telco", "telelogs")
```

### 3GPP TSG

Technical specification group document classification dataset.

| Attribute | Value |
|-----------|-------|
| **Source** | 3GPP technical documents |
| **Task** | Classify documents by working group |
| **Categories** | Multiple TSG working groups |

**Usage:**
```python
dataset = load_dataset("GSMA/open_telco", "three_gpp")
```

## Methodology

### Dataset Creation

Our datasets are created through:

1. **Expert Curation**: Telecommunications experts review and validate content
2. **Standards Alignment**: Questions align with 3GPP and other industry standards
3. **Difficulty Balancing**: Mix of basic, intermediate, and advanced questions
4. **Regular Updates**: Datasets are updated to reflect current standards

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct answers |
| **F1 Score** | Balance of precision and recall |
| **Maj@K** | Majority voting across K samples |

### Quality Assurance

- Multiple expert reviews per question
- Automated validation checks
- Community feedback integration
- Regular benchmark updates

## Data Access

### Requirements

To access the datasets, you need:

1. A Hugging Face account
2. Acceptance of the dataset terms
3. A valid HF_TOKEN in your environment

### Configuration

```bash
# Set your Hugging Face token
export HF_TOKEN=your_token_here

# Or add to .env file
echo "HF_TOKEN=your_token_here" >> .env
```

## Contributing Data

We welcome contributions to expand our datasets:

1. Review our [contribution guidelines](/docs/contributing)
2. Ensure data quality meets our standards
3. Submit via pull request with documentation
4. Include licensing information

## Citation

If you use our datasets in your research, please cite:

```bibtex
@dataset{gsma_open_telco_2024,
  title={Open Telco: Telecommunications Evaluation Datasets},
  author={GSMA Research},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/GSMA/open_telco}
}
```
