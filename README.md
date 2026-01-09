<p align="center">
  <img src="docs/imgs/open_telco.svg" alt="GSMA Open_Telco" width="400">
</p>

# Open Telco

A suite of telco-specific benchmarks for evaluating AI models on telecommunications tasks. Built on [Inspect AI](https://inspect.aisi.org.uk/), Open Telco provides standardized evaluations for knowledge, reasoning, and operational capabilities in the telecom domain.

## Quick Start

```bash
# Install
git clone https://github.com/gsma-research/open_telco.git && cd open_telco && uv sync

# Configure (create .env with HF_TOKEN and model API key)

# Run your first evaluation
uv run inspect eval src/open_telco/teleqna/teleqna.py --model openai/gpt-4o --limit 20
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, configuration, prerequisites |
| [Quickstart](docs/quickstart.md) | Run your first evaluation in 5 minutes |
| [List of Evaluations](docs/eval-list.md) | Available benchmarks and what they measure |
| [Running Evaluations](docs/running-evaluations.md) | Advanced usage, eval sets, multi-model runs |
| [FAQ](docs/faq.md) | Common questions and troubleshooting |

## Available Benchmarks

| Benchmark | Description |
|-----------|-------------|
| **TeleQnA** | 10,000 Q&A pairs on telecom knowledge |
| **TeleMath** | Mathematical reasoning in telecom domain |
| **TeleLogs** | Root cause analysis in 5G networks |
| **3GPP TSG** | Technical standards classification |
| **TeleYAML** | Network configuration generation (coming soon) |

## Why Open Telco?

We are developing evaluations that are realistic and address the complementary capabilities necessary to ensure safe and optimal deployment of AI in a telco environment.

If you share this mission, please [reach out](mailto:emolero@gsma.com) - we are always looking for collaborators and contributors!

## Collaborators

**Tech & Research:** GSMA, Huawei GTS, The Linux Foundation, Khalifa University, Universitat Pompeu Fabra (UPF), University of Texas, and Queen’s University.

**Telcos:** AT&T, China Telecom, Deutsche Telekom, du, KDDI, KPN, Liberty Global, Orange, Telefónica, Turkcell, Swisscom, Vodafone.

**Industry Labs & SMEs:** NetoAI, Datumo, Adaptive-AI

