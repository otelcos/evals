# List of Evals

## Knowledge & QA

**[TeleQnA](../src/open_telco/teleqna/): Benchmark Dataset to Assess Large Language Models for Telecommunications**

A benchmark dataset of 10,000 question-answer pairs sourced from telecommunications standards and research articles. Evaluates LLMs' knowledge across general telecom inquiries and complex standards-related questions.

```bash
uv run inspect eval src/open_telco/teleqna/teleqna.py --model <model>
```

[Paper](https://arxiv.org/abs/2310.15051) | [Dataset](https://huggingface.co/datasets/netop/TeleQnA)

## Mathematical Reasoning

**[TeleMath](../src/open_telco/telemath/): Evaluating Mathematical Reasoning in Telecom Domain**

500 mathematically intensive problems covering signal processing, network optimization, and performance analysis. Implemented as a ReAct agent using bash and python tools to solve domain-specific mathematical computations.

```bash
uv run inspect eval src/open_telco/telemath/telemath.py --model <model>
```

[Paper](https://arxiv.org/abs/2506.10674) | [Dataset](https://huggingface.co/datasets/netop/TeleMath)

## Network Operations & Diagnostics

**[TeleLogs](../src/open_telco/telelogs/): Root Cause Analysis in 5G Networks**

A synthetic dataset for root cause analysis (RCA) in 5G networks. Given network configuration parameters and user-plane data (throughput, RSRP, SINR), models must identify which of 8 predefined root causes explain throughput degradation below 600 Mbps.

```bash
uv run inspect eval src/open_telco/telelogs/telelogs.py --model <model>
```

[Paper](https://arxiv.org/abs/2507.21974) | [Dataset](https://huggingface.co/datasets/netop/TeleLogs)

## Network Configuration

**TeleYaml: 5G Network Configuration Generation** *(In Progress)*

Evaluates the capability of LLMs to generate standard-compliant YAML configurations for 5G core network tasks: AMF Configuration, Network Slicing, and UE Provisioning. This benchmark is currently being revamped.

[Dataset](https://huggingface.co/datasets/otellm/gsma-sample-data)

## Standardization

**[3GPP TSG](../src/open_telco/three_gpp/): Technical Specification Group Classification**

Classifies 3GPP technical documents according to their working group. Models must identify the correct group for a given technical text.

```bash
uv run inspect eval src/open_telco/three_gpp/three_gpp.py --model <model>
```

[Dataset](https://huggingface.co/datasets/eaguaida/gsma_sample)
