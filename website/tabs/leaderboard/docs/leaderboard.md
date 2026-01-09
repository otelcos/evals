---
id: leaderboard
title: Leaderboard
sidebar_label: Leaderboard
slug: /
---

# Leaderboard

Track the performance of language models across Open Telco evaluations.

## Overall Rankings

| Rank | Provider | Model | Mean | TeleQnA | TeleLogs | TeleMath | 3GPP-TSG |
|------|----------|-------|------|---------|----------|----------|----------|
| 1 | OpenAI | GPT-5 | **65.55** | 82.51 | 80.00 | 70.27 | 67.90 |
| 2 | Grok | Grok-4-fast | **61.52** | 79.39 | 78.12 | 62.80 | 60.60 |
| 3 | Claude | Claude-Sonnet-4.5 | **60.64** | 80.57 | 67.12 | 62.80 | 65.25 |
| 4 | Google | Gemini-2.5-pro | **58.44** | 80.11 | 44.30 | 74.40 | 68.20 |
| 5 | NetoAI | TSLAM-18B | **49.93** | 72.00 | 20.62 | 69.50 | 63.50 |
| 6 | OpenAI | GPT-OSS-120B | **49.71** | 78.51 | 44.70 | 60.40 | 36.25 |
| 7 | Qwen | Qwen3-32B | **47.46** | 76.14 | 33.77 | 69.51 | 34.78 |
| 8 | Mistral | Mistral-Large-123B | **44.93** | 75.85 | 30.55 | 38.80 | 54.95 |
| 9 | OpenAI | GPT-OSS-20B | **43.97** | 75.79 | 40.10 | 53.80 | 30.18 |
| 10 | Meta | Llama-3.3-70B-Instruct | **42.40** | 74.98 | 22.01 | 36.23 | 55.18 |
| 11 | Qwen | Qwen3-8B | **42.18** | 73.21 | 36.42 | 49.73 | 31.42 |
| 12 | Qwen | Qwen3-4B | **40.57** | 70.50 | 32.00 | 45.62 | 31.65 |
| 13 | NetoAI | TSLAM-G3 | **40.10** | 82.50 | 11.25 | 26.50 | 58.50 |
| 14 | IBM Granite | granite-4.0-h-small | **35.16** | 72.15 | 17.24 | 32.40 | 32.53 |
| 15 | DeepSeek | DeepSeek-R1-Distill-Qwen-7B | **29.48** | 69.31 | 12.05 | 24.90 | 21.43 |
| 16 | Meta | Llama-3.1-8B-Instruct | **28.11** | 68.03 | 13.42 | 13.56 | 25.27 |
| 17 | IBM Granite | granite-3.3-8b-instruct | **27.28** | 62.35 | 13.31 | 14.37 | 26.00 |
| 18 | LiquidAI | LFM2-2.6B | **25.79** | 57.90 | 9.08 | 18.05 | 24.20 |
| 19 | NetoAI | TSLAM-2B MINI | **25.35** | 62.00 | 13.50 | 4.50 | 27.00 |
| 20 | Microsoft | Phi-4-mini-instruct | **22.45** | 45.90 | 6.56 | 14.40 | 24.87 |
| 21 | Swiss AI | Apertus-8B-Instruct-2509 | **21.52** | 56.40 | 4.25 | 6.03 | 20.38 |

### Partial Results

| Rank | Provider | Model | Mean | TeleQnA | TeleLogs | TeleMath | 3GPP-TSG |
|------|----------|-------|------|---------|----------|----------|----------|
| — | ByteDance | Seed-OSS-36B | — | 75.67 | 57.00 | 56.05 | 37.66 |
| — | Google | AT&T FT Gemma-3-4B-IT | — | — | 80.09 | — | — |

## Evaluation Details

### Scoring Methods

- **TeleQnA, TeleLogs, TeleMath, 3GPP-TSG**: Raw accuracy scores

### Model Repository Links

| Provider | Model | Repository |
|----------|-------|------------|
| OpenAI | GPT-5 | `openai/gpt-5` |
| Grok | Grok-4-fast | `grok/grok-4-fast` |
| Claude | Claude-Sonnet-4.5 | `claude/claude-sonnet-4.5` |
| Google | Gemini-2.5-pro | `google/gemini-2.5-pro` |
| NetoAI | TSLAM-18B | `NetoAI/TSLAM-18B` |
| OpenAI | GPT-OSS-120B | `openai/gpt-oss-120b` |
| Qwen | Qwen3-32B | `qwen/qwen3-32b` |
| Mistral | Mistral-Large-123B | `mistralai/Mistral-Large-Instruct-2411` |
| OpenAI | GPT-OSS-20B | `openai/gpt-oss-20b` |
| Meta | Llama-3.3-70B-Instruct | `meta/llama-3.3-70b-instruct` |
| Qwen | Qwen3-8B | `qwen/qwen3-8b` |
| Qwen | Qwen3-4B | `qwen/qwen3-4b` |
| NetoAI | TSLAM-G3 | `NetoAI/TSLAM-G3` |
| IBM Granite | granite-4.0-h-small | `ibm-granite/granite-4.0-h-small` |
| DeepSeek | DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| Meta | Llama-3.1-8B-Instruct | `meta/llama-3.1-8b-instruct` |
| IBM Granite | granite-3.3-8b-instruct | `ibm-granite/granite-3.3-8b-instruct` |
| LiquidAI | LFM2-2.6B | `liquidai/LFM2-2.6B` |
| NetoAI | TSLAM-2B MINI | `NetoAI/TSLAM-2B MINI` |
| Microsoft | Phi-4-mini-instruct | `microsoft/phi-4-mini-instruct` |
| Swiss AI | Apertus-8B-Instruct-2509 | `swiss-ai/Apertus-8B-Instruct-2509` |
| ByteDance | Seed-OSS-36B | `bytedance/seed-oss-36b-instruct` |
| Google | AT&T FT Gemma-3-4B-IT | `AT&T/gemma-3-4b-fine-tuned` |

## Submission Guidelines

Want to add your model to the leaderboard?

1. **Run the full evaluation suite** using the standard configuration
2. **Document your setup** including model version, temperature, and parameters
3. **Submit via GitHub** - Open an issue with your results and evaluation logs

### Required Information

- Model name and version
- Provider (OpenAI, Anthropic, etc.)
- Evaluation date
- Full evaluation logs
- Any custom configurations used

## Evaluation Standards

To ensure fair comparison, all leaderboard entries must:

- Use **temperature 0.0** for deterministic outputs
- Run on the **full dataset** (no sampling)
- Use the **standard prompts** provided by Open Telco
- Report results from a **single run** (no cherry-picking)
