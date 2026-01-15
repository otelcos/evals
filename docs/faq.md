# Frequently Asked Questions

## Getting Started

### What models can I use?

Open Telco supports any model available through [Inspect AI](https://inspect.aisi.org.uk/models.html), including:

| Provider | Example Models | API Key Variable |
|----------|---------------|------------------|
| OpenAI | `openai/gpt-4o`, `openai/gpt-4-turbo` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-3-5-haiku-latest` | `ANTHROPIC_API_KEY` |
| OpenRouter | `openrouter/google/gemini-2.0-flash-001`, `openrouter/meta-llama/llama-3.1-70b-instruct` | `OPENROUTER_API_KEY` |
| Local | `ollama/llama3`, `vllm/mistral-7b` | - |

### How do I get a HuggingFace token?

1. Create an account at [huggingface.co](https://huggingface.co/join)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Click "New token" and create a token with read access
4. Add it to your `.env` file as `HF_TOKEN=your_token_here`

### Which benchmark should I start with?

**TeleQnA** is the best starting point:
- Fastest to run (simple Q&A format)
- Broad coverage of telecom topics
- Good baseline for model comparison

```bash
uv run inspect eval src/evals/teleqna/teleqna.py --model openai/gpt-4o --limit 50
```

## Understanding Benchmarks

### What does each benchmark measure?

| Benchmark | What It Measures | Use Case |
|-----------|-----------------|----------|
| **TeleQnA** | General telecom knowledge | Baseline capability assessment |
| **TeleMath** | Mathematical reasoning in telecom | Signal processing, network optimization |
| **TeleLogs** | Root cause analysis | Network operations, diagnostics |
| **3GPP TSG** | Standards document understanding | Technical specification work |
| **TeleYAML** | Configuration generation | Network automation tasks |

### How do I interpret results?

- **Accuracy**: Percentage of correct answers (higher is better)
- **Epochs**: Multiple runs for statistical confidence (use `--epochs 3`)
- **Samples**: Number of test cases evaluated

Compare models using the same number of samples and epochs for fair comparison.

### What's a good accuracy score?

Scores vary by benchmark complexity:

| Benchmark | Typical Range | Notes |
|-----------|---------------|-------|
| TeleQnA | 40-70% | Multiple choice, tests breadth |
| TeleLogs | 30-60% | Requires reasoning about network state |
| TeleMath | 20-50% | Complex multi-step calculations |
| 3GPP TSG | 50-80% | Document classification |

## Running Evaluations

### How long do evaluations take?

Depends on the benchmark, model, and sample count:

| Benchmark | 50 samples | Full dataset |
|-----------|------------|--------------|
| TeleQnA | ~2-5 min | ~30-60 min |
| TeleLogs | ~5-10 min | ~1-2 hours |
| TeleMath | ~10-20 min | ~2-4 hours |

Use `--limit N` to run fewer samples for testing.

### How do I run multiple models at once?

```bash
uv run inspect eval-set src/evals/teleqna/teleqna.py \
   --model openai/gpt-4o,anthropic/claude-sonnet-4-20250514 \
   --log-dir logs/comparison
```

### Can I resume an interrupted evaluation?

Yes. Re-run the same command with the same `--log-dir` and it will pick up where it left off:

```bash
uv run inspect eval-set src/evals/teleqna/teleqna.py \
   --model openai/gpt-4o \
   --log-dir logs/my-run
```

## Troubleshooting

### "Rate limit exceeded" errors

Add retry options:

```bash
uv run inspect eval src/evals/teleqna/teleqna.py \
   --model openai/gpt-4o \
   --retry-attempts 5 \
   --retry-wait 60
```

### Evaluation is very slow

- Check your internet connection
- Use `--limit 10` for initial testing
- Consider using faster models (e.g., `gpt-4o-mini`)

### Results seem inconsistent

Run multiple epochs for more stable results:

```bash
uv run inspect eval src/evals/teleqna/teleqna.py \
   --model openai/gpt-4o \
   --epochs 3
```

## Getting Help

- **GitHub Issues**: [github.com/otelcos/evals/issues](https://github.com/otelcos/evals/issues)
- **Inspect AI Docs**: [inspect.aisi.org.uk](https://inspect.aisi.org.uk/)
- **Contact**: [emolero@gsma.com](mailto:emolero@gsma.com)
