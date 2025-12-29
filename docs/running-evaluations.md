# Running Evaluations

## Single Evaluation

Run individual evaluations using the `inspect eval` command. See the [full list of available evaluations](eval-list.md).

```bash
uv run inspect eval <eval_path> --model <model>
```

## Common Options

```bash
# Limit samples for testing
uv run inspect eval src/open_telco/telemath/telemath.py --model openai/gpt-4o --limit 10

# Run multiple epochs (for majority voting metrics)
uv run inspect eval src/open_telco/telelogs/telelogs.py --model openai/gpt-4o -T 4

# View logs in browser
uv run inspect view
```

## Running Eval Sets

For running multiple evaluations across multiple models, use `eval_set`. This is the recommended approach for benchmarking.

### Command Line

```bash
uv run inspect eval-set src/open_telco/teleqna/teleqna.py src/open_telco/telemath/telemath.py \
   --model openai/gpt-4o,anthropic/claude-sonnet-4-20250514 \
   --log-dir logs/benchmark-run-1
```

### Python Script

Use the provided [run_evals.py](../src/open_telco/run_evals.py) script:

```python
from inspect_ai import eval_set

success, logs = eval_set(
   tasks=[
      "telelogs/telelogs.py",
      "three_gpp/three_gpp.py",
      "teleqna/teleqna.py",
      "telemath/telemath.py"
   ],
   model=[
      "openai/gpt-4o",
      "anthropic/claude-sonnet-4-20250514"
   ],
   log_dir="logs/benchmark-run-1",
   limit=None,  # Remove to run all samples
   epochs=1,
)
```

Run it with:

```bash
uv run python src/open_telco/run_evals.py
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `tasks` | List of evaluation files to run |
| `model` | List of models to evaluate (runs in parallel) |
| `log_dir` | **Required.** Directory for logs and tracking progress |
| `limit` | Number of samples to run (useful for testing) |
| `epochs` | Number of times to run each sample (for majority voting) |
| `max_tasks` | Maximum concurrent tasks (default: max of 4 or number of models) |

## Re-Running and Recovery

Eval sets that don't complete due to errors or cancellation can be re-run. Simply re-execute the same command and any work not yet completed will be scheduled.

```bash
# Initial run (interrupted)
uv run inspect eval-set src/open_telco/teleqna/teleqna.py \
   --model openai/gpt-4o \
   --log-dir logs/benchmark-run-1

# Re-run to complete remaining work
uv run inspect eval-set src/open_telco/teleqna/teleqna.py \
   --model openai/gpt-4o \
   --log-dir logs/benchmark-run-1
```

You can also extend an eval set with additional models or epochs:

```bash
# Add another model to the same run
uv run inspect eval-set src/open_telco/teleqna/teleqna.py \
   --model openai/gpt-4o,openai/gpt-4-turbo \
   --epochs 3 \
   --log-dir logs/benchmark-run-1
```

## Retry Options

| Option | Description | Default |
|--------|-------------|---------|
| `--retry-attempts` | Maximum retry attempts | 10 |
| `--retry-wait` | Base wait time between retries (exponential backoff) | 30s |
| `--retry-connections` | Reduce max connections rate per retry | 0.5 |

Example with custom retry settings:

```bash
uv run inspect eval-set src/open_telco/teleqna/teleqna.py \
   --model openai/gpt-4o \
   --log-dir logs/benchmark-run-1 \
   --retry-wait 120 \
   --retry-attempts 5
```

## Viewing Results

```bash
# Launch the log viewer
uv run inspect view

# View logs from a specific directory
uv run inspect view logs/benchmark-run-1
```

## IDE Integration

Use the [Inspect VS Code Extension](https://marketplace.visualstudio.com/items?itemName=inspect-ai.inspect) for an integrated development experience with:

- Run evaluations directly from the editor
- View results inline
- Debug evaluation code

## More Information

For detailed documentation on eval sets, see the [Inspect AI documentation](https://inspect.aisi.org.uk/eval-sets.html).
