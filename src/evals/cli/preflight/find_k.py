"""Find-K module for determining optimal epochs to reduce evaluation variance.

Based on Evan Miller's paper "Adding Error Bars to Evals: A Statistical Approach
to Language Model Evaluations" (https://arxiv.org/abs/2411.00640).

Formula: Var(K>1) = Var(K=1) × (1 + 2/K) / 3

Variance reduction by K:
    K=1: 0% (baseline)
    K=2: 33%
    K=3: 44%
    K=4: 50%
    K=5: 53%
"""

from __future__ import annotations

import itertools
import json
import re
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from evals.cli.utils.process import communicate_with_timeout, start_process

_project_root = Path(__file__).parent.parent.parent.parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

DEFAULT_EPOCHS = 5
DEFAULT_FIND_K_TIMEOUT = 600
DEFAULT_TARGET_REDUCTION = 50.0
DEFAULT_MAX_K = 5

DEFAULT_TASKS = (
    "telelogs/telelogs.py",
    "telemath/telemath.py",
    "teleqna/teleqna.py",
    "three_gpp/three_gpp.py",
)


@dataclass
class FindKResult:
    """Result of find-k optimization."""

    optimal_k: int
    variance_reduction_pct: float
    task_consistency: dict[str, list[bool]] = field(default_factory=dict)
    observed_variance: float = 0.0
    error: str | None = None


def calculate_theoretical_variance_reduction(k: int) -> float:
    """Calculate theoretical max variance reduction for given K (0-100%)."""
    if k <= 1:
        return 0.0
    return (1 - (1 + 2 / k) / 3) * 100


def calculate_variance_reduction(k: int, observed_inconsistency: float = 1.0) -> float:
    """Calculate model-specific variance reduction for given K.

    Scales theoretical reduction by observed inconsistency rate.
    """
    if k <= 1 or observed_inconsistency <= 0:
        return 0.0
    return calculate_theoretical_variance_reduction(k) * observed_inconsistency


def _calculate_observed_variance(task_consistency: dict[str, list[bool]]) -> float:
    if not task_consistency:
        return 0.0

    inconsistent = sum(
        1 for results in task_consistency.values() if results and len(set(results)) > 1
    )
    return inconsistent / len(task_consistency)


def find_optimal_k(
    task_consistency: dict[str, list[bool]],
    target_reduction: float = DEFAULT_TARGET_REDUCTION,
    max_k: int = DEFAULT_MAX_K,
) -> tuple[int, float, float]:
    """Find minimum K to achieve target variance reduction.

    Returns:
        (optimal_k, achieved_reduction_pct, observed_inconsistency)
    """
    observed = _calculate_observed_variance(task_consistency)

    if observed == 0.0:
        return 1, 0.0, 0.0

    for k in range(2, max_k + 1):
        reduction = calculate_variance_reduction(k, observed)
        if reduction >= target_reduction:
            return k, reduction, observed

    return max_k, calculate_variance_reduction(max_k, observed), observed


def _try_load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _extract_epoch_results_from_samples(samples: list[dict]) -> list[bool]:
    """Extract per-epoch correctness from samples array.

    Args:
        samples: List of sample dicts with epoch and scores

    Returns:
        List of booleans, one per epoch, indicating correctness (C=True, I=False)
    """
    epoch_results: dict[int, bool] = {}

    for sample in samples:
        epoch = sample.get("epoch", 0)
        if epoch == 0:
            continue

        scores = sample.get("scores", {})
        for score_data in scores.values():
            value = score_data.get("value", "")
            is_correct = value == "C"
            if epoch not in epoch_results:
                epoch_results[epoch] = is_correct
            break

    return [epoch_results[epoch] for epoch in sorted(epoch_results.keys())]


def _extract_epoch_results_from_legacy_scores(scores: list[dict]) -> list[bool]:
    """Extract correctness from legacy format (one file per epoch).

    Args:
        scores: List of score dicts with name and value

    Returns:
        List with single boolean if accuracy found, empty list otherwise
    """
    for score in scores:
        if score.get("name") == "accuracy":
            is_correct = score.get("value", 0) > 0
            return [is_correct]
    return []


def _process_epoch_data(
    data: dict,
    task_consistency: dict[str, list[bool]],
    model: str,
) -> None:
    """Extract per-epoch correctness from inspect eval JSON log.

    Inspect eval with --epochs N creates ONE JSON file per task containing:
    - results.scores[].metrics.accuracy.value: aggregated accuracy (not used)
    - samples[]: array of per-epoch sample results
        - sample["epoch"]: epoch number (1-N)
        - sample["scores"]["<scorer_name>"]["value"]: "C" (correct) or "I" (incorrect)
    """
    eval_info = data.get("eval", {})

    if eval_info.get("model", "") != model:
        return

    task_name = eval_info.get("task", "")
    if not task_name:
        return

    # Try samples array first (real inspect format)
    samples = data.get("samples", [])
    if samples:
        epoch_results = _extract_epoch_results_from_samples(samples)
        if epoch_results:
            task_consistency.setdefault(task_name, []).extend(epoch_results)
        return

    # Fallback to legacy format (one file per epoch with direct accuracy)
    scores = data.get("results", {}).get("scores", [])
    if not scores:
        return

    epoch_results = _extract_epoch_results_from_legacy_scores(scores)
    if epoch_results:
        task_consistency.setdefault(task_name, []).extend(epoch_results)


def _parse_epoch_results(log_dir: Path, model: str) -> dict[str, list[bool]]:
    task_consistency: dict[str, list[bool]] = {}

    if not log_dir.exists():
        return task_consistency

    for json_file in sorted(log_dir.glob("*.json")):
        data = _try_load_json(json_file)
        if data:
            _process_epoch_data(data, task_consistency, model)

    return task_consistency


def _parse_output_for_consistency(output: str) -> dict[str, list[bool]]:
    task_consistency: dict[str, list[bool]] = {}

    patterns = (
        r"(telelogs|telemath|teleqna|three_gpp).*?accuracy[=:\s]+([0-9.]+)",
        r"(telelogs|telemath|teleqna|3gpp_tsg).*?accuracy[=:\s]+([0-9.]+)",
    )

    matches = itertools.chain.from_iterable(
        re.findall(pattern, output, re.IGNORECASE) for pattern in patterns
    )

    for task_name, accuracy in matches:
        task_consistency.setdefault(task_name.lower(), []).append(float(accuracy) > 0)

    return task_consistency


def _create_fallback_result(error: str) -> FindKResult:
    return FindKResult(
        optimal_k=DEFAULT_MAX_K,
        variance_reduction_pct=calculate_variance_reduction(DEFAULT_MAX_K, 1.0),
        observed_variance=1.0,
        error=error,
    )


def _extract_last_error_line(output: str) -> str:
    """Extract the last non-empty line from output as error message.

    Args:
        output: Command output string

    Returns:
        Last non-empty line, or default error message if none found
    """
    default_error = "Find-K evaluation failed"

    if not output:
        return default_error

    lines = [line for line in output.strip().split("\n") if line.strip()]
    if not lines:
        return default_error

    return lines[-1]


def run_find_k(
    model: str,
    epochs: int = DEFAULT_EPOCHS,
    tasks: tuple[str, ...] | None = None,
    evals_dir: Path | None = None,
    timeout: int = DEFAULT_FIND_K_TIMEOUT,
) -> FindKResult:
    """Run find-k to determine optimal number of epochs.

    Runs mini evaluation with multiple epochs to measure model consistency,
    then calculates optimal K using variance formula.
    """
    tasks = tasks or DEFAULT_TASKS
    evals_dir = evals_dir or Path(__file__).parent.parent.parent

    log_dir = evals_dir / "logs" / "find_k"
    log_dir.mkdir(parents=True, exist_ok=True)

    for old_log in log_dir.glob("*.json"):
        with suppress(OSError):
            old_log.unlink()

    cmd = [
        "uv",
        "run",
        "inspect",
        "eval",
        *tasks,
        "--model",
        model,
        "--limit",
        "1",
        "--epochs",
        str(epochs),
        "--log-dir",
        "logs/find_k",
        "--log-format",
        "json",
    ]

    process = start_process(cmd, cwd=evals_dir)
    if not process:
        return _create_fallback_result("Failed to start find-k process")

    stdout, stderr, timed_out = communicate_with_timeout(process, timeout)

    if timed_out:
        return _create_fallback_result(f"Find-K timed out after {timeout} seconds")

    if process.returncode != 0:
        return _create_fallback_result(_extract_last_error_line(stderr or stdout))

    task_consistency = _parse_epoch_results(log_dir, model)
    if not task_consistency:
        task_consistency = _parse_output_for_consistency(stdout + stderr)

    optimal_k, variance_reduction, observed = find_optimal_k(task_consistency)

    return FindKResult(
        optimal_k=optimal_k,
        variance_reduction_pct=variance_reduction,
        task_consistency=task_consistency,
        observed_variance=observed,
    )


def run_find_k_sync(
    model: str,
    epochs: int = DEFAULT_EPOCHS,
    tasks: tuple[str, ...] | None = None,
    evals_dir: Path | None = None,
    timeout: int = DEFAULT_FIND_K_TIMEOUT,
) -> FindKResult:
    """Synchronous wrapper for run_find_k (for threaded workers)."""
    return run_find_k(
        model=model,
        epochs=epochs,
        tasks=tasks,
        evals_dir=evals_dir,
        timeout=timeout,
    )
