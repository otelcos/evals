"""IRT parameter fitting for TCI calculation.

Implements 2-Parameter Logistic (2PL) Item Response Theory model fitting
using non-linear least squares with ridge regularization, similar to
Epoch AI's ECI methodology.

The key insight is that benchmark difficulty should be DYNAMIC - as models
improve and more pass a benchmark with higher scores, that benchmark
automatically becomes "easier" in the ranking system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from open_telco.cli.services.tci_calculator import LeaderboardEntry

# Benchmark identifiers (must match LeaderboardEntry attributes)
BENCHMARKS = ["teleqna", "telelogs", "telemath", "tsg"]

# Default regularization strengths (tuned for sparse data)
DEFAULT_LAMBDA_D = 0.1  # Difficulty regularization
DEFAULT_LAMBDA_ALPHA = 0.5  # Slope regularization toward 1.0
DEFAULT_LAMBDA_C = 0.1  # Capability regularization


@dataclass
class IRTParameters:
    """Fitted IRT parameters from the 2PL model.

    These parameters are dynamically fitted from leaderboard data,
    ensuring benchmark difficulty adapts as models improve.
    """

    difficulty: dict[str, float]  # D_b for each benchmark
    slope: dict[str, float]  # alpha_b for each benchmark
    capability: dict[str, float]  # C_m for each model
    fit_residual: float  # Final loss value
    n_models: int
    n_benchmarks: int


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:  # type: ignore[type-arg]
    """Numerically stable sigmoid function.

    Args:
        x: Input value or array

    Returns:
        Sigmoid of input (float or ndarray)
    """
    # Use numpy's clip and exp for numerical stability
    x_clipped = np.clip(x, -500, 500)
    result: float | np.ndarray = 1.0 / (1.0 + np.exp(-x_clipped))  # type: ignore[type-arg]
    return result


def fit_irt_parameters(
    entries: list[LeaderboardEntry],
    lambda_d: float = DEFAULT_LAMBDA_D,
    lambda_alpha: float = DEFAULT_LAMBDA_ALPHA,
    lambda_c: float = DEFAULT_LAMBDA_C,
    max_iter: int = 1000,
) -> IRTParameters:
    """Fit IRT parameters from leaderboard entries.

    Uses 2PL IRT model: P(m,b) = sigmoid(alpha_b * (C_m - D_b))

    The optimization minimizes:
        Loss = sum((observed - predicted)^2)
               + lambda_d * sum(D_b^2)
               + lambda_alpha * sum((alpha_b - 1)^2)
               + lambda_c * sum(C_m^2)

    Args:
        entries: List of LeaderboardEntry with benchmark scores
        lambda_d: Ridge regularization for difficulty parameters
        lambda_alpha: Ridge regularization for slopes toward 1.0
        lambda_c: Ridge regularization for capability parameters
        max_iter: Maximum optimization iterations

    Returns:
        IRTParameters with fitted difficulty, slope, and capability values
    """
    # Handle empty entries
    if not entries:
        return _default_parameters([])

    models = [e.model for e in entries]
    n_models = len(models)
    n_benchmarks = len(BENCHMARKS)

    # Build score matrix: (n_models, n_benchmarks), normalized to 0-1
    scores = np.zeros((n_models, n_benchmarks))
    mask = np.zeros((n_models, n_benchmarks), dtype=bool)

    for i, entry in enumerate(entries):
        for j, bench in enumerate(BENCHMARKS):
            score = getattr(entry, bench, None)
            if score is not None:
                scores[i, j] = score / 100.0  # Normalize to 0-1
                mask[i, j] = True

    # Handle edge case: no valid scores
    if not mask.any():
        return _default_parameters(models)

    # Handle edge case: too few data points - increase regularization
    n_valid = int(mask.sum())
    n_params = n_models + 2 * n_benchmarks
    if n_valid < n_params:
        lambda_d *= 2.0
        lambda_alpha *= 2.0
        lambda_c *= 2.0

    # Parameter vector layout: [D_0..D_B, alpha_0..alpha_B, C_0..C_M]
    # Initial guesses: difficulties at 0, slopes at 1, capabilities at 0
    x0 = np.concatenate(
        [
            np.zeros(n_benchmarks),  # D_b: start at 0 (average difficulty)
            np.ones(n_benchmarks),  # alpha_b: start at 1
            np.zeros(n_models),  # C_m: start at 0 (average capability)
        ]
    )

    def objective(x: np.ndarray) -> float:  # type: ignore[type-arg]
        """Compute loss for IRT model fitting."""
        d_params = x[:n_benchmarks]
        alpha = x[n_benchmarks : 2 * n_benchmarks]
        c_params = x[2 * n_benchmarks :]

        # Prediction loss (only for observed entries)
        loss = 0.0
        for i in range(n_models):
            for j in range(n_benchmarks):
                if mask[i, j]:
                    pred = float(sigmoid(alpha[j] * (c_params[i] - d_params[j])))
                    loss += (scores[i, j] - pred) ** 2

        # Ridge regularization
        loss += lambda_d * float(np.sum(d_params**2))
        loss += lambda_alpha * float(np.sum((alpha - 1.0) ** 2))
        loss += lambda_c * float(np.sum(c_params**2))

        return loss

    # Bounds: alpha must be positive and bounded
    bounds = (
        [(None, None)] * n_benchmarks  # D: unbounded
        + [(0.1, 5.0)] * n_benchmarks  # alpha: positive, bounded
        + [(None, None)] * n_models  # C: unbounded
    )

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )

    # Extract fitted parameters
    d_fitted = result.x[:n_benchmarks]
    alpha_fitted = result.x[n_benchmarks : 2 * n_benchmarks]
    c_fitted = result.x[2 * n_benchmarks :]

    return IRTParameters(
        difficulty={bench: float(d_fitted[j]) for j, bench in enumerate(BENCHMARKS)},
        slope={bench: float(alpha_fitted[j]) for j, bench in enumerate(BENCHMARKS)},
        capability={model: float(c_fitted[i]) for i, model in enumerate(models)},
        fit_residual=float(result.fun),
        n_models=n_models,
        n_benchmarks=n_benchmarks,
    )


def _default_parameters(models: list[str]) -> IRTParameters:
    """Return sensible defaults when fitting fails or no data available.

    Args:
        models: List of model names

    Returns:
        IRTParameters with default values
    """
    return IRTParameters(
        difficulty={bench: 0.0 for bench in BENCHMARKS},
        slope={bench: 1.0 for bench in BENCHMARKS},
        capability={model: 0.0 for model in models},
        fit_residual=float("inf"),
        n_models=len(models),
        n_benchmarks=len(BENCHMARKS),
    )
