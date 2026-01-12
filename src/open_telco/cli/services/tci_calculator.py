"""TCI (Telco Capability Index) calculation utilities.

Uses Item Response Theory (IRT) methodology for meaningful cross-model
comparisons. Unlike the previous static approach, benchmark difficulty
and discrimination parameters are dynamically fitted from leaderboard data.

This ensures that as models improve and benchmarks become "easier",
the TCI calculation automatically adapts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from open_telco.cli.services.irt_fitter import (
    BENCHMARKS,
    IRTParameters,
    fit_irt_parameters,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configuration for TCI scaling (not benchmark-specific)
TCI_BASE_ERRORS: dict[str, float] = {
    "teleqna": 1.5,
    "telelogs": 3.6,
    "telemath": 2.8,
    "tsg": 2.4,
    "tci": 1.8,
}
TCI_MIN_SCORES_REQUIRED: int = 3
TCI_BASE_SCORE: int = 115
TCI_SCALE_FACTOR: int = 20

# Keep TCI_CONFIG for backwards compatibility with external callers
TCI_CONFIG: dict[str, object] = {
    "base_errors": TCI_BASE_ERRORS,
    "min_scores_required": TCI_MIN_SCORES_REQUIRED,
    "base_score": TCI_BASE_SCORE,
    "scale_factor": TCI_SCALE_FACTOR,
}


@dataclass
class LeaderboardEntry:
    """Entry for leaderboard calculations."""

    model: str
    provider: str = ""
    teleqna: float | None = None
    teleqna_stderr: float | None = None
    telelogs: float | None = None
    telelogs_stderr: float | None = None
    telemath: float | None = None
    telemath_stderr: float | None = None
    tsg: float | None = None
    tsg_stderr: float | None = None
    tci: float | None = field(default=None, repr=False)
    is_user: bool = field(default=False, repr=False)


def calculate_tci(entry: LeaderboardEntry, irt_params: IRTParameters) -> float | None:
    """Calculate TCI score using fitted IRT parameters.

    The TCI score is calculated based on weighted performance across
    benchmarks, using dynamically fitted difficulty and slope parameters.

    Args:
        entry: Leaderboard entry with benchmark scores
        irt_params: Fitted IRT parameters from fit_irt_parameters()

    Returns:
        TCI score (typically 90-150 range) or None if insufficient data
    """
    scores: list[tuple[str, float]] = []
    for bench in BENCHMARKS:
        score = getattr(entry, bench, None)
        if score is not None:
            scores.append((bench, score / 100.0))

    if len(scores) < TCI_MIN_SCORES_REQUIRED:
        return None

    # Calculate weighted logit-transformed capability estimate
    total_weight = 0.0
    weighted_capability = 0.0

    for bench, observed in scores:
        d_b = irt_params.difficulty.get(bench, 0.0)
        alpha_b = irt_params.slope.get(bench, 1.0)

        # Clamp observed score to prevent log(0) or log(inf)
        observed = max(0.01, min(0.99, observed))

        # Logit transform
        logit_score = math.log(observed / (1 - observed))

        # Weight by discrimination (slope)
        weight = alpha_b
        weighted_capability += (logit_score + d_b) * weight
        total_weight += weight

    # Scale to TCI range (roughly 90-150)
    raw_capability = weighted_capability / total_weight
    tci = TCI_BASE_SCORE + raw_capability * TCI_SCALE_FACTOR

    return round(tci * 10) / 10


def calculate_all_tci(
    entries: list[LeaderboardEntry],
) -> tuple[list[LeaderboardEntry], IRTParameters]:
    """Fit IRT parameters and calculate TCI for all entries.

    This is the main entry point for TCI calculation. It:
    1. Fits IRT parameters (difficulty, slope, capability) from all entries
    2. Calculates TCI for each entry using the fitted parameters
    3. Returns entries with updated TCI values

    Args:
        entries: List of leaderboard entries

    Returns:
        Tuple of (entries with updated TCI, fitted IRT parameters)
    """
    # Fit IRT parameters from all entries
    irt_params = fit_irt_parameters(entries)

    # Calculate TCI for each entry using fitted parameters
    for entry in entries:
        entry.tci = calculate_tci(entry, irt_params)

    return entries, irt_params


def calculate_error(score: float, benchmark_key: str) -> float:
    """Calculate synthetic error based on score and benchmark difficulty.

    Higher scores have lower error, lower scores have higher error.
    Each benchmark has a different base error reflecting measurement uncertainty.

    Args:
        score: The benchmark or TCI score
        benchmark_key: The benchmark identifier (e.g., 'teleqna', 'tci')

    Returns:
        Error margin value
    """
    base_error = TCI_BASE_ERRORS.get(benchmark_key, 2.0)
    return round((base_error * (1 + (100 - score) / 200)) * 100) / 100


def sort_by_tci(entries: Sequence[LeaderboardEntry]) -> list[LeaderboardEntry]:
    """Sort entries by TCI score (descending, nulls last).

    Args:
        entries: List of leaderboard entries

    Returns:
        Sorted list of entries
    """

    def sort_key(entry: LeaderboardEntry) -> tuple[int, float]:
        if entry.tci is None:
            return (1, 0.0)  # Nulls last
        return (0, -entry.tci)  # Descending

    return sorted(entries, key=sort_key)
