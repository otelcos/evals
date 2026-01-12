"""CLI services for data fetching and calculations."""

from open_telco.cli.services.huggingface_client import (
    fetch_and_transform_leaderboard,
    fetch_leaderboard,
    parse_model_provider,
)
from open_telco.cli.services.irt_fitter import (
    BENCHMARKS,
    IRTParameters,
    fit_irt_parameters,
)
from open_telco.cli.services.tci_calculator import (
    TCI_CONFIG,
    LeaderboardEntry,
    calculate_all_tci,
    calculate_error,
    calculate_tci,
    sort_by_tci,
)

__all__ = [
    # HuggingFace client
    "fetch_leaderboard",
    "fetch_and_transform_leaderboard",
    "parse_model_provider",
    # IRT fitting
    "BENCHMARKS",
    "IRTParameters",
    "fit_irt_parameters",
    # TCI calculation
    "LeaderboardEntry",
    "TCI_CONFIG",
    "calculate_tci",
    "calculate_all_tci",
    "calculate_error",
    "sort_by_tci",
]
