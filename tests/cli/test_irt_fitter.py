"""Tests for IRT parameter fitting."""

from __future__ import annotations

import pytest

from open_telco.cli.services.irt_fitter import (
    BENCHMARKS,
    IRTParameters,
    fit_irt_parameters,
    sigmoid,
)
from open_telco.cli.services.tci_calculator import LeaderboardEntry


class TestSigmoid:
    """Test sigmoid function."""

    def test_sigmoid_zero(self) -> None:
        """Sigmoid of 0 should be 0.5."""
        assert sigmoid(0) == 0.5

    def test_sigmoid_large_positive(self) -> None:
        """Large positive values should approach 1."""
        assert sigmoid(10) > 0.99

    def test_sigmoid_large_negative(self) -> None:
        """Large negative values should approach 0."""
        assert sigmoid(-10) < 0.01

    def test_sigmoid_numerical_stability(self) -> None:
        """Should not overflow for extreme values."""
        # Should not raise or return inf/nan
        result_neg = sigmoid(-1000)
        result_pos = sigmoid(1000)

        assert 0 <= result_neg < 1
        assert 0 < result_pos <= 1


class TestIRTFitting:
    """Test IRT parameter fitting."""

    def test_fit_with_clear_capability_difference(self) -> None:
        """If Model A scores higher than Model B everywhere, A should have higher capability."""
        entries = [
            LeaderboardEntry(model="A", teleqna=90, telelogs=85, telemath=80, tsg=75),
            LeaderboardEntry(model="B", teleqna=30, telelogs=35, telemath=40, tsg=45),
        ]
        params = fit_irt_parameters(entries)

        # A should have higher capability than B
        assert params.capability["A"] > params.capability["B"]

    def test_fit_identifies_difficult_benchmark(self) -> None:
        """Benchmark with lower average scores should have higher difficulty."""
        entries = [
            LeaderboardEntry(model="A", teleqna=90, telelogs=30),
            LeaderboardEntry(model="B", teleqna=80, telelogs=20),
            LeaderboardEntry(model="C", teleqna=70, telelogs=10),
        ]
        params = fit_irt_parameters(entries)

        # telelogs is harder (lower scores) -> higher difficulty
        assert params.difficulty["telelogs"] > params.difficulty["teleqna"]

    def test_fit_handles_missing_scores(self) -> None:
        """Should handle entries with missing benchmark scores."""
        entries = [
            LeaderboardEntry(model="A", teleqna=90, telelogs=None),
            LeaderboardEntry(model="B", teleqna=None, telelogs=80),
        ]
        params = fit_irt_parameters(entries)

        assert len(params.capability) == 2
        assert len(params.difficulty) == len(BENCHMARKS)

    def test_fit_with_single_model(self) -> None:
        """Should handle edge case of single model."""
        entries = [
            LeaderboardEntry(model="A", teleqna=75, telelogs=65, telemath=70, tsg=60),
        ]
        params = fit_irt_parameters(entries)

        # Should return valid parameters (heavily regularized)
        assert "A" in params.capability
        assert all(b in params.difficulty for b in BENCHMARKS)

    def test_fit_empty_entries(self) -> None:
        """Should handle empty entry list gracefully."""
        params = fit_irt_parameters([])

        # Should return defaults
        assert params.n_models == 0
        assert params.fit_residual == float("inf")

    def test_fit_all_same_scores(self) -> None:
        """When all models score same, capabilities should be similar."""
        entries = [
            LeaderboardEntry(model="A", teleqna=50, telelogs=50, telemath=50, tsg=50),
            LeaderboardEntry(model="B", teleqna=50, telelogs=50, telemath=50, tsg=50),
        ]
        params = fit_irt_parameters(entries)

        # Capabilities should be very close
        assert abs(params.capability["A"] - params.capability["B"]) < 0.5

    def test_fit_no_valid_scores(self) -> None:
        """Should handle entries where all scores are None."""
        entries = [
            LeaderboardEntry(model="A"),
            LeaderboardEntry(model="B"),
        ]
        params = fit_irt_parameters(entries)

        # Should return defaults
        assert params.fit_residual == float("inf")


class TestIRTWithRealisticData:
    """Tests simulating real leaderboard data."""

    def test_eight_models_four_benchmarks(self) -> None:
        """Test with realistic sparse data (8 models, 4 benchmarks)."""
        entries = [
            LeaderboardEntry(
                model="gpt-4o", teleqna=85.0, telelogs=72.0, telemath=68.0, tsg=65.0
            ),
            LeaderboardEntry(
                model="claude-3.5", teleqna=82.0, telelogs=75.0, telemath=65.0, tsg=62.0
            ),
            LeaderboardEntry(
                model="gemini-pro", teleqna=78.0, telelogs=68.0, telemath=62.0, tsg=58.0
            ),
            LeaderboardEntry(
                model="llama-70b", teleqna=70.0, telelogs=55.0, telemath=52.0, tsg=48.0
            ),
            LeaderboardEntry(
                model="mixtral", teleqna=68.0, telelogs=52.0, telemath=50.0, tsg=45.0
            ),
            LeaderboardEntry(
                model="deepseek", teleqna=65.0, telelogs=48.0, telemath=45.0, tsg=42.0
            ),
            LeaderboardEntry(
                model="mistral-7b", teleqna=58.0, telelogs=40.0, telemath=38.0, tsg=35.0
            ),
            LeaderboardEntry(
                model="phi-2", teleqna=45.0, telelogs=30.0, telemath=28.0, tsg=25.0
            ),
        ]

        params = fit_irt_parameters(entries)

        # Verify ordering of capabilities matches general score ordering
        caps = [params.capability[e.model] for e in entries]
        assert caps == sorted(caps, reverse=True), "Capabilities should rank-order with scores"

        # teleqna should be easiest (highest avg score)
        assert params.difficulty["teleqna"] < params.difficulty["telelogs"]

    def test_irt_parameters_structure(self) -> None:
        """Verify IRTParameters has expected structure."""
        entries = [
            LeaderboardEntry(model="A", teleqna=80, telelogs=60),
            LeaderboardEntry(model="B", teleqna=70, telelogs=50),
        ]
        params = fit_irt_parameters(entries)

        assert isinstance(params, IRTParameters)
        assert isinstance(params.difficulty, dict)
        assert isinstance(params.slope, dict)
        assert isinstance(params.capability, dict)
        assert isinstance(params.fit_residual, float)
        assert params.n_models == 2
        assert params.n_benchmarks == len(BENCHMARKS)

    def test_slopes_are_positive(self) -> None:
        """All fitted slopes should be positive."""
        entries = [
            LeaderboardEntry(model="A", teleqna=90, telelogs=70, telemath=80, tsg=60),
            LeaderboardEntry(model="B", teleqna=50, telelogs=30, telemath=40, tsg=20),
        ]
        params = fit_irt_parameters(entries)

        for bench, slope in params.slope.items():
            assert slope > 0, f"Slope for {bench} should be positive, got {slope}"

    def test_fit_residual_decreases_with_better_fit(self) -> None:
        """Fit residual should be lower for data that fits IRT model well."""
        # Perfect IRT-like data: consistent ordering
        consistent_entries = [
            LeaderboardEntry(model="A", teleqna=90, telelogs=90, telemath=90, tsg=90),
            LeaderboardEntry(model="B", teleqna=50, telelogs=50, telemath=50, tsg=50),
        ]

        # Noisy data: inconsistent ordering
        noisy_entries = [
            LeaderboardEntry(model="A", teleqna=90, telelogs=30, telemath=80, tsg=20),
            LeaderboardEntry(model="B", teleqna=40, telelogs=70, telemath=20, tsg=80),
        ]

        consistent_params = fit_irt_parameters(consistent_entries)
        noisy_params = fit_irt_parameters(noisy_entries)

        # Consistent data should fit better (lower residual)
        assert consistent_params.fit_residual < noisy_params.fit_residual
