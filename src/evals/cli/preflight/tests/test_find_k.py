"""Tests for find-k functionality."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evals.cli.preflight.find_k import (
    FindKResult,
    _calculate_observed_variance,
    _extract_epoch_results_from_legacy_scores,
    _extract_epoch_results_from_samples,
    _extract_last_error_line,
    _parse_epoch_results,
    calculate_theoretical_variance_reduction,
    calculate_variance_reduction,
    find_optimal_k,
)


class TestTheoreticalVarianceReductionFormula:
    """Test the paper's theoretical variance reduction formula."""

    def test_k1_returns_zero_reduction(self) -> None:
        """K=1 should return 0% variance reduction (baseline)."""
        assert calculate_theoretical_variance_reduction(1) == 0.0

    def test_k2_returns_33_percent(self) -> None:
        """K=2 should return ~33% variance reduction."""
        reduction = calculate_theoretical_variance_reduction(2)
        assert abs(reduction - 33.33) < 0.5

    def test_k3_returns_44_percent(self) -> None:
        """K=3 should return ~44% variance reduction."""
        reduction = calculate_theoretical_variance_reduction(3)
        assert abs(reduction - 44.44) < 0.5

    def test_k4_returns_50_percent(self) -> None:
        """K=4 should return ~50% variance reduction."""
        reduction = calculate_theoretical_variance_reduction(4)
        assert abs(reduction - 50.0) < 0.5

    def test_k5_returns_53_percent(self) -> None:
        """K=5 should return ~53% variance reduction."""
        reduction = calculate_theoretical_variance_reduction(5)
        assert abs(reduction - 53.33) < 0.5

    def test_k6_returns_56_percent(self) -> None:
        """K=6 should return ~56% variance reduction."""
        reduction = calculate_theoretical_variance_reduction(6)
        assert abs(reduction - 55.56) < 0.5


class TestModelSpecificVarianceReduction:
    """Test model-specific variance reduction based on observed inconsistency."""

    def test_k1_always_returns_zero(self) -> None:
        """K=1 should return 0% regardless of inconsistency."""
        assert calculate_variance_reduction(1, 1.0) == 0.0
        assert calculate_variance_reduction(1, 0.5) == 0.0
        assert calculate_variance_reduction(1, 0.0) == 0.0

    def test_zero_inconsistency_returns_zero(self) -> None:
        """A perfectly consistent model gets 0% benefit from more epochs."""
        assert calculate_variance_reduction(5, 0.0) == 0.0
        assert calculate_variance_reduction(3, 0.0) == 0.0

    def test_full_inconsistency_returns_theoretical_max(self) -> None:
        """100% inconsistent model gets full theoretical reduction."""
        # K=4 with 100% inconsistency should give ~50%
        reduction = calculate_variance_reduction(4, 1.0)
        assert abs(reduction - 50.0) < 0.5

    def test_half_inconsistency_returns_half_reduction(self) -> None:
        """50% inconsistent model gets half the theoretical reduction."""
        # K=4 theoretical is ~50%, so 50% inconsistency gives ~25%
        reduction = calculate_variance_reduction(4, 0.5)
        assert abs(reduction - 25.0) < 0.5

    def test_quarter_inconsistency(self) -> None:
        """25% inconsistent model gets quarter the theoretical reduction."""
        # K=4 theoretical is ~50%, so 25% inconsistency gives ~12.5%
        reduction = calculate_variance_reduction(4, 0.25)
        assert abs(reduction - 12.5) < 0.5

    def test_k0_returns_zero(self) -> None:
        """K=0 should return 0% (edge case)."""
        assert calculate_variance_reduction(0, 1.0) == 0.0

    def test_negative_k_returns_zero(self) -> None:
        """Negative K should return 0% (edge case)."""
        assert calculate_variance_reduction(-1, 1.0) == 0.0


class TestFindOptimalK:
    """Test optimal K calculation based on consistency."""

    def test_perfect_consistency_optimal_k_is_1(self) -> None:
        """If model is consistent across all epochs, optimal K=1."""
        task_consistency = {
            "telelogs": [True, True, True, True, True],
            "telemath": [True, True, True, True, True],
            "teleqna": [True, True, True, True, True],
            "three_gpp": [True, True, True, True, True],
        }
        optimal_k, _, _ = find_optimal_k(task_consistency)
        assert optimal_k == 1

    def test_perfect_consistency_reduction_is_zero(self) -> None:
        """If model is consistent, variance reduction is 0%."""
        task_consistency = {
            "telelogs": [True, True, True, True, True],
            "telemath": [True, True, True, True, True],
            "teleqna": [True, True, True, True, True],
            "three_gpp": [True, True, True, True, True],
        }
        _, reduction, _ = find_optimal_k(task_consistency)
        assert reduction == 0.0

    def test_perfect_consistency_observed_variance_is_zero(self) -> None:
        """If model is consistent, observed variance is 0."""
        task_consistency = {
            "telelogs": [True, True, True, True, True],
            "telemath": [True, True, True, True, True],
            "teleqna": [True, True, True, True, True],
            "three_gpp": [True, True, True, True, True],
        }
        _, _, observed = find_optimal_k(task_consistency)
        assert observed == 0.0

    def test_all_false_consistency_optimal_k_is_1(self) -> None:
        """If model is consistently wrong, K=1 is still sufficient."""
        task_consistency = {
            "telelogs": [False, False, False, False, False],
            "telemath": [False, False, False, False, False],
            "teleqna": [False, False, False, False, False],
            "three_gpp": [False, False, False, False, False],
        }
        optimal_k, _, _ = find_optimal_k(task_consistency)
        assert optimal_k == 1

    def test_all_false_consistency_reduction_is_zero(self) -> None:
        """If model is consistently wrong, variance reduction is 0%."""
        task_consistency = {
            "telelogs": [False, False, False, False, False],
            "telemath": [False, False, False, False, False],
            "teleqna": [False, False, False, False, False],
            "three_gpp": [False, False, False, False, False],
        }
        _, reduction, _ = find_optimal_k(task_consistency)
        assert reduction == 0.0

    def test_all_false_consistency_observed_variance_is_zero(self) -> None:
        """If model is consistently wrong, observed variance is 0."""
        task_consistency = {
            "telelogs": [False, False, False, False, False],
            "telemath": [False, False, False, False, False],
            "teleqna": [False, False, False, False, False],
            "three_gpp": [False, False, False, False, False],
        }
        _, _, observed = find_optimal_k(task_consistency)
        assert observed == 0.0

    def test_inconsistent_results_optimal_k_is_at_least_2(self) -> None:
        """If model shows variance, higher K should be recommended."""
        task_consistency = {
            "telelogs": [True, False, True, False, True],  # Inconsistent
            "telemath": [True, True, True, True, True],
            "teleqna": [True, True, True, True, True],
            "three_gpp": [True, True, True, True, True],
        }
        optimal_k, _, _ = find_optimal_k(task_consistency, target_reduction=50.0)
        assert optimal_k >= 2

    def test_inconsistent_results_reduction_is_positive(self) -> None:
        """If model shows variance, reduction should be positive."""
        task_consistency = {
            "telelogs": [True, False, True, False, True],  # Inconsistent
            "telemath": [True, True, True, True, True],
            "teleqna": [True, True, True, True, True],
            "three_gpp": [True, True, True, True, True],
        }
        _, reduction, _ = find_optimal_k(task_consistency, target_reduction=50.0)
        assert reduction > 0

    def test_inconsistent_results_observed_variance_is_quarter(self) -> None:
        """If 1 out of 4 tasks is inconsistent, observed variance is 0.25."""
        task_consistency = {
            "telelogs": [True, False, True, False, True],  # Inconsistent
            "telemath": [True, True, True, True, True],
            "teleqna": [True, True, True, True, True],
            "three_gpp": [True, True, True, True, True],
        }
        _, _, observed = find_optimal_k(task_consistency, target_reduction=50.0)
        assert observed == 0.25

    def test_target_reduction_affects_k(self) -> None:
        """Higher target reduction should result in higher K."""
        task_consistency = {
            "telelogs": [True, False, True, False, True],
            "telemath": [True, False, True, True, True],
            "teleqna": [True, True, True, True, True],
            "three_gpp": [True, True, True, True, True],
        }

        k_low, _, _ = find_optimal_k(task_consistency, target_reduction=5.0)
        k_high, _, _ = find_optimal_k(task_consistency, target_reduction=25.0)

        assert k_high >= k_low

    def test_empty_consistency_optimal_k_is_1(self) -> None:
        """Empty consistency dict should return K=1 (safe default)."""
        optimal_k, _, _ = find_optimal_k({})
        assert optimal_k == 1

    def test_empty_consistency_reduction_is_zero(self) -> None:
        """Empty consistency dict should have 0% variance reduction."""
        _, reduction, _ = find_optimal_k({})
        assert reduction == 0.0

    def test_empty_consistency_observed_variance_is_zero(self) -> None:
        """Empty consistency dict should have 0 observed variance."""
        _, _, observed = find_optimal_k({})
        assert observed == 0.0

    def test_max_k_cap(self) -> None:
        """K should not exceed max_k parameter."""
        task_consistency = {
            "telelogs": [True, False, True, False, True],
        }
        optimal_k, _, _ = find_optimal_k(
            task_consistency, target_reduction=99.0, max_k=3
        )
        assert optimal_k <= 3

    def test_full_inconsistency_returns_model_specific_reduction(self) -> None:
        """100% inconsistent model should get theoretical max reduction."""
        task_consistency = {
            "telelogs": [True, False, True, False, True],
            "telemath": [True, False, True, False, True],
            "teleqna": [True, False, True, False, True],
            "three_gpp": [True, False, True, False, True],
        }
        optimal_k, reduction, observed = find_optimal_k(
            task_consistency, target_reduction=50.0
        )
        assert observed == 1.0  # 100% inconsistent
        # With 100% inconsistency, K=4 gives ~50% reduction
        assert optimal_k == 4
        assert abs(reduction - 50.0) < 1.0


class TestObservedVariance:
    """Test observed variance calculation."""

    def test_all_consistent_returns_zero(self) -> None:
        """Fully consistent results should have 0 variance."""
        task_consistency = {
            "telelogs": [True, True, True, True, True],
            "telemath": [False, False, False, False, False],
        }
        variance = _calculate_observed_variance(task_consistency)
        assert variance == 0.0

    def test_all_inconsistent_returns_one(self) -> None:
        """All tasks inconsistent should have variance of 1.0."""
        task_consistency = {
            "telelogs": [True, False, True, False, True],
            "telemath": [True, False, True, False, True],
        }
        variance = _calculate_observed_variance(task_consistency)
        assert variance == 1.0

    def test_half_inconsistent_returns_half(self) -> None:
        """Half consistent, half inconsistent should return 0.5."""
        task_consistency = {
            "telelogs": [True, True, True, True, True],  # Consistent
            "telemath": [True, False, True, False, True],  # Inconsistent
        }
        variance = _calculate_observed_variance(task_consistency)
        assert variance == 0.5

    def test_empty_returns_zero(self) -> None:
        """Empty consistency dict should return 0."""
        variance = _calculate_observed_variance({})
        assert variance == 0.0


class TestFindKResult:
    """Test FindKResult dataclass."""

    @pytest.fixture
    def default_result(self) -> FindKResult:
        """Create a result with only required fields."""
        return FindKResult(optimal_k=3, variance_reduction_pct=44.0)

    @pytest.fixture
    def full_result(self) -> FindKResult:
        """Create a result with all fields populated."""
        return FindKResult(
            optimal_k=2,
            variance_reduction_pct=33.33,
            task_consistency={"telelogs": [True, True, False]},
            observed_variance=0.25,
            error="Test error",
        )

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("optimal_k", 3),
            ("variance_reduction_pct", 44.0),
            ("task_consistency", {}),
            ("observed_variance", 0.0),
            ("error", None),
        ],
    )
    def test_default_values(
        self, default_result: FindKResult, field: str, expected: object
    ) -> None:
        """Test default values are set correctly."""
        assert getattr(default_result, field) == expected

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("optimal_k", 2),
            ("variance_reduction_pct", 33.33),
            ("task_consistency", {"telelogs": [True, True, False]}),
            ("observed_variance", 0.25),
            ("error", "Test error"),
        ],
    )
    def test_with_all_fields(
        self, full_result: FindKResult, field: str, expected: object
    ) -> None:
        """Test creating result with all fields."""
        assert getattr(full_result, field) == expected


class TestKSelectionScreen:
    """Test KSelectionScreen UI component."""

    @pytest.mark.asyncio
    async def test_k_selection_screen_renders(self) -> None:
        """KSelectionScreen should render with correct information."""
        from textual.app import App

        from evals.cli.screens.run_evals.run_evals_screen import KSelectionScreen

        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(
                    KSelectionScreen(
                        model="openai/gpt-4o",
                        optimal_k=3,
                        variance_reduction=44.0,
                        task_consistency={
                            "telelogs": [True, True, True, True, True],
                            "telemath": [True, False, True, True, True],
                        },
                    )
                )

        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(pilot.app.screen, KSelectionScreen)
            assert pilot.app.screen.optimal_k == 3
            assert pilot.app.screen.selected_k == 3

    @pytest.mark.asyncio
    async def test_k_selection_number_keys(self) -> None:
        """Number keys should change selected K."""
        from textual.app import App

        from evals.cli.screens.run_evals.run_evals_screen import KSelectionScreen

        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(
                    KSelectionScreen(
                        model="openai/gpt-4o",
                        optimal_k=3,
                        variance_reduction=44.0,
                        task_consistency={},
                    )
                )

        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert screen.selected_k == 3

            await pilot.press("5")
            assert screen.selected_k == 5

            await pilot.press("1")
            assert screen.selected_k == 1

            await pilot.press("2")
            assert screen.selected_k == 2


class TestChecklistItemFindK:
    """Test ChecklistItem display for find-k step."""

    def test_find_k_checklist_render_with_variance(self) -> None:
        """Find-k step should show K and variance reduction."""
        from evals.cli.screens.run_evals.run_evals_screen import ChecklistItem

        item = ChecklistItem("find-k", "find_k")
        item.status = "passed"
        item.score = 3.0
        item.variance_reduction = 44.0

        rendered = item.render()
        assert "K=3" in rendered
        assert "44%" in rendered
        assert "variance reduction" in rendered

    def test_find_k_checklist_render_without_variance(self) -> None:
        """Find-k step without variance reduction should show regular score."""
        from evals.cli.screens.run_evals.run_evals_screen import ChecklistItem

        item = ChecklistItem("find-k", "find_k")
        item.status = "passed"
        item.score = 3.0
        item.variance_reduction = None

        rendered = item.render()
        assert "score: 3.00" in rendered


class TestRunEvalsScreenFindK:
    """Test RunEvalsScreen find-k integration."""

    def test_stage_enum_has_find_k(self) -> None:
        """Stage enum should include FIND_K."""
        from evals.cli.screens.run_evals.run_evals_screen import Stage

        assert hasattr(Stage, "FIND_K")
        assert Stage.FIND_K.value == "find_k"

    @pytest.mark.asyncio
    async def test_checklist_has_find_k_item(self) -> None:
        """RunEvalsScreen should have find-k in checklist."""
        from evals.cli.app import OpenTelcoApp
        from evals.cli.screens.run_evals import RunEvalsScreen
        from evals.cli.screens.run_evals.run_evals_screen import ChecklistItem

        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            # Mock preflight check to return False so we stay on RunEvalsScreen
            with patch(
                "evals.cli.screens.run_evals.run_evals_screen.RunEvalsScreen._check_preflight_passed",
                return_value=False,
            ):
                # Navigate to run-evals screen
                await pilot.press("enter")
                await pilot.press("down")
                await pilot.press("enter")

                await pilot.pause()

                if isinstance(pilot.app.screen, RunEvalsScreen):
                    items = list(pilot.app.screen.query(ChecklistItem))
                    step_ids = [item.step_id for item in items]
                    assert "find_k" in step_ids

    def test_run_evals_has_selected_k(self) -> None:
        """RunEvalsScreen should have _selected_k attribute."""
        from evals.cli.screens.run_evals.run_evals_screen import RunEvalsScreen

        screen = RunEvalsScreen()
        assert hasattr(screen, "_selected_k")
        assert screen._selected_k == 1  # Default value


class TestRunEvalsWithEpochs:
    """Test that full eval uses epochs parameter."""

    def test_run_full_eval_includes_epochs(self) -> None:
        """The full eval command should include --epochs flag."""
        import inspect

        from evals.cli.screens.run_evals.run_evals_screen import RunEvalsScreen

        screen = RunEvalsScreen()
        source = inspect.getsource(screen._run_full_eval.__wrapped__)

        assert "--epochs" in source
        assert "self._selected_k" in source


# Test data for synthetic K optimization tests
# Each tuple: (name, synthetic_data, expected_inconsistency, target_reduction, expected_k)
SYNTHETIC_K_TEST_CASES = [
    # User's exact example from requirements:
    # "blabla" → C, C, I, C (3/4) - varies
    # "sld" → C, C, I, I (2/4) - varies
    # "sdkdkd" → C,C,C,C (4/4) - consistent
    pytest.param(
        "user_example",
        {
            "blabla": [True, True, False, True],
            "sld": [True, True, False, False],
            "sdkdkd": [True, True, True, True],
        },
        2 / 3,  # 2 out of 3 questions vary
        50.0,
        5,  # Cannot achieve 50% with 66.7% inconsistency, returns max_k
        id="user_example_3questions_4epochs",
    ),
    # All consistent → K=1 (no benefit from more epochs)
    pytest.param(
        "all_consistent",
        {
            "q1": [True, True, True, True],
            "q2": [False, False, False, False],
            "q3": [True, True, True, True],
        },
        0.0,
        50.0,
        1,
        id="all_consistent_returns_k1",
    ),
    # 100% inconsistent → K=4 achieves exactly 50%
    pytest.param(
        "all_inconsistent",
        {
            "q1": [True, False, True, False],
            "q2": [False, True, False, True],
        },
        1.0,
        50.0,
        4,
        id="100pct_inconsistent_k4_achieves_50pct",
    ),
    # 50% inconsistent → max achievable is 26.67%, returns K=5
    pytest.param(
        "half_inconsistent",
        {
            "q1": [True, False, True, False],  # varies
            "q2": [True, True, True, True],  # consistent
        },
        0.5,
        50.0,
        5,  # Cannot achieve 50% with 50% inconsistency
        id="50pct_inconsistent_returns_max_k",
    ),
    # Single question that varies
    pytest.param(
        "single_question_varies",
        {"q1": [True, False, True, False]},
        1.0,
        50.0,
        4,
        id="single_varying_question_k4",
    ),
    # Low target (10%) with 100% inconsistency → K=2 (33.33% > 10%)
    pytest.param(
        "low_target",
        {"q1": [True, False, True, False]},
        1.0,
        10.0,
        2,
        id="low_target_10pct_returns_k2",
    ),
    # Target exactly at K=2 threshold (33.33%)
    pytest.param(
        "exact_k2_threshold",
        {"q1": [True, False, True, False]},
        1.0,
        33.33,
        2,
        id="exact_threshold_33pct_returns_k2",
    ),
    # Target exactly at K=3 threshold (44.44%)
    pytest.param(
        "exact_k3_threshold",
        {"q1": [True, False, True, False]},
        1.0,
        44.0,
        3,
        id="threshold_44pct_returns_k3",
    ),
    # Target exactly at K=4 threshold (50%)
    pytest.param(
        "exact_k4_threshold",
        {"q1": [True, False, True, False]},
        1.0,
        50.0,
        4,
        id="threshold_50pct_returns_k4",
    ),
    # 25% inconsistency (1 of 4 questions varies)
    pytest.param(
        "quarter_inconsistent",
        {
            "q1": [True, False, True, False],  # varies
            "q2": [True, True, True, True],
            "q3": [False, False, False, False],
            "q4": [True, True, True, True],
        },
        0.25,
        50.0,
        5,  # Max achievable is ~13.33%, cannot reach 50%
        id="25pct_inconsistent_returns_max_k",
    ),
    # 75% inconsistency (3 of 4 questions vary)
    pytest.param(
        "three_quarter_inconsistent",
        {
            "q1": [True, False, True, False],
            "q2": [False, True, False, True],
            "q3": [True, True, False, False],
            "q4": [True, True, True, True],  # consistent
        },
        0.75,
        50.0,
        5,  # Max achievable at K=5 is 53.33% × 0.75 = 40%
        id="75pct_inconsistent_returns_max_k",
    ),
    # 3/3 vary → inc=1.0 → K=4 (50% reduction achievable)
    pytest.param(
        "lucky_guesser",
        {
            "q1": [True, False, False, False, False],
            "q2": [True, False, False, False, False],
            "q3": [True, False, False, False, False],
        },
        1.0,
        50.0,
        4,
        id="lucky_guesser",
    ),
    # 2/3 vary → inc=0.667 → K=5 (max 35.56%)
    pytest.param(
        "slow_learner",
        {
            "q1": [False, False, True, True, True],
            "q2": [False, False, False, True, True],
            "q3": [True, True, True, True, True],
        },
        2 / 3,
        50.0,
        5,
        id="slow_learner",
    ),
    # 2/6 vary → inc=0.333 → K=5 (max 17.78%)
    pytest.param(
        "overconfident_model",
        {
            "q1": [True, True, True, True, True],
            "q2": [True, True, True, True, True],
            "q3": [True, True, True, True, True],
            "q4": [True, True, True, True, True],
            "q5": [True, False, True, False, True],
            "q6": [False, True, False, True, False],
        },
        2 / 6,
        50.0,
        5,
        id="overconfident_model",
    ),
    # 4/4 vary → inc=1.0 → K=4 (50% reduction achievable)
    pytest.param(
        "coin_flipper",
        {
            "q1": [True, False, True, False, True],
            "q2": [False, True, False, True, False],
            "q3": [True, True, False, False, True],
            "q4": [False, False, True, True, False],
        },
        1.0,
        50.0,
        4,
        id="coin_flipper",
    ),
    # 2/5 vary → inc=0.4 → K=5 (max 21.33%)
    pytest.param(
        "specialist",
        {
            "q1": [True, True, True, True, True],
            "q2": [True, True, True, True, True],
            "q3": [True, True, True, True, True],
            "q4": [True, False, False, True, False],
            "q5": [False, True, False, False, True],
        },
        2 / 5,
        50.0,
        5,
        id="specialist",
    ),
    # 3/3 vary → inc=1.0 → K=4 (50% reduction achievable)
    pytest.param(
        "fatigue_model",
        {
            "q1": [True, True, True, False, False],
            "q2": [True, True, False, False, False],
            "q3": [True, True, True, True, False],
        },
        1.0,
        50.0,
        4,
        id="fatigue_model",
    ),
    # 5/10 vary → inc=0.5 → K=5 (max 26.67%)
    pytest.param(
        "binary_split",
        {
            "q1": [True, True, True, True, True],
            "q2": [False, False, False, False, False],
            "q3": [True, True, True, True, True],
            "q4": [False, False, False, False, False],
            "q5": [True, True, True, True, True],
            "q6": [True, False, True, False, True],
            "q7": [False, True, False, True, False],
            "q8": [True, True, False, False, True],
            "q9": [False, False, True, True, False],
            "q10": [True, False, False, True, False],
        },
        0.5,
        50.0,
        5,
        id="binary_split",
    ),
    # 1/4 vary → inc=0.25 → K=5 (max 13.33%)
    pytest.param(
        "one_off_error",
        {
            "q1": [True, True, True, True, False],
            "q2": [True, True, True, True, True],
            "q3": [True, True, True, True, True],
            "q4": [False, False, False, False, False],
        },
        0.25,
        50.0,
        5,
        id="one_off_error",
    ),
    # 2/3 vary → inc=0.667 → K=5 (max 35.56%)
    pytest.param(
        "temperature_sensitive",
        {
            "q1": [True, True, False, True, True],
            "q2": [False, False, True, False, False],
            "q3": [True, True, True, True, True],
        },
        2 / 3,
        50.0,
        5,
        id="temperature_sensitive",
    ),
    # 1/1 vary → inc=1.0 → K=4 (50% reduction achievable)
    pytest.param(
        "edge_case_champion",
        {
            "q1": [True, True, True, True, False],
        },
        1.0,
        50.0,
        4,
        id="edge_case_champion",
    ),
]


class TestFindKSyntheticData:
    """End-to-end tests with synthetic data verifying optimal K calculation.

    These tests use the variance reduction formula from the Anthropic paper:
        Var(K>1) = Var(K=1) × (1 + 2/K) / 3
        Reduction = (1 - (1 + 2/K) / 3) × 100
        Model-specific reduction = theoretical_reduction × observed_inconsistency

    Reference: "Adding Error Bars to Evals" (Section 3.1 Resampling)
    """

    @staticmethod
    def calculate_ground_truth_reduction(k: int, inconsistency: float) -> float:
        """Calculate expected variance reduction using paper's exact formula.

        Args:
            k: Number of samples per question
            inconsistency: Proportion of questions that vary across samples (0.0-1.0)

        Returns:
            Expected variance reduction percentage
        """
        if k <= 1 or inconsistency <= 0:
            return 0.0
        theoretical = (1 - (1 + 2 / k) / 3) * 100
        return theoretical * inconsistency

    @staticmethod
    def calculate_ground_truth_k(
        inconsistency: float, target: float, max_k: int = 5
    ) -> int:
        """Calculate expected optimal K that achieves target reduction.

        Args:
            inconsistency: Proportion of questions that vary (0.0-1.0)
            target: Target variance reduction percentage
            max_k: Maximum K value to consider

        Returns:
            Minimum K that achieves target, or max_k if unachievable
        """
        if inconsistency == 0:
            return 1
        for k in range(2, max_k + 1):
            theoretical = (1 - (1 + 2 / k) / 3) * 100
            if theoretical * inconsistency >= target:
                return k
        return max_k

    # =========================================================================
    # Parametrized end-to-end tests with synthetic data
    # =========================================================================

    @pytest.mark.parametrize(
        "name,synthetic_data,expected_inconsistency,target_reduction,expected_k",
        SYNTHETIC_K_TEST_CASES,
    )
    def test_synthetic_data_observed_inconsistency(
        self,
        name: str,  # noqa: ARG002
        synthetic_data: dict[str, list[bool]],
        expected_inconsistency: float,
        target_reduction: float,
        expected_k: int,  # noqa: ARG002
    ) -> None:
        """Verify observed_inconsistency matches expected value."""
        _, _, observed_inconsistency = find_optimal_k(
            synthetic_data, target_reduction=target_reduction
        )
        assert abs(observed_inconsistency - expected_inconsistency) < 0.0001, (
            f"Inconsistency mismatch: got {observed_inconsistency}, "
            f"expected {expected_inconsistency}"
        )

    @pytest.mark.parametrize(
        "name,synthetic_data,expected_inconsistency,target_reduction,expected_k",
        SYNTHETIC_K_TEST_CASES,
    )
    def test_synthetic_data_optimal_k(
        self,
        name: str,  # noqa: ARG002
        synthetic_data: dict[str, list[bool]],
        expected_inconsistency: float,  # noqa: ARG002
        target_reduction: float,
        expected_k: int,
    ) -> None:
        """Verify optimal_k matches the mathematically expected K."""
        optimal_k, _, _ = find_optimal_k(
            synthetic_data, target_reduction=target_reduction
        )
        assert optimal_k == expected_k, (
            f"Optimal K mismatch: got {optimal_k}, expected {expected_k}"
        )

    @pytest.mark.parametrize(
        "name,synthetic_data,expected_inconsistency,target_reduction,expected_k",
        SYNTHETIC_K_TEST_CASES,
    )
    def test_synthetic_data_achieved_reduction(
        self,
        name: str,  # noqa: ARG002
        synthetic_data: dict[str, list[bool]],
        expected_inconsistency: float,  # noqa: ARG002
        target_reduction: float,
        expected_k: int,  # noqa: ARG002
    ) -> None:
        """Verify achieved_reduction matches ground truth formula."""
        optimal_k, achieved_reduction, observed_inconsistency = find_optimal_k(
            synthetic_data, target_reduction=target_reduction
        )
        expected_reduction = self.calculate_ground_truth_reduction(
            optimal_k, observed_inconsistency
        )
        assert abs(achieved_reduction - expected_reduction) < 0.01, (
            f"Reduction mismatch: got {achieved_reduction:.4f}, "
            f"expected {expected_reduction:.4f}"
        )

    # =========================================================================
    # Exact formula verification tests
    # =========================================================================

    @pytest.mark.parametrize(
        "k,inconsistency,expected_reduction",
        [
            # Theoretical max values (100% inconsistency)
            (1, 1.0, 0.0),
            (2, 1.0, 100 * (1 - (1 + 2 / 2) / 3)),  # 33.3333...
            (3, 1.0, 100 * (1 - (1 + 2 / 3) / 3)),  # 44.4444...
            (4, 1.0, 100 * (1 - (1 + 2 / 4) / 3)),  # 50.0
            (5, 1.0, 100 * (1 - (1 + 2 / 5) / 3)),  # 53.3333...
            (6, 1.0, 100 * (1 - (1 + 2 / 6) / 3)),  # 55.5555...
            # Scaled by inconsistency
            (2, 0.5, 100 * (1 - (1 + 2 / 2) / 3) * 0.5),  # 16.6667
            (4, 0.5, 100 * (1 - (1 + 2 / 4) / 3) * 0.5),  # 25.0
            (4, 2 / 3, 100 * (1 - (1 + 2 / 4) / 3) * (2 / 3)),  # 33.3333 (user example)
            (
                5,
                2 / 3,
                100 * (1 - (1 + 2 / 5) / 3) * (2 / 3),
            ),  # 35.5556 (user example max)
            # Edge cases
            (4, 0.25, 100 * (1 - (1 + 2 / 4) / 3) * 0.25),  # 12.5
            (4, 0.0, 0.0),  # Zero inconsistency = zero reduction
        ],
    )
    def test_exact_formula_values(
        self, k: int, inconsistency: float, expected_reduction: float
    ) -> None:
        """Verify calculate_variance_reduction matches exact formula values."""
        actual = calculate_variance_reduction(k, inconsistency)
        assert abs(actual - expected_reduction) < 0.0001, (
            f"Formula mismatch for K={k}, inconsistency={inconsistency}: "
            f"got {actual:.6f}, expected {expected_reduction:.6f}"
        )

    # =========================================================================
    # Edge case tests
    # =========================================================================

    def test_all_correct_no_variance(self) -> None:
        """All correct answers across all epochs = no variance = K=1."""
        synthetic_data = {
            "q1": [True, True, True, True, True],
            "q2": [True, True, True, True, True],
            "q3": [True, True, True, True, True],
        }
        optimal_k, reduction, observed = find_optimal_k(synthetic_data)
        assert optimal_k == 1
        assert reduction == 0.0
        assert observed == 0.0

    def test_all_incorrect_no_variance(self) -> None:
        """All incorrect answers across all epochs = no variance = K=1."""
        synthetic_data = {
            "q1": [False, False, False, False, False],
            "q2": [False, False, False, False, False],
            "q3": [False, False, False, False, False],
        }
        optimal_k, reduction, observed = find_optimal_k(synthetic_data)
        assert optimal_k == 1
        assert reduction == 0.0
        assert observed == 0.0

    def test_empty_data_returns_k1(self) -> None:
        """Empty dataset should return K=1 safely."""
        optimal_k, reduction, observed = find_optimal_k({})
        assert optimal_k == 1
        assert reduction == 0.0
        assert observed == 0.0

    def test_single_epoch_per_question(self) -> None:
        """Single sample per question = no variance measurable = K=1."""
        synthetic_data = {
            "q1": [True],
            "q2": [False],
            "q3": [True],
        }
        optimal_k, reduction, observed = find_optimal_k(synthetic_data)
        assert optimal_k == 1
        # With single epoch, there's no way to detect variance
        assert observed == 0.0

    def test_mixed_epoch_counts(self) -> None:
        """Questions with different epoch counts should still work."""
        synthetic_data = {
            "q1": [True, False, True],  # 3 epochs, varies
            "q2": [True, True, True, True, True],  # 5 epochs, consistent
            "q3": [False, True],  # 2 epochs, varies
        }
        optimal_k, reduction, observed = find_optimal_k(synthetic_data)
        # 2 out of 3 questions vary
        assert abs(observed - 2 / 3) < 0.0001
        assert optimal_k >= 2

    def test_alternating_pattern(self) -> None:
        """Alternating T/F pattern should be detected as inconsistent."""
        synthetic_data = {
            "q1": [True, False, True, False, True],
        }
        optimal_k, reduction, observed = find_optimal_k(
            synthetic_data, target_reduction=50.0
        )
        assert observed == 1.0  # 100% inconsistent
        assert optimal_k == 4  # K=4 achieves exactly 50%

    def test_max_k_parameter_respected(self) -> None:
        """max_k parameter should cap the returned K value."""
        synthetic_data = {
            "q1": [True, False, True, False],
        }
        # With 100% inconsistency, normally K=4 for 50% target
        # But with max_k=3, should return 3
        optimal_k, _, _ = find_optimal_k(synthetic_data, target_reduction=50.0, max_k=3)
        assert optimal_k == 3

    def test_ground_truth_k_calculation(self) -> None:
        """Verify ground truth K calculation helper matches implementation."""
        test_cases = [
            (1.0, 50.0, 5, 4),  # 100% inconsistency, 50% target → K=4
            (1.0, 33.33, 5, 2),  # 100% inconsistency, 33.33% target → K=2
            (0.5, 50.0, 5, 5),  # 50% inconsistency, 50% target → K=5 (unachievable)
            (0.0, 50.0, 5, 1),  # 0% inconsistency → K=1
            (2 / 3, 50.0, 5, 5),  # User's example → K=5 (unachievable)
        ]
        for inconsistency, target, max_k, expected_k in test_cases:
            result = self.calculate_ground_truth_k(inconsistency, target, max_k)
            assert result == expected_k, (
                f"Ground truth K mismatch: inconsistency={inconsistency}, "
                f"target={target}, max_k={max_k}, got {result}, expected {expected_k}"
            )


class TestParseEpochResults:
    """Test JSON log parsing for find-k."""

    @staticmethod
    def _create_log_file(
        log_dir: Path, model: str, task: str, accuracy: float, index: int
    ) -> None:
        """Helper to create a mock JSON log file."""
        log_data = {
            "eval": {"model": model, "task": task},
            "results": {"scores": [{"name": "accuracy", "value": accuracy}]},
        }
        log_file = log_dir / f"log_{index:03d}.json"
        log_file.write_text(json.dumps(log_data))

    @pytest.mark.parametrize(
        ("check", "expected"),
        [
            ("contains_task", True),
            ("task_results", [True, False, True, False, True]),
        ],
    )
    def test_parses_single_task_multiple_epochs(
        self, tmp_path: Path, check: str, expected: object
    ) -> None:
        """Parse 5 epochs of a single task with varying results."""
        # task1: [1.0, 0.0, 1.0, 0.0, 1.0] → [True, False, True, False, True]
        accuracies = [1.0, 0.0, 1.0, 0.0, 1.0]
        for i, acc in enumerate(accuracies):
            self._create_log_file(tmp_path, "test-model", "task1", acc, i)

        result = _parse_epoch_results(tmp_path, "test-model")

        if check == "contains_task":
            assert ("task1" in result) == expected
        else:
            assert result["task1"] == expected

    @pytest.mark.parametrize(
        ("task", "expected"),
        [
            ("task1", [True, True, True]),
            ("task2", [True, False, True]),
        ],
    )
    def test_parses_multiple_tasks(
        self, tmp_path: Path, task: str, expected: list[bool]
    ) -> None:
        """Parse multiple tasks with different consistency patterns."""
        # task1: consistent [1.0, 1.0, 1.0] → [True, True, True]
        # task2: varies [1.0, 0.0, 1.0] → [True, False, True]
        log_index = 0
        for acc in [1.0, 1.0, 1.0]:
            self._create_log_file(tmp_path, "test-model", "task1", acc, log_index)
            log_index += 1
        for acc in [1.0, 0.0, 1.0]:
            self._create_log_file(tmp_path, "test-model", "task2", acc, log_index)
            log_index += 1

        result = _parse_epoch_results(tmp_path, "test-model")

        assert result[task] == expected

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("model-a", [True, True]),
            ("model-b", [False]),
        ],
    )
    def test_filters_by_model(
        self, tmp_path: Path, model: str, expected: list[bool]
    ) -> None:
        """Only parse logs matching the requested model."""
        self._create_log_file(tmp_path, "model-a", "task1", 1.0, 0)
        self._create_log_file(tmp_path, "model-b", "task1", 0.0, 1)
        self._create_log_file(tmp_path, "model-a", "task1", 1.0, 2)

        result = _parse_epoch_results(tmp_path, model)

        assert result["task1"] == expected

    def test_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        """Empty log directory returns empty dict."""
        result = _parse_epoch_results(tmp_path, "any-model")
        assert result == {}

    def test_nonexistent_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        """Non-existent directory returns empty dict."""
        nonexistent = tmp_path / "does_not_exist"
        result = _parse_epoch_results(nonexistent, "any-model")
        assert result == {}

    def test_handles_malformed_json(self, tmp_path: Path) -> None:
        """Skip files with invalid JSON."""
        # Create valid log
        self._create_log_file(tmp_path, "test-model", "task1", 1.0, 0)
        # Create malformed JSON file
        malformed = tmp_path / "log_001.json"
        malformed.write_text("{ invalid json }")

        result = _parse_epoch_results(tmp_path, "test-model")

        assert result["task1"] == [True]

    def test_handles_missing_task(self, tmp_path: Path) -> None:
        """Skip logs without eval.task field."""
        log_data = {
            "eval": {"model": "test-model"},  # missing task
            "results": {"scores": [{"name": "accuracy", "value": 1.0}]},
        }
        (tmp_path / "log_000.json").write_text(json.dumps(log_data))

        result = _parse_epoch_results(tmp_path, "test-model")
        assert result == {}

    def test_handles_missing_scores(self, tmp_path: Path) -> None:
        """Skip logs without results.scores field."""
        log_data = {
            "eval": {"model": "test-model", "task": "task1"},
            "results": {},  # missing scores
        }
        (tmp_path / "log_000.json").write_text(json.dumps(log_data))

        result = _parse_epoch_results(tmp_path, "test-model")
        assert result == {}

    def test_handles_no_accuracy_score(self, tmp_path: Path) -> None:
        """Skip logs without accuracy in scores."""
        log_data = {
            "eval": {"model": "test-model", "task": "task1"},
            "results": {"scores": [{"name": "f1", "value": 0.9}]},  # not accuracy
        }
        (tmp_path / "log_000.json").write_text(json.dumps(log_data))

        result = _parse_epoch_results(tmp_path, "test-model")
        assert result == {}

    def test_accuracy_zero_is_incorrect(self, tmp_path: Path) -> None:
        """Accuracy of exactly 0.0 is considered incorrect."""
        self._create_log_file(tmp_path, "test-model", "task1", 0.0, 0)

        result = _parse_epoch_results(tmp_path, "test-model")
        assert result["task1"] == [False]

    @pytest.mark.parametrize(
        ("task", "accuracy"),
        [
            ("task1", 0.001),
            ("task2", 0.5),
            ("task3", 1.0),
        ],
    )
    def test_accuracy_positive_is_correct(
        self, tmp_path: Path, task: str, accuracy: float
    ) -> None:
        """Any positive accuracy is considered correct."""
        self._create_log_file(tmp_path, "test-model", task, accuracy, 0)

        result = _parse_epoch_results(tmp_path, "test-model")
        assert result[task] == [True]

    def test_e2e_json_to_optimal_k(self, tmp_path: Path) -> None:
        """Full pipeline: JSON logs → parse → find_optimal_k."""
        # Create logs for 3 tasks, 5 epochs:
        # task1: [1.0, 1.0, 0.0, 1.0, 1.0] → varies
        # task2: [1.0, 1.0, 1.0, 1.0, 1.0] → consistent
        # task3: [0.0, 1.0, 0.0, 1.0, 0.0] → varies
        log_index = 0
        for acc in [1.0, 1.0, 0.0, 1.0, 1.0]:
            self._create_log_file(tmp_path, "test-model", "task1", acc, log_index)
            log_index += 1
        for acc in [1.0, 1.0, 1.0, 1.0, 1.0]:
            self._create_log_file(tmp_path, "test-model", "task2", acc, log_index)
            log_index += 1
        for acc in [0.0, 1.0, 0.0, 1.0, 0.0]:
            self._create_log_file(tmp_path, "test-model", "task3", acc, log_index)
            log_index += 1

        # Parse logs
        task_consistency = _parse_epoch_results(tmp_path, "test-model")

        # Verify parsed correctly
        assert task_consistency["task1"] == [True, True, False, True, True]
        assert task_consistency["task2"] == [True, True, True, True, True]
        assert task_consistency["task3"] == [False, True, False, True, False]

        # Calculate optimal K
        # inc = 2/3 (task1 and task3 vary, task2 consistent)
        # For 50% target with inc=0.667, max achievable is 35.56% → K=5
        optimal_k, reduction, observed = find_optimal_k(task_consistency)

        assert abs(observed - 2 / 3) < 0.0001
        assert optimal_k == 5


class TestParseEpochDataRealFormat:
    """Test parsing of REAL inspect eval JSON format with samples array.

    This tests the actual JSON structure produced by `inspect eval --epochs N`,
    which creates ONE file per task containing all epochs in a samples array.

    Bug being fixed: The old code expected one file per epoch with
    `results.scores[].name == "accuracy"`, but real format has:
    - `samples[]` array with per-epoch results
    - `sample["scores"]["<scorer_name>"]["value"]` = "C" (correct) or "I" (incorrect)
    """

    @staticmethod
    def _create_real_format_log(
        log_dir: Path,
        model: str,
        task: str,
        scorer_name: str,
        epoch_results: list[str],  # ["C", "I", "C", ...] for each epoch
        index: int = 0,
    ) -> None:
        """Create a log file matching real inspect eval output format."""
        samples = [
            {
                "id": 1,
                "epoch": i + 1,
                "scores": {scorer_name: {"value": result}},
            }
            for i, result in enumerate(epoch_results)
        ]

        # Calculate aggregated accuracy for results section
        correct_count = sum(1 for r in epoch_results if r == "C")
        accuracy = correct_count / len(epoch_results) if epoch_results else 0.0

        log_data = {
            "eval": {"model": model, "task": task},
            "results": {
                "scores": [
                    {
                        "name": scorer_name,
                        "metrics": {
                            "accuracy": {"name": "accuracy", "value": accuracy}
                        },
                    }
                ]
            },
            "samples": samples,
        }
        log_file = log_dir / f"log_{index:03d}.json"
        log_file.write_text(json.dumps(log_data))

    def test_parses_samples_array_with_epochs(self, tmp_path: Path) -> None:
        """Parse real inspect JSON format with samples array containing epochs.

        This is the core test for the bug fix. Real inspect output has epochs
        in samples array, not separate files.
        """
        # Create log with 5 epochs: C, I, C, I, C (alternating)
        self._create_real_format_log(
            tmp_path,
            model="test-model",
            task="telelogs",
            scorer_name="telelogs_scorer",
            epoch_results=["C", "I", "C", "I", "C"],
        )

        result = _parse_epoch_results(tmp_path, "test-model")

        # Should extract per-epoch correctness: C=True, I=False
        assert "telelogs" in result
        assert result["telelogs"] == [True, False, True, False, True]

    def test_handles_different_scorer_names(self, tmp_path: Path) -> None:
        """Each task has a different scorer name (telelogs_scorer, telemath_scorer, etc.)."""
        # Create logs for two different tasks with different scorer names
        self._create_real_format_log(
            tmp_path,
            model="test-model",
            task="telelogs",
            scorer_name="telelogs_scorer",
            epoch_results=["C", "C", "C"],
            index=0,
        )
        self._create_real_format_log(
            tmp_path,
            model="test-model",
            task="telemath",
            scorer_name="telemath_scorer",
            epoch_results=["C", "I", "C"],
            index=1,
        )

        result = _parse_epoch_results(tmp_path, "test-model")

        assert result["telelogs"] == [True, True, True]
        assert result["telemath"] == [True, False, True]

    def test_e2e_real_json_to_nonzero_variance(self, tmp_path: Path) -> None:
        """Full pipeline: real JSON format -> parse -> find_optimal_k -> non-zero variance.

        This is the end-to-end test proving the bug is fixed.
        """
        # Create 4 task logs mimicking real find-k run
        # telelogs: varies (C, I, C, I, C) -> inconsistent
        # telemath: consistent (C, C, C, C, C)
        # teleqna: varies (I, C, I, C, I) -> inconsistent
        # three_gpp: consistent (C, C, C, C, C)
        self._create_real_format_log(
            tmp_path,
            "test-model",
            "telelogs",
            "telelogs_scorer",
            ["C", "I", "C", "I", "C"],
            index=0,
        )
        self._create_real_format_log(
            tmp_path,
            "test-model",
            "telemath",
            "telemath_scorer",
            ["C", "C", "C", "C", "C"],
            index=1,
        )
        self._create_real_format_log(
            tmp_path,
            "test-model",
            "teleqna",
            "teleqna_scorer",
            ["I", "C", "I", "C", "I"],
            index=2,
        )
        self._create_real_format_log(
            tmp_path,
            "test-model",
            "three_gpp",
            "three_gpp_scorer",
            ["C", "C", "C", "C", "C"],
            index=3,
        )

        # Parse and calculate
        task_consistency = _parse_epoch_results(tmp_path, "test-model")
        optimal_k, variance_reduction, observed = find_optimal_k(task_consistency)

        # 2 out of 4 tasks vary -> observed = 0.5
        assert abs(observed - 0.5) < 0.0001
        # With 50% inconsistency, variance reduction should be positive
        assert variance_reduction > 0
        # K should be > 1 since there is variance to reduce
        assert optimal_k > 1

    def test_all_consistent_real_format_returns_zero_variance(
        self, tmp_path: Path
    ) -> None:
        """When all tasks are consistent in real format, variance should be 0."""
        self._create_real_format_log(
            tmp_path,
            "test-model",
            "telelogs",
            "telelogs_scorer",
            ["C", "C", "C", "C", "C"],
            index=0,
        )
        self._create_real_format_log(
            tmp_path,
            "test-model",
            "telemath",
            "telemath_scorer",
            ["C", "C", "C", "C", "C"],
            index=1,
        )

        task_consistency = _parse_epoch_results(tmp_path, "test-model")
        optimal_k, variance_reduction, observed = find_optimal_k(task_consistency)

        assert observed == 0.0
        assert variance_reduction == 0.0
        assert optimal_k == 1

    def test_backwards_compatible_with_old_format(self, tmp_path: Path) -> None:
        """Old test format (one file per epoch with direct accuracy) should still work."""
        # This is the OLD format used in existing tests
        for i, accuracy in enumerate([1.0, 0.0, 1.0, 0.0, 1.0]):
            log_data = {
                "eval": {"model": "test-model", "task": "task1"},
                "results": {"scores": [{"name": "accuracy", "value": accuracy}]},
            }
            (tmp_path / f"log_{i:03d}.json").write_text(json.dumps(log_data))

        result = _parse_epoch_results(tmp_path, "test-model")

        # Should still parse correctly via fallback
        assert result["task1"] == [True, False, True, False, True]

    def test_filters_by_model_real_format(self, tmp_path: Path) -> None:
        """Only parse logs matching the requested model (real format)."""
        self._create_real_format_log(
            tmp_path,
            "model-a",
            "telelogs",
            "telelogs_scorer",
            ["C", "C"],
            index=0,
        )
        self._create_real_format_log(
            tmp_path,
            "model-b",
            "telelogs",
            "telelogs_scorer",
            ["I", "I"],
            index=1,
        )

        result_a = _parse_epoch_results(tmp_path, "model-a")
        result_b = _parse_epoch_results(tmp_path, "model-b")

        assert result_a["telelogs"] == [True, True]
        assert result_b["telelogs"] == [False, False]


class TestExtractEpochResultsFromSamples:
    """Test _extract_epoch_results_from_samples helper function."""

    def test_extracts_correct_incorrect_values(self) -> None:
        """Should convert C to True and I to False."""
        samples = [
            {"epoch": 1, "scores": {"scorer": {"value": "C"}}},
            {"epoch": 2, "scores": {"scorer": {"value": "I"}}},
            {"epoch": 3, "scores": {"scorer": {"value": "C"}}},
        ]
        result = _extract_epoch_results_from_samples(samples)
        assert result == [True, False, True]

    def test_returns_sorted_by_epoch(self) -> None:
        """Should return results sorted by epoch number."""
        samples = [
            {"epoch": 3, "scores": {"scorer": {"value": "C"}}},
            {"epoch": 1, "scores": {"scorer": {"value": "I"}}},
            {"epoch": 2, "scores": {"scorer": {"value": "C"}}},
        ]
        result = _extract_epoch_results_from_samples(samples)
        assert result == [False, True, True]  # epoch 1, 2, 3

    def test_skips_epoch_zero(self) -> None:
        """Should skip samples with epoch=0."""
        samples = [
            {"epoch": 0, "scores": {"scorer": {"value": "C"}}},
            {"epoch": 1, "scores": {"scorer": {"value": "I"}}},
        ]
        result = _extract_epoch_results_from_samples(samples)
        assert result == [False]

    def test_uses_first_sample_per_epoch(self) -> None:
        """Should use first sample's result when multiple samples per epoch."""
        samples = [
            {"epoch": 1, "scores": {"scorer": {"value": "C"}}},
            {"epoch": 1, "scores": {"scorer": {"value": "I"}}},  # ignored
        ]
        result = _extract_epoch_results_from_samples(samples)
        assert result == [True]

    def test_empty_samples_returns_empty_list(self) -> None:
        """Should return empty list for empty samples."""
        result = _extract_epoch_results_from_samples([])
        assert result == []

    def test_handles_missing_scores(self) -> None:
        """Should handle samples without scores dict."""
        samples = [
            {"epoch": 1},
            {"epoch": 2, "scores": {"scorer": {"value": "C"}}},
        ]
        result = _extract_epoch_results_from_samples(samples)
        assert result == [True]


class TestExtractEpochResultsFromLegacyScores:
    """Test _extract_epoch_results_from_legacy_scores helper function."""

    def test_extracts_accuracy_score(self) -> None:
        """Should extract accuracy score and convert to boolean."""
        scores = [{"name": "accuracy", "value": 1.0}]
        result = _extract_epoch_results_from_legacy_scores(scores)
        assert result == [True]

    def test_zero_accuracy_is_false(self) -> None:
        """Zero accuracy should return False."""
        scores = [{"name": "accuracy", "value": 0.0}]
        result = _extract_epoch_results_from_legacy_scores(scores)
        assert result == [False]

    def test_positive_accuracy_is_true(self) -> None:
        """Any positive accuracy should return True."""
        scores = [{"name": "accuracy", "value": 0.001}]
        result = _extract_epoch_results_from_legacy_scores(scores)
        assert result == [True]

    def test_ignores_non_accuracy_scores(self) -> None:
        """Should only look for accuracy score."""
        scores = [{"name": "f1", "value": 0.9}]
        result = _extract_epoch_results_from_legacy_scores(scores)
        assert result == []

    def test_empty_scores_returns_empty_list(self) -> None:
        """Should return empty list for empty scores."""
        result = _extract_epoch_results_from_legacy_scores([])
        assert result == []

    def test_returns_first_accuracy_found(self) -> None:
        """Should return first accuracy score if multiple exist."""
        scores = [
            {"name": "accuracy", "value": 1.0},
            {"name": "accuracy", "value": 0.0},
        ]
        result = _extract_epoch_results_from_legacy_scores(scores)
        assert result == [True]


class TestExtractLastErrorLine:
    """Test _extract_last_error_line helper function."""

    def test_returns_last_line(self) -> None:
        """Should return last non-empty line."""
        output = "line1\nline2\nlast line"
        result = _extract_last_error_line(output)
        assert result == "last line"

    def test_empty_output_returns_default(self) -> None:
        """Empty output should return default error message."""
        result = _extract_last_error_line("")
        assert result == "Find-K evaluation failed"

    def test_whitespace_only_returns_default(self) -> None:
        """Whitespace-only output should return default."""
        result = _extract_last_error_line("   \n\n   ")
        assert result == "Find-K evaluation failed"

    def test_skips_empty_lines(self) -> None:
        """Should skip empty lines and return last non-empty."""
        output = "line1\n\nlast line\n\n"
        result = _extract_last_error_line(output)
        assert result == "last line"

    def test_single_line(self) -> None:
        """Should handle single line input."""
        result = _extract_last_error_line("only line")
        assert result == "only line"
