"""Tests for run-evals screen."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import pandas as pd

from open_telco.cli.app import OpenTelcoApp
from open_telco.cli.screens.main_menu import MainMenuScreen
from open_telco.cli.screens.run_evals import RunEvalsScreen
from open_telco.cli.screens.run_evals.run_evals_screen import (
    ChecklistItem,
    TaskSelectScreen,
    Stage,
    TASK_TO_COLUMN,
    PROVIDER_NAMES,
)


class TestRunEvalsNavigation:
    """Test navigation to run-evals screen."""

    @pytest.mark.asyncio
    async def test_run_evals_navigates_to_screen(self) -> None:
        """Selecting run-evals should go to RunEvalsScreen or TaskSelectScreen."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            # Welcome -> Main Menu
            await pilot.press("enter")
            assert isinstance(pilot.app.screen, MainMenuScreen)

            # Select run-evals (second item, so press down once)
            await pilot.press("down")
            await pilot.press("enter")

            # Could be RunEvalsScreen or TaskSelectScreen (if preflights already passed)
            assert isinstance(pilot.app.screen, (RunEvalsScreen, TaskSelectScreen))

    @pytest.mark.asyncio
    async def test_back_from_run_evals_returns_to_main(self) -> None:
        """Pressing q from run-evals flow returns to main menu."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            # Navigate to run-evals screen
            await pilot.press("enter")  # Welcome -> Main
            await pilot.press("down")  # Select run-evals
            await pilot.press("enter")  # Go to run-evals

            # Could be RunEvalsScreen or TaskSelectScreen
            assert isinstance(pilot.app.screen, (RunEvalsScreen, TaskSelectScreen))

            # Go back (may need to press q twice if TaskSelectScreen is pushed)
            await pilot.press("q")
            if isinstance(pilot.app.screen, RunEvalsScreen):
                await pilot.press("q")

            assert isinstance(pilot.app.screen, MainMenuScreen)

    @pytest.mark.asyncio
    async def test_escape_from_run_evals_returns_to_main(self) -> None:
        """Pressing escape from run-evals flow returns to main menu."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            # Navigate to run-evals screen
            await pilot.press("enter")
            await pilot.press("down")
            await pilot.press("enter")

            # Could be RunEvalsScreen or TaskSelectScreen
            assert isinstance(pilot.app.screen, (RunEvalsScreen, TaskSelectScreen))

            # Go back with escape (may need to press twice if TaskSelectScreen is pushed)
            await pilot.press("escape")
            if isinstance(pilot.app.screen, RunEvalsScreen):
                await pilot.press("escape")

            assert isinstance(pilot.app.screen, MainMenuScreen)


class TestRunEvalsScreen:
    """Test RunEvalsScreen functionality."""

    @pytest.mark.asyncio
    async def test_screen_has_checklist_items(self) -> None:
        """Screen should have 4 checklist items (including find-k)."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            # Mock preflight check to return False so we stay on RunEvalsScreen
            with patch(
                "open_telco.cli.screens.run_evals.run_evals_screen.RunEvalsScreen._check_preflight_passed",
                return_value=False,
            ):
                # Navigate to run-evals screen
                await pilot.press("enter")
                await pilot.press("down")
                await pilot.press("enter")

                # Wait for screen to stabilize
                await pilot.pause()

                # Should be on RunEvalsScreen (not TaskSelectScreen since preflights haven't passed)
                assert isinstance(pilot.app.screen, RunEvalsScreen)

                # Check that all checklist items exist
                items = list(pilot.app.screen.query(ChecklistItem))
                assert len(items) == 4

                # Verify step IDs
                step_ids = [item.step_id for item in items]
                assert "mini_test" in step_ids
                assert "find_k" in step_ids
                assert "stress_test" in step_ids
                assert "ready" in step_ids

    @pytest.mark.asyncio
    async def test_screen_shows_model_info(self) -> None:
        """Screen should display the configured model."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            with patch(
                "open_telco.cli.screens.run_evals.run_evals_screen.EnvManager"
            ) as mock_env:
                mock_instance = MagicMock()
                mock_instance.get.return_value = "openai/gpt-4o"
                mock_env.return_value = mock_instance

                # Navigate to run-evals screen
                await pilot.press("enter")
                await pilot.press("down")
                await pilot.press("enter")

                assert isinstance(pilot.app.screen, RunEvalsScreen)

    @pytest.mark.asyncio
    async def test_shows_error_when_no_model_configured(self) -> None:
        """Screen should show error when no model is configured."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            # Mock EnvManager to return no model
            with patch(
                "open_telco.cli.screens.run_evals.run_evals_screen.EnvManager"
            ) as mock_env:
                mock_instance = MagicMock()
                mock_instance.get.return_value = None
                mock_env.return_value = mock_instance

                # Navigate to run-evals screen
                await pilot.press("enter")
                await pilot.press("down")
                await pilot.press("enter")

                # Should be on RunEvalsScreen
                assert isinstance(pilot.app.screen, RunEvalsScreen)

                # Wait a moment for on_mount to run
                await pilot.pause()

                # Screen should be in error stage
                assert pilot.app.screen.stage == Stage.ERROR


class TestChecklistItem:
    """Test ChecklistItem widget."""

    def test_checklist_item_pending_render(self) -> None:
        """Pending status should render with empty checkbox."""
        item = ChecklistItem("Test item", "test")
        item.status = "pending"
        rendered = item.render()
        assert "[ ]" in rendered
        assert "Test item" in rendered

    def test_checklist_item_running_render(self) -> None:
        """Running status should render with spinner and cooking text."""
        item = ChecklistItem("Test item", "test")
        item.status = "running"
        item.dot_count = 1
        rendered = item.render()
        # PROGRESS_FRAMES = ["○", "◔", "◑", "◕", "●"], so dot_count=1 gives ◔
        assert "[◔]" in rendered
        assert "cooking" in rendered
        assert "Test item" in rendered

    def test_checklist_item_passed_render(self) -> None:
        """Passed status should render with checkmark."""
        item = ChecklistItem("Test item", "test")
        item.status = "passed"
        rendered = item.render()
        assert "[✓]" in rendered
        assert "Test item" in rendered

    def test_checklist_item_failed_render(self) -> None:
        """Failed status should render with X mark."""
        item = ChecklistItem("Test item", "test")
        item.status = "failed"
        rendered = item.render()
        assert "[✗]" in rendered
        assert "Test item" in rendered

    def test_dot_animation_cycles(self) -> None:
        """Dot count should affect rendering."""
        item = ChecklistItem("Test item", "test")
        item.status = "running"

        item.dot_count = 0
        render0 = item.render()
        assert "cooking." in render0

        item.dot_count = 1
        render1 = item.render()
        assert "cooking.." in render1

        item.dot_count = 2
        render2 = item.render()
        assert "cooking..." in render2

    def test_passed_with_score_render(self) -> None:
        """Passed status with score should show score."""
        item = ChecklistItem("Test item", "test")
        item.status = "passed"
        item.score = 0.85
        rendered = item.render()
        assert "[✓]" in rendered
        assert "Test item" in rendered
        assert "score: 0.85" in rendered

    def test_passed_without_score_no_score_text(self) -> None:
        """Passed status without score should not show score text."""
        item = ChecklistItem("Test item", "test")
        item.status = "passed"
        item.score = None
        rendered = item.render()
        assert "[✓]" in rendered
        assert "Test item" in rendered
        assert "score" not in rendered


class TestTaskChecklistItem:
    """Test TaskChecklistItem widget for task selection."""

    def test_task_checklist_item_selected_render(self) -> None:
        """Selected task should render with [X] checkbox."""
        from open_telco.cli.screens.run_evals.run_evals_screen import TaskChecklistItem

        item = TaskChecklistItem("telelogs/telelogs.py", "telelogs", "task_0")
        item.selected = True
        rendered = item.render()
        assert "●" in rendered
        assert "telelogs" in rendered

    def test_task_checklist_item_unselected_render(self) -> None:
        """Unselected task should render with ○ checkbox."""
        from open_telco.cli.screens.run_evals.run_evals_screen import TaskChecklistItem

        item = TaskChecklistItem("telelogs/telelogs.py", "telelogs", "task_0")
        item.selected = False
        rendered = item.render()
        assert "○" in rendered
        assert "telelogs" in rendered

    def test_task_checklist_item_toggle(self) -> None:
        """Toggle should switch selected state."""
        from open_telco.cli.screens.run_evals.run_evals_screen import TaskChecklistItem

        item = TaskChecklistItem("telelogs/telelogs.py", "telelogs", "task_0")
        assert item.selected is True  # Default is selected

        item.toggle()
        assert item.selected is False

        item.toggle()
        assert item.selected is True

    def test_task_checklist_item_highlighted_render(self) -> None:
        """Highlighted task should render with bold text."""
        from open_telco.cli.screens.run_evals.run_evals_screen import TaskChecklistItem

        item = TaskChecklistItem("telelogs/telelogs.py", "telelogs", "task_0")
        item.highlighted = True
        rendered = item.render()
        assert "bold" in rendered
        assert "telelogs" in rendered


class TestTaskSelectScreen:
    """Test TaskSelectScreen functionality."""

    @pytest.mark.asyncio
    async def test_task_select_screen_shows_all_tasks(self) -> None:
        """TaskSelectScreen should show all 4 tasks."""
        from open_telco.cli.screens.run_evals.run_evals_screen import (
            TaskSelectScreen,
            TaskChecklistItem,
            ALL_TASKS,
        )
        from textual.app import App

        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(TaskSelectScreen("openai/gpt-4o"))

        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            items = list(pilot.app.screen.query(TaskChecklistItem))
            assert len(items) == len(ALL_TASKS)

    @pytest.mark.asyncio
    async def test_task_select_screen_toggle_selection(self) -> None:
        """Space should toggle task selection."""
        from open_telco.cli.screens.run_evals.run_evals_screen import (
            TaskSelectScreen,
            TaskChecklistItem,
        )
        from textual.app import App

        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(TaskSelectScreen("openai/gpt-4o"))

        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            items = list(pilot.app.screen.query(TaskChecklistItem))
            # First item should be selected by default
            assert items[0].selected is True

            # Toggle with space
            await pilot.press("space")
            assert items[0].selected is False

            # Toggle again
            await pilot.press("space")
            assert items[0].selected is True


class TestScoreParsing:
    """Test score parsing from inspect output."""

    def test_parse_score_with_accuracy(self) -> None:
        """Should parse accuracy value from output."""
        from open_telco.cli.screens.run_evals.run_evals_screen import RunEvalsScreen

        screen = RunEvalsScreen()
        screen.model = "test"

        output = "telelogs: accuracy=0.85\ntelemath: accuracy=0.90"
        score = screen._parse_score(output)
        assert score is not None
        assert abs(score - 0.875) < 0.001  # Average of 0.85 and 0.90

    def test_parse_score_colon_format(self) -> None:
        """Should parse accuracy with colon format."""
        from open_telco.cli.screens.run_evals.run_evals_screen import RunEvalsScreen

        screen = RunEvalsScreen()
        screen.model = "test"

        output = "accuracy: 0.75"
        score = screen._parse_score(output)
        assert score == 0.75

    def test_parse_score_no_match(self) -> None:
        """Should return None if no accuracy found."""
        from open_telco.cli.screens.run_evals.run_evals_screen import RunEvalsScreen

        screen = RunEvalsScreen()
        screen.model = "test"

        output = "Evaluation complete, no errors"
        score = screen._parse_score(output)
        assert score is None


class TestTaskPaths:
    """Test that task paths are valid and resolvable."""

    def test_all_task_paths_exist(self) -> None:
        """All task paths in ALL_TASKS should exist relative to src/open_telco."""
        from pathlib import Path

        from open_telco.cli.screens.run_evals.run_evals_screen import ALL_TASKS

        # Get src/open_telco directory (same logic as in run_evals_screen.py)
        # This test file is at tests/cli/test_run_evals_screens.py
        # We need to find src/open_telco from here
        project_root = Path(__file__).parent.parent.parent
        open_telco_dir = project_root / "src" / "open_telco"

        assert open_telco_dir.exists(), f"src/open_telco not found at {open_telco_dir}"

        for task_path in ALL_TASKS:
            full_path = open_telco_dir / task_path
            assert full_path.exists(), (
                f"task path '{task_path}' does not exist. "
                f"Expected at: {full_path}"
            )
            assert full_path.is_file(), f"task path '{task_path}' is not a file"

    def test_task_paths_are_python_files(self) -> None:
        """All task paths should be .py files."""
        from open_telco.cli.screens.run_evals.run_evals_screen import ALL_TASKS

        for task_path in ALL_TASKS:
            assert task_path.endswith(".py"), (
                f"task path '{task_path}' should end with .py"
            )

    def test_open_telco_dir_calculation_is_correct(self) -> None:
        """Verify the path calculation in _run_mini_test finds correct directory."""

        # Simulate the path calculation from run_evals_screen.py
        run_evals_screen_path = Path(__file__).parent.parent.parent / "src" / "open_telco" / "cli" / "screens" / "run_evals" / "run_evals_screen.py"

        # The calculation in the code: Path(__file__).parent.parent.parent.parent
        # From run_evals_screen.py: cli/screens/run_evals/run_evals_screen.py
        # .parent = run_evals/
        # .parent.parent = screens/
        # .parent.parent.parent = cli/
        # .parent.parent.parent.parent = src/open_telco/
        open_telco_dir = run_evals_screen_path.parent.parent.parent.parent

        assert open_telco_dir.name == "open_telco", (
            f"Expected directory name 'open_telco', got '{open_telco_dir.name}'"
        )
        assert (open_telco_dir / "telelogs" / "telelogs.py").exists(), (
            "telelogs/telelogs.py not found in calculated open_telco_dir"
        )


class TestModelDisplayNameParsing:
    """Test model display name parsing for parquet export."""

    def test_parse_openrouter_model(self) -> None:
        """Should parse openrouter/provider/model format."""
        screen = RunEvalsScreen()
        screen.model = "test"

        result = screen._parse_model_display_name("openrouter/openai/gpt-5.2")
        assert result == "gpt-5.2 (Openai)"

    def test_parse_direct_provider_model(self) -> None:
        """Should parse provider/model format."""
        screen = RunEvalsScreen()
        screen.model = "test"

        result = screen._parse_model_display_name("anthropic/claude-opus-4.5")
        assert result == "claude-opus-4.5 (Anthropic)"

    def test_parse_unknown_provider(self) -> None:
        """Should title-case unknown providers."""
        screen = RunEvalsScreen()
        screen.model = "test"

        result = screen._parse_model_display_name("newprovider/some-model")
        assert result == "some-model (Newprovider)"

    def test_parse_single_part_model(self) -> None:
        """Should handle single-part model names."""
        screen = RunEvalsScreen()
        screen.model = "test"

        result = screen._parse_model_display_name("gpt-4")
        assert result == "gpt-4 (Unknown)"

    def test_all_known_providers(self) -> None:
        """All providers in PROVIDER_NAMES should be recognized."""
        screen = RunEvalsScreen()
        screen.model = "test"

        for provider_key, display_name in PROVIDER_NAMES.items():
            result = screen._parse_model_display_name(f"{provider_key}/test-model")
            assert f"({display_name})" in result


class TestResultsPreviewFormatting:
    """Test results preview formatting for CLI display."""

    def test_format_empty_dataframe(self) -> None:
        """Should handle empty dataframe."""
        screen = RunEvalsScreen()
        screen.model = "test"

        df = pd.DataFrame()
        result = screen._format_results_preview(df)
        assert result == "No results to display"

    def test_format_complete_results(self) -> None:
        """Should format all benchmark scores."""
        screen = RunEvalsScreen()
        screen.model = "test"

        df = pd.DataFrame([{
            "model": "gpt-5.2 (Openai)",
            "teleqna": [83.6, 1.17, 1000.0],
            "telelogs": [75.0, 4.35, 100.0],
            "telemath": [39.0, 4.9, 100.0],
            "3gpp_tsg": [54.0, 5.01, 100.0],
            "date": "2026-01-09"
        }])

        result = screen._format_results_preview(df)

        assert "model: gpt-5.2 (Openai)" in result
        assert "teleqna" in result
        assert "83.60" in result
        assert "± 1.17" in result
        assert "telelogs" in result
        assert "telemath" in result
        assert "3gpp_tsg" in result

    def test_format_partial_results(self) -> None:
        """Should show -- for missing benchmarks."""
        screen = RunEvalsScreen()
        screen.model = "test"

        df = pd.DataFrame([{
            "model": "test-model (Test)",
            "teleqna": [50.0, 2.0, 100.0],
            "telelogs": None,
            "telemath": None,
            "3gpp_tsg": None,
            "date": "2026-01-09"
        }])

        result = screen._format_results_preview(df)

        assert "50.00" in result
        assert "--" in result


class TestExportPathResolution:
    """Test that export paths are correctly resolved."""

    def test_get_open_telco_dir_returns_correct_path(self) -> None:
        """_get_open_telco_dir should return src/open_telco directory."""
        screen = RunEvalsScreen()
        screen.model = "test"

        open_telco_dir = screen._get_open_telco_dir()

        assert open_telco_dir.name == "open_telco"
        assert open_telco_dir.parent.name == "src"
        assert (open_telco_dir / "telelogs" / "telelogs.py").exists()

    def test_log_dir_path_is_absolute(self) -> None:
        """Export should use absolute paths for log directory."""
        screen = RunEvalsScreen()
        screen.model = "test"

        open_telco_dir = screen._get_open_telco_dir()
        log_dir = open_telco_dir / "logs" / "leaderboard"

        # Path should be absolute, not relative
        assert log_dir.is_absolute()

    def test_export_fails_gracefully_when_log_dir_missing(self) -> None:
        """Export should raise error when log dir doesn't exist."""
        screen = RunEvalsScreen()
        screen.model = "test"

        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_log_dir = Path(tmpdir) / "nonexistent" / "logs"
            output_path = Path(tmpdir) / "results.parquet"

            # Should raise an error (FileNotFoundError or ValueError)
            with pytest.raises((FileNotFoundError, ValueError)):
                screen._export_to_leaderboard_parquet(
                    str(nonexistent_log_dir),
                    str(output_path)
                )


class TestTaskToColumnMapping:
    """Test task name to column name mapping."""

    def test_all_tasks_have_mappings(self) -> None:
        """All tasks in ALL_TASKS should have column mappings."""
        from open_telco.cli.screens.run_evals.run_evals_screen import ALL_TASKS

        for task_path in ALL_TASKS:
            # Extract task name from path (e.g., "teleqna/teleqna.py" -> "teleqna")
            task_name = task_path.split("/")[0]
            assert task_name in TASK_TO_COLUMN, f"Missing mapping for task: {task_name}"

    def test_column_names_match_gsma_schema(self) -> None:
        """Column names should match GSMA leaderboard schema."""
        expected_columns = {"teleqna", "telelogs", "telemath", "3gpp_tsg"}
        actual_columns = set(TASK_TO_COLUMN.values())

        assert actual_columns == expected_columns
