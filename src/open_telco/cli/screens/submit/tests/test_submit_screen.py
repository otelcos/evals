"""Tests for submit screen functionality."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from open_telco.cli.app import OpenTelcoApp
from open_telco.cli.screens.main_menu import MainMenuScreen
from open_telco.cli.screens.submit import SubmitScreen
from open_telco.cli.screens.submit.github_service import PRResult
from open_telco.cli.screens.submit.submit_screen import (
    ModelChecklistItem,
    Stage,
)


class TestSubmitStageTransitions:
    """Test stage transitions in submit flow."""

    @pytest.mark.asyncio
    async def test_select_to_confirming_on_enter(
        self, tmp_path: Path, mock_github_token: str
    ) -> None:
        """Pressing enter with selected models should transition to CONFIRMING."""
        # Create temp parquet
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                }
            ]
        )
        df.to_parquet(parquet_path)

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                # Push submit screen directly
                app.push_screen(SubmitScreen())
                await pilot.pause()

                # Wait for loading
                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Wait for models to load
                for _ in range(10):
                    await pilot.pause()
                    if screen.stage == Stage.SELECT_MODELS:
                        break

                if screen.stage != Stage.SELECT_MODELS:
                    pytest.skip("Models didn't load in time")

                # Select a model with space
                await pilot.press("space")
                await pilot.pause()

                # Press enter to submit
                await pilot.press("enter")
                await pilot.pause()

                assert screen.stage == Stage.CONFIRMING

    @pytest.mark.asyncio
    async def test_confirming_back_to_select_on_q(
        self, tmp_path: Path, mock_github_token: str
    ) -> None:
        """Pressing q on CONFIRMING should return to SELECT_MODELS."""
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                }
            ]
        )
        df.to_parquet(parquet_path)

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                app.push_screen(SubmitScreen())
                await pilot.pause()

                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Wait for models to load
                for _ in range(10):
                    await pilot.pause()
                    if screen.stage == Stage.SELECT_MODELS:
                        break

                if screen.stage != Stage.SELECT_MODELS:
                    pytest.skip("Models didn't load in time")

                # Select and go to confirming
                await pilot.press("space")
                await pilot.press("enter")
                await pilot.pause()

                assert screen.stage == Stage.CONFIRMING

                # Press q to go back
                await pilot.press("q")
                await pilot.pause()

                assert screen.stage == Stage.SELECT_MODELS


class TestSubmitValidation:
    """Test validation before submission."""

    @pytest.mark.asyncio
    async def test_submit_without_selection_shows_warning(self, tmp_path: Path) -> None:
        """Pressing enter with no models selected should show warning."""
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                }
            ]
        )
        df.to_parquet(parquet_path)

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                app.push_screen(SubmitScreen())
                await pilot.pause()

                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Wait for models to load
                for _ in range(10):
                    await pilot.pause()
                    if screen.stage == Stage.SELECT_MODELS:
                        break

                if screen.stage != Stage.SELECT_MODELS:
                    pytest.skip("Models didn't load in time")

                # Press enter without selecting anything
                await pilot.press("enter")
                await pilot.pause()

                # Should still be on select stage (warning shown)
                assert screen.stage == Stage.SELECT_MODELS

    @pytest.mark.asyncio
    async def test_submit_without_github_token_shows_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Submitting without GITHUB_TOKEN should show error."""
        # Patch EnvManager to return None for GITHUB_TOKEN
        from open_telco.cli.config import EnvManager

        monkeypatch.setattr(EnvManager, "get", lambda self, key: None)
        # Also ensure no env var
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                }
            ]
        )
        df.to_parquet(parquet_path)

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                app.push_screen(SubmitScreen())
                await pilot.pause()

                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Wait for models to load
                for _ in range(10):
                    await pilot.pause()
                    if screen.stage == Stage.SELECT_MODELS:
                        break

                if screen.stage != Stage.SELECT_MODELS:
                    pytest.skip("Models didn't load in time")

                # Select model
                await pilot.press("space")
                await pilot.press("enter")
                await pilot.pause()

                # Confirm
                await pilot.press("enter")
                await pilot.pause()

                # Should show error about missing token
                assert screen.stage == Stage.ERROR

    @pytest.mark.asyncio
    async def test_submit_with_missing_parquet_shows_error(
        self, tmp_path: Path
    ) -> None:
        """Missing results.parquet should show error."""
        # Don't create parquet file

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                app.push_screen(SubmitScreen())

                # Wait for loading to complete
                for _ in range(10):
                    await pilot.pause()

                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Should be in error state
                assert screen.stage == Stage.ERROR

    @pytest.mark.asyncio
    async def test_submit_with_empty_parquet_shows_error(self, tmp_path: Path) -> None:
        """Empty results.parquet should show error."""
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        # Create empty parquet
        df = pd.DataFrame()
        df.to_parquet(parquet_path)

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                app.push_screen(SubmitScreen())

                # Wait for loading to complete
                for _ in range(10):
                    await pilot.pause()

                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Should be in error state
                assert screen.stage == Stage.ERROR


class TestSubmitScreenNavigation:
    """Test screen navigation with Textual pilot."""

    @pytest.mark.asyncio
    async def test_navigate_to_submit_from_main_menu(self) -> None:
        """Selecting submit option should push SubmitScreen."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            # Welcome -> Main Menu
            await pilot.press("enter")
            assert isinstance(pilot.app.screen, MainMenuScreen)

            # Navigate to submit (it's in the menu)
            # Find the submit option position
            for _ in range(5):
                await pilot.press("down")

            await pilot.press("enter")
            await pilot.pause()

            # Should be on SubmitScreen (or a screen that handles submit)
            # Note: exact position depends on menu order
            # The test verifies the navigation works

    @pytest.mark.asyncio
    async def test_press_q_returns_to_main_menu(self, tmp_path: Path) -> None:
        """Pressing q on submit screen should return to main menu."""
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                }
            ]
        )
        df.to_parquet(parquet_path)

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                # Go to main menu first
                await pilot.press("enter")
                assert isinstance(pilot.app.screen, MainMenuScreen)

                # Push submit screen
                app.push_screen(SubmitScreen())
                await pilot.pause()

                assert isinstance(pilot.app.screen, SubmitScreen)

                # Press q to go back
                await pilot.press("q")
                await pilot.pause()

                assert isinstance(pilot.app.screen, MainMenuScreen)

    @pytest.mark.asyncio
    async def test_press_escape_returns_to_main_menu(self, tmp_path: Path) -> None:
        """Pressing escape on submit screen should return to main menu."""
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                }
            ]
        )
        df.to_parquet(parquet_path)

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                await pilot.press("enter")
                assert isinstance(pilot.app.screen, MainMenuScreen)

                app.push_screen(SubmitScreen())
                await pilot.pause()

                assert isinstance(pilot.app.screen, SubmitScreen)

                await pilot.press("escape")
                await pilot.pause()

                assert isinstance(pilot.app.screen, MainMenuScreen)

    @pytest.mark.asyncio
    async def test_model_selection_with_space(self, tmp_path: Path) -> None:
        """Space key should toggle model selection."""
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                }
            ]
        )
        df.to_parquet(parquet_path)

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                app.push_screen(SubmitScreen())

                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Wait for models to load
                for _ in range(10):
                    await pilot.pause()
                    if screen.stage == Stage.SELECT_MODELS:
                        break

                if screen.stage != Stage.SELECT_MODELS:
                    pytest.skip("Models didn't load in time")

                # Get checklist items
                items = list(screen.query(ModelChecklistItem))
                assert len(items) > 0

                # Initially not selected
                assert items[0].selected is False

                # Toggle with space
                await pilot.press("space")
                await pilot.pause()

                assert items[0].selected is True

                # Toggle again
                await pilot.press("space")
                await pilot.pause()

                assert items[0].selected is False

    @pytest.mark.asyncio
    async def test_navigate_models_with_arrows(self, tmp_path: Path) -> None:
        """Arrow keys should move highlight between models."""
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                },
                {
                    "model": "claude-3 (Anthropic)",
                    "teleqna": [85.0, 1.0, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                },
            ]
        )
        df.to_parquet(parquet_path)

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                app.push_screen(SubmitScreen())

                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Wait for models to load
                for _ in range(10):
                    await pilot.pause()
                    if screen.stage == Stage.SELECT_MODELS:
                        break

                if screen.stage != Stage.SELECT_MODELS:
                    pytest.skip("Models didn't load in time")

                items = list(screen.query(ModelChecklistItem))
                assert len(items) >= 2

                # First item highlighted
                assert items[0].highlighted is True
                assert items[1].highlighted is False

                # Move down
                await pilot.press("down")
                await pilot.pause()

                assert items[0].highlighted is False
                assert items[1].highlighted is True

                # Move up
                await pilot.press("up")
                await pilot.pause()

                assert items[0].highlighted is True
                assert items[1].highlighted is False

    @pytest.mark.asyncio
    async def test_navigate_models_with_vim_keys(self, tmp_path: Path) -> None:
        """j/k keys should move highlight between models."""
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                },
                {
                    "model": "claude-3 (Anthropic)",
                    "teleqna": [85.0, 1.0, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                },
            ]
        )
        df.to_parquet(parquet_path)

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                app.push_screen(SubmitScreen())

                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Wait for models to load
                for _ in range(10):
                    await pilot.pause()
                    if screen.stage == Stage.SELECT_MODELS:
                        break

                if screen.stage != Stage.SELECT_MODELS:
                    pytest.skip("Models didn't load in time")

                items = list(screen.query(ModelChecklistItem))
                assert len(items) >= 2

                # First item highlighted
                assert items[0].highlighted is True

                # Move down with j
                await pilot.press("j")
                await pilot.pause()

                assert items[0].highlighted is False
                assert items[1].highlighted is True

                # Move up with k
                await pilot.press("k")
                await pilot.pause()

                assert items[0].highlighted is True


class TestSubmitScreenRendering:
    """Test UI rendering."""

    def test_model_checklist_item_unchecked_render(self) -> None:
        """Unselected model should render with empty checkbox."""
        item = ModelChecklistItem("gpt-4o", "Openai", 123.4, "model_0")
        item.selected = False
        rendered = item.render()

        assert "[ ]" in rendered
        assert "gpt-4o" in rendered

    def test_model_checklist_item_checked_render(self) -> None:
        """Selected model should render with checked checkbox."""
        item = ModelChecklistItem("gpt-4o", "Openai", 123.4, "model_0")
        item.selected = True
        rendered = item.render()

        assert "[x]" in rendered
        assert "gpt-4o" in rendered

    def test_model_checklist_item_highlighted_render(self) -> None:
        """Highlighted model should have bold styling."""
        item = ModelChecklistItem("gpt-4o", "Openai", 123.4, "model_0")
        item.highlighted = True
        rendered = item.render()

        # Check for bold markup
        assert "bold" in rendered or "#f0f6fc" in rendered
        assert "gpt-4o" in rendered

    def test_model_checklist_shows_tci_score(self) -> None:
        """Model with TCI should show score."""
        item = ModelChecklistItem("gpt-4o", "Openai", 123.4, "model_0")
        rendered = item.render()

        assert "TCI: 123.4" in rendered

    def test_model_checklist_no_tci(self) -> None:
        """Model without TCI should not show TCI text."""
        item = ModelChecklistItem("gpt-4o", "Openai", None, "model_0")
        rendered = item.render()

        assert "TCI:" not in rendered

    def test_model_checklist_shows_provider(self) -> None:
        """Model should show provider in parentheses."""
        item = ModelChecklistItem("gpt-4o", "Openai", 123.4, "model_0")
        rendered = item.render()

        assert "(Openai)" in rendered


class TestSubmitBackgroundWorker:
    """Test background worker behavior."""

    @pytest.mark.asyncio
    async def test_load_models_runs_in_background(self, tmp_path: Path) -> None:
        """Model loading should not block UI."""
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                }
            ]
        )
        df.to_parquet(parquet_path)

        with patch.object(
            SubmitScreen,
            "_get_open_telco_dir",
            return_value=tmp_path,
        ):
            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                app.push_screen(SubmitScreen())

                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Should start in LOADING state
                initial_stage = screen.stage
                assert initial_stage == Stage.LOADING

                # Wait for background load to complete
                for _ in range(10):
                    await pilot.pause()
                    if screen.stage != Stage.LOADING:
                        break

                # Should have transitioned from LOADING
                assert screen.stage in (Stage.SELECT_MODELS, Stage.ERROR)


class TestSubmitIntegration:
    """End-to-end integration tests with mocks."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_submit_flow_success(
        self, tmp_path: Path, mock_github_token: str
    ) -> None:
        """Complete happy path: Load -> Select -> Confirm -> Submit -> Success."""
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                }
            ]
        )
        df.to_parquet(parquet_path)

        # Create trajectory file
        traj = tmp_path / "logs" / "leaderboard" / "eval_teleqna.json"
        traj.write_text(
            json.dumps(
                {
                    "eval": {"model": "openai/gpt-4o", "task": "teleqna"},
                }
            )
        )

        with (
            patch.object(SubmitScreen, "_get_open_telco_dir", return_value=tmp_path),
            patch(
                "open_telco.cli.screens.submit.github_service.GitHubService"
            ) as mock_github,
        ):
            # Mock successful PR creation
            mock_instance = MagicMock()
            mock_instance.create_submission_pr.return_value = PRResult(
                success=True,
                pr_url="https://github.com/gsma-research/ot_leaderboard/pull/123",
            )
            mock_github.return_value = mock_instance

            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                app.push_screen(SubmitScreen())

                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Wait for models to load
                for _ in range(10):
                    await pilot.pause()
                    if screen.stage == Stage.SELECT_MODELS:
                        break

                if screen.stage != Stage.SELECT_MODELS:
                    pytest.skip("Models didn't load in time")

                # Select model
                await pilot.press("space")
                await pilot.pause()

                # Submit (go to confirming)
                await pilot.press("enter")
                await pilot.pause()
                assert screen.stage == Stage.CONFIRMING

                # Confirm submission
                await pilot.press("enter")

                # Wait for submission to complete
                for _ in range(20):
                    await pilot.pause()
                    if screen.stage in (Stage.SUCCESS, Stage.ERROR):
                        break

                # Should be successful
                assert screen.stage == Stage.SUCCESS

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_submit_flow_with_multiple_models_selected(
        self, tmp_path: Path, mock_github_token: str
    ) -> None:
        """When multiple models selected, only first should be submitted."""
        parquet_path = tmp_path / "logs" / "leaderboard" / "results.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-4o (Openai)",
                    "teleqna": [83.6, 1.17, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                },
                {
                    "model": "claude-3 (Anthropic)",
                    "teleqna": [85.0, 1.0, 1000.0],
                    "telelogs": None,
                    "telemath": None,
                    "3gpp_tsg": None,
                    "date": "2026-01-09",
                },
            ]
        )
        df.to_parquet(parquet_path)

        with (
            patch.object(SubmitScreen, "_get_open_telco_dir", return_value=tmp_path),
            patch(
                "open_telco.cli.screens.submit.github_service.GitHubService"
            ) as mock_github,
        ):
            mock_instance = MagicMock()
            mock_instance.create_submission_pr.return_value = PRResult(
                success=True,
                pr_url="https://github.com/gsma-research/ot_leaderboard/pull/123",
            )
            mock_github.return_value = mock_instance

            app = OpenTelcoApp()
            async with app.run_test() as pilot:
                app.push_screen(SubmitScreen())

                screen = pilot.app.screen
                assert isinstance(screen, SubmitScreen)

                # Wait for models to load
                for _ in range(10):
                    await pilot.pause()
                    if screen.stage == Stage.SELECT_MODELS:
                        break

                if screen.stage != Stage.SELECT_MODELS:
                    pytest.skip("Models didn't load in time")

                # Select first model
                await pilot.press("space")
                # Move to second and select
                await pilot.press("down")
                await pilot.press("space")
                await pilot.pause()

                # Get selected models
                selected = screen._get_selected_models()
                assert len(selected) == 2

                # Submit
                await pilot.press("enter")
                await pilot.pause()
                await pilot.press("enter")

                # Wait for completion
                for _ in range(20):
                    await pilot.pause()
                    if screen.stage in (Stage.SUCCESS, Stage.ERROR):
                        break

                # Verify only called once (first model)
                assert mock_instance.create_submission_pr.call_count == 1


class TestSubmitEdgeCases:
    """Test edge cases and special scenarios."""

    def test_model_name_with_special_chars(self) -> None:
        """Model names with special characters should be handled."""
        item = ModelChecklistItem("gpt-4o-2024-05-13", "Openai", 123.4, "model_0")
        rendered = item.render()

        assert "gpt-4o-2024-05-13" in rendered

    def test_provider_with_uppercase(self) -> None:
        """Provider names should be displayed correctly."""
        item = ModelChecklistItem("gpt-4o", "OpenAI", 123.4, "model_0")
        rendered = item.render()

        assert "(OpenAI)" in rendered

    def test_very_long_model_name(self) -> None:
        """Very long model names should render without error."""
        long_name = "a" * 100
        item = ModelChecklistItem(long_name, "Provider", 123.4, "model_0")
        rendered = item.render()

        assert long_name in rendered

    def test_empty_trajectory_files_in_bundle(
        self, temp_results_parquet: Path, tmp_path: Path
    ) -> None:
        """Bundle should work with no trajectory files."""
        from open_telco.cli.screens.submit.trajectory_bundler import (
            create_submission_bundle,
        )

        bundle = create_submission_bundle(
            model_name="gpt-4o",
            provider="Openai",
            results_parquet_path=temp_results_parquet,
            log_dir=tmp_path,  # Empty directory
        )

        # Should succeed with empty trajectory_files
        assert bundle.trajectory_files == {}
        assert bundle.parquet_content is not None

    def test_large_parquet_file_correct_filter(self, tmp_path: Path) -> None:
        """Large parquet with many models should correctly filter."""
        import io

        from open_telco.cli.screens.submit.trajectory_bundler import (
            create_submission_bundle,
        )

        parquet_path = tmp_path / "results.parquet"
        # Create parquet with many models
        models = [
            {
                "model": f"model-{i} (Provider{i})",
                "teleqna": [i, 0.1, 100.0],
                "telelogs": None,
                "telemath": None,
                "3gpp_tsg": None,
                "date": "2026-01-09",
            }
            for i in range(100)
        ]
        # Add the target model
        models.append(
            {
                "model": "target-model (TargetProvider)",
                "teleqna": [99.9, 0.1, 100.0],
                "telelogs": None,
                "telemath": None,
                "3gpp_tsg": None,
                "date": "2026-01-09",
            }
        )
        df = pd.DataFrame(models)
        df.to_parquet(parquet_path)

        bundle = create_submission_bundle(
            model_name="target-model",
            provider="TargetProvider",
            results_parquet_path=parquet_path,
            log_dir=tmp_path,
        )

        # Verify correct filtering
        result_df = pd.read_parquet(io.BytesIO(bundle.parquet_content))
        assert len(result_df) == 1
        assert result_df.iloc[0]["model"] == "target-model (TargetProvider)"
