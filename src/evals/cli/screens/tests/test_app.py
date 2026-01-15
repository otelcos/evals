"""Tests for Open Telco CLI application core functionality.

Tests covered:
1. CLI startup test - Verify app starts without error
2. Timing test - Startup must be under 1 second
3. Navigation test - Press enter on WelcomeScreen goes to MainMenuScreen
4. Quit test - Press 'q' on MainMenuScreen exits app
5. set-model option test - Select set-model goes to SetModelsCategoryScreen
"""

import time

import pytest

from evals.cli.app import OpenTelcoApp
from evals.cli.screens.main_menu import MainMenuScreen
from evals.cli.screens.set_models import SetModelsCategoryScreen
from evals.cli.screens.welcome import WelcomeScreen


class TestCLIStartup:
    """Test 1: Verify app starts without error."""

    @pytest.mark.asyncio
    async def test_app_starts_successfully(self) -> None:
        """App should start and display WelcomeScreen without errors."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            assert pilot.app is not None
            assert isinstance(pilot.app.screen, WelcomeScreen)

    @pytest.mark.asyncio
    async def test_app_title_is_correct(self) -> None:
        """App title should be 'Open Telco'."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            assert pilot.app.TITLE == "Open Telco"


class TestStartupTiming:
    """Test 2: Startup must be under 0.1 seconds."""

    @pytest.mark.asyncio
    async def test_startup_under_100ms(self) -> None:
        """App startup should complete in under 0.1 seconds."""
        app = OpenTelcoApp()

        start_time = time.perf_counter()
        async with app.run_test() as pilot:
            await pilot.pause()
            elapsed = time.perf_counter() - start_time

        assert elapsed < 0.1, f"Startup took {elapsed:.3f}s, expected < 0.1s"


class TestNavigation:
    """Test 3: Navigation from WelcomeScreen to MainMenuScreen."""

    @pytest.mark.asyncio
    async def test_enter_navigates_to_main_menu(self) -> None:
        """Pressing enter on WelcomeScreen should navigate to MainMenuScreen."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            assert isinstance(pilot.app.screen, WelcomeScreen)

            await pilot.press("enter")

            assert isinstance(pilot.app.screen, MainMenuScreen)

    @pytest.mark.asyncio
    async def test_escape_also_navigates_to_main_menu(self) -> None:
        """Pressing escape on WelcomeScreen should also navigate to MainMenuScreen."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            assert isinstance(pilot.app.screen, WelcomeScreen)

            await pilot.press("escape")

            assert isinstance(pilot.app.screen, MainMenuScreen)


class TestQuit:
    """Test 4: Press 'q' on MainMenuScreen exits app."""

    @pytest.mark.asyncio
    async def test_quit_from_main_menu(self) -> None:
        """Pressing 'q' on MainMenuScreen should exit the application."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            await pilot.press("enter")
            assert isinstance(pilot.app.screen, MainMenuScreen)

            await pilot.press("q")

            assert pilot.app._exit


class TestSetModelOption:
    """Test 5: Select set-model option goes to SetModelsCategoryScreen."""

    @pytest.mark.asyncio
    async def test_set_model_navigates_to_category_screen(self) -> None:
        """Selecting set-model should navigate to SetModelsCategoryScreen."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            await pilot.press("enter")
            assert isinstance(pilot.app.screen, MainMenuScreen)

            await pilot.press("enter")

            assert isinstance(pilot.app.screen, SetModelsCategoryScreen)

    @pytest.mark.asyncio
    async def test_navigate_down_and_back_to_set_model(self) -> None:
        """Can navigate down/up and still select set-model correctly."""
        app = OpenTelcoApp()
        async with app.run_test() as pilot:
            await pilot.press("enter")

            await pilot.press("down")
            await pilot.press("up")

            await pilot.press("enter")

            assert isinstance(pilot.app.screen, SetModelsCategoryScreen)
