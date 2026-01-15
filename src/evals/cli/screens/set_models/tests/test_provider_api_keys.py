"""Tests for provider API key flow - skips ApiKeyInputScreen when key exists.

Test 6: For EACH provider, if .env has that provider's API key,
skip ApiKeyInputScreen and go directly to ModelInputScreen.
Uses temp .env files for isolation.
"""

from pathlib import Path

import pytest

from evals.cli.app import OpenTelcoApp
from evals.cli.config.env_manager import PROVIDERS
from evals.cli.screens.main_menu import MainMenuScreen
from evals.cli.screens.set_models import (
    ApiKeyInputScreen,
    ModelInputScreen,
    ProviderSelectScreen,
    SetModelsCategoryScreen,
)

# Generate test parameters for all 9 providers
PROVIDER_TEST_PARAMS = [
    pytest.param(
        name,
        config["env_key"],
        id=name.lower().replace(" ", "-").replace("(", "").replace(")", ""),
    )
    for name, config in PROVIDERS.items()
]


class TestProviderApiKeySkip:
    """Test that ApiKeyInputScreen is skipped when API key exists."""

    @pytest.mark.parametrize(
        ("provider_name", "env_key"),
        PROVIDER_TEST_PARAMS,
    )
    async def test_skip_api_key_screen_when_key_exists(
        self,
        provider_name: str,
        env_key: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When API key exists in .env, should skip to ModelInputScreen directly."""
        # Create .env in temp directory with the API key
        temp_env = tmp_path / ".env"
        temp_env.write_text(f'{env_key}="test-api-key-12345"\n')

        # Change to temp directory so EnvManager finds our .env
        monkeypatch.chdir(tmp_path)

        app = OpenTelcoApp()

        async with app.run_test() as pilot:
            # Navigate: Welcome -> MainMenu -> SetModelsCategory -> ProviderSelect
            await pilot.press("enter")  # Welcome -> MainMenu
            assert isinstance(pilot.app.screen, MainMenuScreen)

            await pilot.press("enter")  # MainMenu -> SetModelsCategory
            assert isinstance(pilot.app.screen, SetModelsCategoryScreen)

            await pilot.press("enter")  # SetModelsCategory -> ProviderSelect
            assert isinstance(pilot.app.screen, ProviderSelectScreen)

            # Navigate to the correct provider in the list
            provider_names = list(PROVIDERS.keys())
            provider_index = provider_names.index(provider_name)

            for _ in range(provider_index):
                await pilot.press("down")

            # Select the provider
            await pilot.press("enter")

            # Should go directly to ModelInputScreen, NOT ApiKeyInputScreen
            assert isinstance(pilot.app.screen, ModelInputScreen), (
                f"Expected ModelInputScreen for {provider_name}, "
                f"got {type(pilot.app.screen).__name__}"
            )

    @pytest.mark.parametrize(
        ("provider_name", "env_key"),
        PROVIDER_TEST_PARAMS,
    )
    async def test_show_api_key_screen_when_key_missing(
        self,
        provider_name: str,
        env_key: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When API key is missing from .env, should show ApiKeyInputScreen."""
        # Create empty temporary .env file (no API key)
        temp_env = tmp_path / ".env"
        temp_env.write_text("")

        monkeypatch.chdir(tmp_path)

        app = OpenTelcoApp()

        async with app.run_test() as pilot:
            # Navigate: Welcome -> MainMenu -> SetModelsCategory -> ProviderSelect
            await pilot.press("enter")  # Welcome -> MainMenu
            await pilot.press("enter")  # MainMenu -> SetModelsCategory
            await pilot.press("enter")  # SetModelsCategory -> ProviderSelect

            # Navigate to the correct provider
            provider_names = list(PROVIDERS.keys())
            provider_index = provider_names.index(provider_name)

            for _ in range(provider_index):
                await pilot.press("down")

            # Select the provider
            await pilot.press("enter")

            # Should show ApiKeyInputScreen
            assert isinstance(pilot.app.screen, ApiKeyInputScreen), (
                f"Expected ApiKeyInputScreen for {provider_name}, "
                f"got {type(pilot.app.screen).__name__}"
            )
