"""Tests for model storage functionality.

Test 7: After entering model name, verify EnvManager.set() is called
with "INSPECT_EVAL_MODEL" and correct value.
Uses mock to avoid writing to real .env.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from open_telco.cli.app import OpenTelcoApp
from open_telco.cli.config.env_manager import PROVIDERS, EnvManager
from open_telco.cli.screens.set_models import ModelInputScreen


class TestModelStorage:
    """Test that model name is saved correctly via EnvManager.set()."""

    async def test_model_name_saved_via_env_manager(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Entering a model name should call EnvManager.set() with correct args."""
        # Setup: Create .env with OpenAI API key
        temp_env = tmp_path / ".env"
        temp_env.write_text('OPENAI_API_KEY="test-key"\n')
        monkeypatch.chdir(tmp_path)

        # Create mock for EnvManager.set()
        mock_set = MagicMock(return_value=True)

        app = OpenTelcoApp()

        with patch.object(EnvManager, "set", mock_set):
            async with app.run_test() as pilot:
                # Navigate: Welcome -> MainMenu -> SetModelsCategory -> ProviderSelect
                await pilot.press("enter")  # Welcome -> MainMenu
                await pilot.press("enter")  # MainMenu -> SetModelsCategory
                await pilot.press("enter")  # SetModelsCategory -> ProviderSelect

                # OpenAI is first provider (index 0), select it
                await pilot.press("enter")  # ProviderSelect -> ModelInputScreen

                assert isinstance(pilot.app.screen, ModelInputScreen)

                # The input has example model pre-filled, submit it
                await pilot.press("enter")

                # Verify EnvManager.set() was called with the example model
                mock_set.assert_called_once()
                call_args = mock_set.call_args[0]
                assert call_args[0] == "INSPECT_EVAL_MODEL"
                assert call_args[1] == PROVIDERS["OpenAI"]["example_model"]

    async def test_custom_model_name_saved(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Custom model name should be saved correctly."""
        # Setup
        temp_env = tmp_path / ".env"
        temp_env.write_text('OPENAI_API_KEY="test-key"\n')
        monkeypatch.chdir(tmp_path)

        mock_set = MagicMock(return_value=True)

        app = OpenTelcoApp()

        with patch.object(EnvManager, "set", mock_set):
            async with app.run_test() as pilot:
                # Navigate to ModelInputScreen
                await pilot.press("enter")  # Welcome -> MainMenu
                await pilot.press("enter")  # MainMenu -> SetModelsCategory
                await pilot.press("enter")  # SetModelsCategory -> ProviderSelect
                await pilot.press("enter")  # ProviderSelect -> ModelInputScreen

                # Clear the input and type a custom model
                model_input = pilot.app.screen.query_one("#model-input")
                model_input.value = "openai/gpt-4-turbo"

                await pilot.press("enter")

                # Verify custom model was saved
                mock_set.assert_called_once_with(
                    "INSPECT_EVAL_MODEL", "openai/gpt-4-turbo"
                )

    async def test_empty_model_name_not_saved(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty model name should not call EnvManager.set()."""
        # Setup
        temp_env = tmp_path / ".env"
        temp_env.write_text('OPENAI_API_KEY="test-key"\n')
        monkeypatch.chdir(tmp_path)

        mock_set = MagicMock(return_value=True)

        app = OpenTelcoApp()

        with patch.object(EnvManager, "set", mock_set):
            async with app.run_test() as pilot:
                # Navigate to ModelInputScreen
                await pilot.press("enter")  # Welcome -> MainMenu
                await pilot.press("enter")  # MainMenu -> SetModelsCategory
                await pilot.press("enter")  # SetModelsCategory -> ProviderSelect
                await pilot.press("enter")  # ProviderSelect -> ModelInputScreen

                # Clear the input
                model_input = pilot.app.screen.query_one("#model-input")
                model_input.value = ""

                await pilot.press("enter")

                # EnvManager.set() should NOT have been called
                mock_set.assert_not_called()


class TestModelStorageForAllProviders:
    """Test model storage works for all providers."""

    @pytest.mark.parametrize(
        ("provider_name", "env_key", "example_model"),
        [
            pytest.param(
                name,
                config["env_key"],
                config["example_model"],
                id=name.lower().replace(" ", "-").replace("(", "").replace(")", ""),
            )
            for name, config in PROVIDERS.items()
        ],
    )
    async def test_model_saved_for_each_provider(
        self,
        provider_name: str,
        env_key: str,
        example_model: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Model should be saved correctly when selected via each provider."""
        # Setup .env with this provider's API key
        temp_env = tmp_path / ".env"
        temp_env.write_text(f'{env_key}="test-key"\n')
        monkeypatch.chdir(tmp_path)

        mock_set = MagicMock(return_value=True)

        app = OpenTelcoApp()

        with patch.object(EnvManager, "set", mock_set):
            async with app.run_test() as pilot:
                # Navigate to ModelInputScreen
                await pilot.press("enter")  # Welcome -> MainMenu
                await pilot.press("enter")  # MainMenu -> SetModelsCategory
                await pilot.press("enter")  # SetModelsCategory -> ProviderSelect

                # Navigate to correct provider
                provider_names = list(PROVIDERS.keys())
                provider_index = provider_names.index(provider_name)
                for _ in range(provider_index):
                    await pilot.press("down")

                await pilot.press("enter")  # Select provider -> ModelInputScreen

                # Submit the default example model
                await pilot.press("enter")

                # Verify correct model was saved
                mock_set.assert_called_once_with("INSPECT_EVAL_MODEL", example_model)
