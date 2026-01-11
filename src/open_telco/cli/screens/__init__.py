"""CLI screens package."""

from open_telco.cli.screens.main_menu import MainMenuScreen
from open_telco.cli.screens.run_evals import RunEvalsScreen
from open_telco.cli.screens.set_models import SetModelsCategoryScreen
from open_telco.cli.screens.settings import SettingsScreen
from open_telco.cli.screens.submit import SubmitScreen
from open_telco.cli.screens.welcome import WelcomeScreen

__all__ = [
    "MainMenuScreen",
    "RunEvalsScreen",
    "SetModelsCategoryScreen",
    "SettingsScreen",
    "SubmitScreen",
    "WelcomeScreen",
]
