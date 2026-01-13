"""CLI screens package.

Screens are imported lazily to avoid loading heavy dependencies (pandas, numpy, etc.)
on startup. Import screens directly when needed:
    from open_telco.cli.screens.welcome import WelcomeScreen
"""

__all__ = [
    "MainMenuScreen",
    "RunEvalsScreen",
    "SetModelsCategoryScreen",
    "SettingsScreen",
    "SubmitScreen",
    "WelcomeScreen",
]


def __getattr__(name: str):
    """Lazy import screens to avoid loading heavy dependencies at startup."""
    if name == "MainMenuScreen":
        from open_telco.cli.screens.main_menu import MainMenuScreen

        return MainMenuScreen
    if name == "RunEvalsScreen":
        from open_telco.cli.screens.run_evals import RunEvalsScreen

        return RunEvalsScreen
    if name == "SetModelsCategoryScreen":
        from open_telco.cli.screens.set_models import SetModelsCategoryScreen

        return SetModelsCategoryScreen
    if name == "SettingsScreen":
        from open_telco.cli.screens.settings import SettingsScreen

        return SettingsScreen
    if name == "SubmitScreen":
        from open_telco.cli.screens.submit import SubmitScreen

        return SubmitScreen
    if name == "WelcomeScreen":
        from open_telco.cli.screens.welcome import WelcomeScreen

        return WelcomeScreen
    raise AttributeError(f"module 'open_telco.cli.screens' has no attribute {name!r}")
