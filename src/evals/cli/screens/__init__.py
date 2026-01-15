"""CLI screens package.

Screens are imported lazily to avoid loading heavy dependencies (pandas, numpy, etc.)
on startup. Import screens directly when needed:
    from evals.cli.screens.welcome import WelcomeScreen
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
        from evals.cli.screens.main_menu import MainMenuScreen

        return MainMenuScreen
    if name == "RunEvalsScreen":
        from evals.cli.screens.run_evals import RunEvalsScreen

        return RunEvalsScreen
    if name == "SetModelsCategoryScreen":
        from evals.cli.screens.set_models import SetModelsCategoryScreen

        return SetModelsCategoryScreen
    if name == "SettingsScreen":
        from evals.cli.screens.settings import SettingsScreen

        return SettingsScreen
    if name == "SubmitScreen":
        from evals.cli.screens.submit import SubmitScreen

        return SubmitScreen
    if name == "WelcomeScreen":
        from evals.cli.screens.welcome import WelcomeScreen

        return WelcomeScreen
    raise AttributeError(f"module 'evals.cli.screens' has no attribute {name!r}")
