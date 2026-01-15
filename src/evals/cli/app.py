"""Main Textual application for Open Telco CLI."""

from textual.app import App

from evals.cli.screens.welcome import WelcomeScreen


class OpenTelcoApp(App[None]):
    """Open Telco CLI application."""

    TITLE = "Open Telco"

    CSS = """
    Screen {
        background: #0d1117;
    }
    """

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.push_screen(WelcomeScreen())
