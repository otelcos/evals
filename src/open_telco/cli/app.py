"""Main Textual application for Open Telco CLI."""

from textual.app import App

from open_telco.cli.screens.welcome import WelcomeScreen


class OpenTelcoApp(App[None]):
    """Open Telco CLI application."""

    TITLE = "Open Telco"

    CSS = """
    Screen {
        background: $surface;
    }
    """

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.push_screen(WelcomeScreen())
