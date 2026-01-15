"""Welcome screen for Open Telco CLI."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Static

WELCOME_BANNER = """\
┌──────────────────────────────┐
│ ✱ Welcome to Open Telco      │
└──────────────────────────────┘"""

ASCII_OPEN = """\
 ██████╗ ██████╗ ███████╗███╗   ██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║
╚██████╔╝██║     ███████╗██║ ╚████║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝"""

ASCII_TELCO = """\
████████╗███████╗██╗      ██████╗ ██████╗
╚══██╔══╝██╔════╝██║     ██╔════╝██╔═══██╗
   ██║   █████╗  ██║     ██║     ██║   ██║
   ██║   ██╔══╝  ██║     ██║     ██║   ██║
   ██║   ███████╗███████╗╚██████╗╚██████╔╝
   ╚═╝   ╚══════╝╚══════╝ ╚═════╝ ╚═════╝"""

PROMPT = "Press Enter to continue"


class WelcomeScreen(Screen[None]):
    """Welcome screen displaying ASCII art logo."""

    BINDINGS = [
        ("enter", "continue", "Continue"),
        ("escape", "continue", "Exit"),
    ]

    DEFAULT_CSS = """
    WelcomeScreen {
        padding: 0 4;
    }

    WelcomeScreen > Container {
        width: auto;
        height: auto;
    }

    WelcomeScreen .banner {
        color: #a61d2d;
        margin-bottom: 1;
    }

    WelcomeScreen .ascii-art {
        color: #a61d2d;
    }

    WelcomeScreen .prompt {
        margin-top: 1;
        color: #8b949e;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the welcome screen layout."""
        with Container():
            with Vertical():
                yield Static(WELCOME_BANNER, classes="banner")
                yield Static(ASCII_OPEN, classes="ascii-art")
                yield Static("", classes="spacer")
                yield Static(ASCII_TELCO, classes="ascii-art")
                yield Static(PROMPT, classes="prompt")

    def action_continue(self) -> None:
        """Handle continue action - go to main menu."""
        from evals.cli.screens.main_menu import MainMenuScreen

        self.app.switch_screen(MainMenuScreen())
