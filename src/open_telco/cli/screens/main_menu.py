"""Main menu screen for Open Telco CLI."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Static

from open_telco.cli.widgets import Menu


class MainMenuScreen(Screen[None]):
    """Main menu screen with navigation options."""

    DEFAULT_CSS = """
    MainMenuScreen {
        padding: 0 4;
        layout: vertical;
    }

    #header {
        color: #a61d2d;
        text-style: bold;
        padding: 0 0 2 0;
        height: auto;
    }

    #menu-container {
        width: 100%;
        max-width: 50;
        height: auto;
        padding: 0 2;
    }

    Menu {
        height: auto;
        padding: 0;
    }

    MenuItem {
        height: 1;
        padding: 0;
        background: transparent;
    }

    #spacer {
        height: 1fr;
    }

    #footer {
        dock: bottom;
        height: 1;
        padding: 0 0;
        color: #484f58;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("enter", "select", "Select"),
        Binding("s", "settings", "Settings"),
        Binding("up", "up", "Up", show=False),
        Binding("down", "down", "Down", show=False),
        Binding("k", "up", "Up", show=False),
        Binding("j", "down", "Down", show=False),
    ]

    MENU_ITEMS = (
        ("set-model", "set_models"),
        ("run-evals", "run_evals"),
        ("preview-leaderboard", "preview_leaderboard"),
        ("submit", "submit"),
    )

    def compose(self) -> ComposeResult:
        yield Static("OPEN TELCO", id="header")
        with Container(id="menu-container"):
            yield Menu(*self.MENU_ITEMS)
        yield Static("", id="spacer")
        yield Static(
            "[#8b949e]↵[/] select [#30363d]·[/] [#8b949e]s[/] settings [#30363d]·[/] [#8b949e]q[/] quit",
            id="footer",
            markup=True,
        )

    def action_up(self) -> None:
        self.query_one(Menu).move_up()

    def action_down(self) -> None:
        self.query_one(Menu).move_down()

    def action_select(self) -> None:
        label, action, _disabled = self.query_one(Menu).get_selected()
        if action == "set_models":
            from open_telco.cli.screens.set_models import SetModelsCategoryScreen

            self.app.push_screen(SetModelsCategoryScreen())
        elif action == "run_evals":
            from open_telco.cli.screens.run_evals import RunEvalsScreen

            self.app.push_screen(RunEvalsScreen())
        elif action == "preview_leaderboard":
            from open_telco.cli.screens.preview_leaderboard import (
                PreviewLeaderboardScreen,
            )

            self.app.push_screen(PreviewLeaderboardScreen())
        elif action == "submit":
            from open_telco.cli.screens.submit import SubmitScreen

            self.app.push_screen(SubmitScreen())

    def action_settings(self) -> None:
        """Open settings screen."""
        from open_telco.cli.screens.settings import SettingsScreen

        self.app.push_screen(SettingsScreen())

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
