"""Main menu screen for Open Telco CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Static

from open_telco.cli.constants import Colors
from open_telco.cli.widgets import Menu

if TYPE_CHECKING:
    from textual.screen import Screen as ScreenType


def _create_set_models_screen() -> ScreenType:
    from open_telco.cli.screens.set_models import SetModelsCategoryScreen

    return SetModelsCategoryScreen()


def _create_run_evals_screen() -> ScreenType:
    from open_telco.cli.screens.run_evals import RunEvalsScreen

    return RunEvalsScreen()


def _create_preview_leaderboard_screen() -> ScreenType:
    from open_telco.cli.screens.preview_leaderboard import PreviewLeaderboardScreen

    return PreviewLeaderboardScreen()


def _create_submit_screen() -> ScreenType:
    from open_telco.cli.screens.submit import SubmitScreen

    return SubmitScreen()


def _create_settings_screen() -> ScreenType:
    from open_telco.cli.screens.settings import SettingsScreen

    return SettingsScreen()


SCREEN_FACTORIES: dict[str, Callable[[], ScreenType]] = {
    "set_models": _create_set_models_screen,
    "run_evals": _create_run_evals_screen,
    "preview_leaderboard": _create_preview_leaderboard_screen,
    "submit": _create_submit_screen,
    "settings": _create_settings_screen,
}


class MainMenuScreen(Screen[None]):
    """Main menu screen with navigation options."""

    DEFAULT_CSS = f"""
    MainMenuScreen {{
        padding: 0 4;
        layout: vertical;
    }}

    #header {{
        color: {Colors.RED};
        text-style: bold;
        padding: 0 0 2 0;
        height: auto;
    }}

    #menu-container {{
        width: 100%;
        max-width: 50;
        height: auto;
        padding: 0 2;
    }}

    Menu {{
        height: auto;
        padding: 0;
    }}

    MenuItem {{
        height: 1;
        padding: 0;
        background: transparent;
    }}

    #spacer {{
        height: 1fr;
    }}

    #footer {{
        dock: bottom;
        height: 1;
        padding: 0 0;
        color: {Colors.TEXT_DISABLED};
    }}
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
            f"[{Colors.TEXT_MUTED}]↵[/] select [{Colors.BORDER}]·[/] "
            f"[{Colors.TEXT_MUTED}]s[/] settings [{Colors.BORDER}]·[/] "
            f"[{Colors.TEXT_MUTED}]q[/] quit",
            id="footer",
            markup=True,
        )

    def action_up(self) -> None:
        self.query_one(Menu).move_up()

    def action_down(self) -> None:
        self.query_one(Menu).move_down()

    def action_select(self) -> None:
        _label, action, _disabled = self.query_one(Menu).get_selected()
        screen = self._get_screen_for_action(action)
        if screen is None:
            return
        self.app.push_screen(screen)

    def _get_screen_for_action(self, action: str) -> ScreenType | None:
        factory = SCREEN_FACTORIES.get(action)
        if factory is None:
            return None
        return factory()

    def action_settings(self) -> None:
        screen = self._get_screen_for_action("settings")
        if screen is None:
            return
        self.app.push_screen(screen)

    def action_quit(self) -> None:
        self.app.exit()
