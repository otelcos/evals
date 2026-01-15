"""Model category selection screen."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Static

from evals.cli.base_screen import BaseScreen
from evals.cli.constants import Colors
from evals.cli.widgets import Menu


class SetModelsCategoryScreen(BaseScreen):
    """Screen for selecting model category."""

    DEFAULT_CSS = (
        BaseScreen.BASE_CSS
        + """
    SetModelsCategoryScreen {
        padding: 0 4;
        layout: vertical;
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
    """
    )

    BINDINGS = BaseScreen.BINDINGS + [
        Binding("enter", "select", "Select"),
        Binding("up", "up", "Up", show=False),
        Binding("down", "down", "Down", show=False),
        Binding("k", "up", "Up", show=False),
        Binding("j", "down", "Down", show=False),
    ]

    MENU_ITEMS = (
        ("lab-apis", "lab_apis", False),
        ("cloud-apis (coming-soon)", "cloud_apis", True),
        ("open-hosted (coming-soon)", "open_hosted", True),
        ("open-local (coming-soon)", "open_local", True),
    )

    def compose(self) -> ComposeResult:
        yield Static("set-model", id="header")
        with Container(id="menu-container"):
            yield Menu(*self.MENU_ITEMS)
        yield Static("", id="spacer")
        yield Static(
            f"[{Colors.TEXT_MUTED}]↵[/] select [{Colors.BORDER}]·[/] [{Colors.TEXT_MUTED}]q[/] back",
            id="footer",
            markup=True,
        )

    def action_up(self) -> None:
        self.query_one(Menu).move_up()

    def action_down(self) -> None:
        self.query_one(Menu).move_down()

    def action_select(self) -> None:
        label, action, disabled = self.query_one(Menu).get_selected()
        if disabled:
            self.notify("coming-soon!", title="info")
            return

        if action == "lab_apis":
            from evals.cli.screens.set_models.provider_select import (
                ProviderSelectScreen,
            )

            self.app.push_screen(ProviderSelectScreen())
