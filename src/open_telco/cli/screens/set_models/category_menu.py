"""Model category selection screen."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Static

from open_telco.cli.widgets import Menu


class SetModelsCategoryScreen(Screen[None]):
    """Screen for selecting model category."""

    DEFAULT_CSS = """
    SetModelsCategoryScreen {
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
        Binding("q", "go_back", "Back"),
        Binding("escape", "go_back", "Back"),
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
            "[#8b949e]↵[/] select [#30363d]·[/] [#8b949e]q[/] back",
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
            from open_telco.cli.screens.set_models.provider_select import (
                ProviderSelectScreen,
            )

            self.app.push_screen(ProviderSelectScreen())

    def action_go_back(self) -> None:
        """Go back to main menu."""
        self.app.pop_screen()
