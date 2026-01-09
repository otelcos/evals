"""Main menu screen for Open Telco CLI."""

from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView


class MainMenuScreen(Screen[None]):
    """Main menu screen with navigation options."""

    BINDINGS = [
        ("escape", "quit", "Quit"),
        ("q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    MainMenuScreen {
        align: center middle;
    }

    MainMenuScreen > Center > Vertical {
        width: auto;
        height: auto;
    }

    MainMenuScreen .title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 2;
    }

    MainMenuScreen ListView {
        width: 40;
        height: auto;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    MainMenuScreen ListItem {
        padding: 1 2;
    }

    MainMenuScreen ListItem:hover {
        background: $primary 20%;
    }

    MainMenuScreen ListView:focus > ListItem.--highlight {
        background: $primary 40%;
    }

    MainMenuScreen .menu-item {
        width: 100%;
    }

    MainMenuScreen .hint {
        text-align: center;
        margin-top: 2;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the main menu layout."""
        with Center():
            with Vertical():
                yield Label("Main Menu", classes="title")
                yield ListView(
                    ListItem(Label("1. Set Models", classes="menu-item"), id="set-models"),
                    ListItem(Label("2. Run Evals", classes="menu-item"), id="run-evals"),
                    ListItem(Label("3. Preview Results", classes="menu-item"), id="preview-results"),
                    ListItem(Label("4. Submit", classes="menu-item"), id="submit"),
                    id="menu-list",
                )
                yield Label("Press Enter to select, Q to quit", classes="hint")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle menu item selection."""
        item_id = event.item.id
        if item_id == "set-models":
            self.notify("Set Models - Coming soon!", title="Info")
        elif item_id == "run-evals":
            self.notify("Run Evals - Coming soon!", title="Info")
        elif item_id == "preview-results":
            self.notify("Preview Results - Coming soon!", title="Info")
        elif item_id == "submit":
            self.notify("Submit - Coming soon!", title="Info")

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
