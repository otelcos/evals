"""Provider selection screen."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Static

from open_telco.cli.config import PROVIDERS, EnvManager

GSMA_RED = "#a61d2d"


class MenuItem(Static):
    """A selectable menu item."""

    highlighted = reactive(False)

    def __init__(self, label: str, action: str) -> None:
        super().__init__()
        self.label = label
        self.action = action

    def render(self) -> str:
        if self.highlighted:
            return f"[{GSMA_RED}]›[/] [bold #f0f6fc]{self.label}[/]"
        return f"  [#8b949e]{self.label}[/]"


class Menu(Vertical):
    """Container for menu items with keyboard navigation."""

    selected_index = reactive(0)

    def __init__(self, *items: tuple[str, str]) -> None:
        super().__init__()
        self.items = items

    def compose(self) -> ComposeResult:
        for label, action in self.items:
            yield MenuItem(label, action)

    def on_mount(self) -> None:
        self._update_highlight()

    def watch_selected_index(self) -> None:
        self._update_highlight()

    def _update_highlight(self) -> None:
        for i, item in enumerate(self.query(MenuItem)):
            item.highlighted = i == self.selected_index

    def move_up(self) -> None:
        self.selected_index = (self.selected_index - 1) % len(self.items)

    def move_down(self) -> None:
        self.selected_index = (self.selected_index + 1) % len(self.items)

    def get_selected(self) -> tuple[str, str]:
        return self.items[self.selected_index]


class ProviderSelectScreen(Screen[None]):
    """Screen for selecting AI provider."""

    # Store provider names list for lookup
    PROVIDER_NAMES = list(PROVIDERS.keys())

    DEFAULT_CSS = """
    ProviderSelectScreen {
        background: #0d1117;
        padding: 2 4;
        layout: vertical;
    }

    #header {
        color: #a61d2d;
        text-style: bold;
        padding: 1 0 2 0;
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

    def compose(self) -> ComposeResult:
        yield Static("select-provider", id="header")
        menu_items = tuple((name, name) for name in PROVIDERS.keys())
        with Container(id="menu-container"):
            yield Menu(*menu_items)
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
        label, provider_name = self.query_one(Menu).get_selected()

        # Check if API key already exists
        env_manager = EnvManager()
        env_key = PROVIDERS[provider_name]["env_key"]

        if env_manager.has_key(env_key):
            # Key exists, go directly to model input
            from open_telco.cli.screens.set_models.model_input import (
                ModelInputScreen,
            )

            self.app.push_screen(ModelInputScreen(provider_name, from_api_key_screen=False))
        else:
            # Key doesn't exist, ask for it first
            from open_telco.cli.screens.set_models.api_key_input import (
                ApiKeyInputScreen,
            )

            self.app.push_screen(ApiKeyInputScreen(provider_name))

    def action_go_back(self) -> None:
        """Go back to category menu."""
        self.app.pop_screen()
