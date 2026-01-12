"""Provider selection screen."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Static

from open_telco.cli.base_screen import BaseScreen
from open_telco.cli.config import PROVIDERS, EnvManager
from open_telco.cli.constants import Colors
from open_telco.cli.widgets import Menu


class ProviderSelectScreen(BaseScreen):
    """Screen for selecting AI provider."""

    # Store provider names list for lookup
    PROVIDER_NAMES = list(PROVIDERS.keys())

    DEFAULT_CSS = BaseScreen.BASE_CSS + f"""
    ProviderSelectScreen {{
        padding: 0 4;
        layout: vertical;
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
    """

    BINDINGS = BaseScreen.BINDINGS + [
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
            f"[{Colors.TEXT_MUTED}]↵[/] select [{Colors.BORDER}]·[/] [{Colors.TEXT_MUTED}]q[/] back",
            id="footer",
            markup=True,
        )

    def action_up(self) -> None:
        self.query_one(Menu).move_up()

    def action_down(self) -> None:
        self.query_one(Menu).move_down()

    def action_select(self) -> None:
        label, provider_name, _disabled = self.query_one(Menu).get_selected()
        next_screen = self._get_next_screen(provider_name)
        self.app.push_screen(next_screen)

    def _get_next_screen(self, provider_name: str) -> Screen:
        """Determine next screen based on whether API key exists."""
        from open_telco.cli.screens.set_models.api_key_input import ApiKeyInputScreen
        from open_telco.cli.screens.set_models.model_input import ModelInputScreen

        env_manager = EnvManager()
        env_key = PROVIDERS[provider_name]["env_key"]

        if env_manager.has_key(env_key):
            return ModelInputScreen(provider_name, from_api_key_screen=False)

        return ApiKeyInputScreen(provider_name)
