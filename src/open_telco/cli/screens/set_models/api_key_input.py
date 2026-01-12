"""API key input screen."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Input, Static

from open_telco.cli.base_screen import BaseScreen
from open_telco.cli.config import PROVIDERS, EnvManager
from open_telco.cli.constants import Colors


class ApiKeyInputScreen(BaseScreen):
    """Screen for entering API key."""

    DEFAULT_CSS = BaseScreen.BASE_CSS + f"""
    ApiKeyInputScreen {{
        padding: 0 4;
        layout: vertical;
    }}

    #form-container {{
        width: 100%;
        max-width: 60;
        height: auto;
        padding: 0 2;
    }}

    .env-var {{
        color: {Colors.TEXT_MUTED};
        margin-top: 1;
        margin-bottom: 1;
    }}

    .current-value {{
        color: {Colors.WARNING};
        margin-bottom: 1;
    }}

    Input {{
        width: 100%;
        margin: 1 0;
        background: {Colors.BG_PRIMARY};
        border: solid {Colors.BORDER};
        color: {Colors.TEXT_PRIMARY};
    }}

    Input:focus {{
        border: solid {Colors.RED};
    }}
    """

    def __init__(self, provider_name: str) -> None:
        """Initialize with provider name."""
        super().__init__()
        self.provider_name = provider_name
        self.provider_config = PROVIDERS[provider_name]
        self.env_manager = EnvManager()

    def compose(self) -> ComposeResult:
        env_key = self.provider_config["env_key"]
        has_existing = self.env_manager.has_key(env_key)

        yield Static(f"enter-{self.provider_name.lower()}-api-key", id="header")
        with Container(id="form-container"):
            with Vertical():
                yield Static(f"environment-variable: {env_key}", classes="env-var")

                if has_existing:
                    yield Static("(key-already-set - will-be-overwritten)", classes="current-value")

                yield Input(
                    placeholder="enter-your-api-key...",
                    password=True,
                    id="api-key-input",
                )
        yield Static("", id="spacer")
        yield Static(
            f"[{Colors.TEXT_MUTED}]↵[/] save [{Colors.BORDER}]·[/] [{Colors.TEXT_MUTED}]esc[/] cancel",
            id="footer",
            markup=True,
        )

    def on_mount(self) -> None:
        """Focus the input field on mount."""
        self.query_one("#api-key-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle API key submission."""
        api_key = event.value.strip()

        if not api_key:
            self.notify("api-key-cannot-be-empty", severity="error")
            return

        # Save API key to .env
        env_key = self.provider_config["env_key"]
        success = self.env_manager.set(env_key, api_key)

        if not success:
            self.notify("failed-to-save-api-key", severity="error")
            return

        self.notify(f"saved-{env_key}", severity="information")
        # Navigate to model input screen
        from open_telco.cli.screens.set_models.model_input import ModelInputScreen

        self.app.push_screen(ModelInputScreen(self.provider_name))
