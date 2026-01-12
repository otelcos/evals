"""Model name input screen."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Input, Static

from open_telco.cli.base_screen import BaseScreen
from open_telco.cli.config import PROVIDERS, EnvManager
from open_telco.cli.constants import Colors


class ModelInputScreen(BaseScreen):
    """Screen for entering model name."""

    DEFAULT_CSS = (
        BaseScreen.BASE_CSS
        + f"""
    ModelInputScreen {{
        padding: 0 4;
        layout: vertical;
    }}

    #form-container {{
        width: 100%;
        max-width: 65;
        height: auto;
        padding: 0 2;
    }}

    .provider-info {{
        color: {Colors.TEXT_MUTED};
        margin-bottom: 1;
    }}

    .info {{
        color: {Colors.TEXT_MUTED};
        margin-bottom: 1;
    }}

    .current-value {{
        color: {Colors.WARNING};
        margin-bottom: 1;
    }}

    .example {{
        color: {Colors.TEXT_DISABLED};
        margin-top: 1;
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
    )

    def __init__(self, provider_name: str, from_api_key_screen: bool = True) -> None:
        """Initialize with provider name."""
        super().__init__()
        self.provider_name = provider_name
        self.provider_config = PROVIDERS[provider_name]
        self.env_manager = EnvManager()
        self.from_api_key_screen = from_api_key_screen

    def compose(self) -> ComposeResult:
        prefix = self.provider_config["prefix"]
        example = self.provider_config["example_model"]
        current_model = self.env_manager.get("INSPECT_EVAL_MODEL")

        yield Static("enter-model-name", id="header")
        with Container(id="form-container"):
            with Vertical():
                yield Static(f"provider: {self.provider_name}", classes="provider-info")
                yield Static(f"format: {prefix}/<model-name>", classes="info")

                if current_model:
                    yield Static(f"current: {current_model}", classes="current-value")

                yield Input(
                    placeholder=example,
                    value=example,
                    id="model-input",
                )
                yield Static(f"example: {example}", classes="example")
        yield Static("", id="spacer")
        yield Static(
            f"[{Colors.TEXT_MUTED}]↵[/] save [{Colors.BORDER}]·[/] [{Colors.TEXT_MUTED}]esc[/] cancel",
            id="footer",
            markup=True,
        )

    def on_mount(self) -> None:
        """Focus the input field on mount."""
        self.query_one("#model-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle model name submission."""
        model_name = event.value.strip()

        if not model_name:
            self.notify("model-name-cannot-be-empty", severity="error")
            return

        # Save model to .env
        result = self.env_manager.set("INSPECT_EVAL_MODEL", model_name)

        if not result.success:
            self.notify(
                result.error or "failed-to-save-model-configuration", severity="error"
            )
            return

        self.notify(
            f"model-configured: {model_name}",
            severity="information",
            title="success",
        )
        # Return to main menu using switch_screen to avoid race conditions
        # from multiple pop_screen() calls
        from open_telco.cli.screens.main_menu import MainMenuScreen

        self.app.switch_screen(MainMenuScreen())
