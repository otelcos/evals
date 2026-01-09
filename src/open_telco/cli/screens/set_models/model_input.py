"""Model name input screen."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Input, Static

from open_telco.cli.config import PROVIDERS, EnvManager

GSMA_RED = "#a61d2d"


class ModelInputScreen(Screen[None]):
    """Screen for entering model name."""

    DEFAULT_CSS = """
    ModelInputScreen {
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

    #form-container {
        width: 100%;
        max-width: 65;
        height: auto;
        padding: 0 2;
    }

    .provider-info {
        color: #8b949e;
        margin-bottom: 1;
    }

    .info {
        color: #8b949e;
        margin-bottom: 1;
    }

    .current-value {
        color: #f0883e;
        margin-bottom: 1;
    }

    .example {
        color: #484f58;
        margin-top: 1;
    }

    Input {
        width: 100%;
        margin: 1 0;
        background: #161b22;
        border: solid #30363d;
        color: #f0f6fc;
    }

    Input:focus {
        border: solid #a61d2d;
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
        Binding("escape", "go_back", "Back"),
    ]

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
            "[#8b949e]↵[/] save [#30363d]·[/] [#8b949e]esc[/] cancel",
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
        success = self.env_manager.set("INSPECT_EVAL_MODEL", model_name)

        if success:
            self.notify(
                f"model-configured: {model_name}",
                severity="information",
                title="success",
            )
            # Return to main menu (pop all set-models screens)
            if self.from_api_key_screen:
                # Pop: ModelInput -> ApiKeyInput -> ProviderSelect -> CategoryMenu
                self.app.pop_screen()  # Back to ApiKeyInput
                self.app.pop_screen()  # Back to ProviderSelect
                self.app.pop_screen()  # Back to CategoryMenu
                self.app.pop_screen()  # Back to MainMenu
            else:
                # Pop: ModelInput -> ProviderSelect -> CategoryMenu
                self.app.pop_screen()  # Back to ProviderSelect
                self.app.pop_screen()  # Back to CategoryMenu
                self.app.pop_screen()  # Back to MainMenu
        else:
            self.notify("failed-to-save-model-configuration", severity="error")

    def action_go_back(self) -> None:
        """Go back to API key input."""
        self.app.pop_screen()
