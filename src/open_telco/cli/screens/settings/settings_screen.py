"""Settings screen for managing GITHUB_TOKEN."""

from __future__ import annotations

from enum import Enum

import requests
from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widgets import Input, Static

from open_telco.cli.base_screen import BaseScreen
from open_telco.cli.config import EnvManager
from open_telco.cli.constants import Colors


class ValidationState(Enum):
    """Token validation state."""

    IDLE = "idle"
    VALIDATING = "validating"
    SUCCESS = "success"
    ERROR = "error"


class SettingsScreen(BaseScreen):
    """Screen for managing settings like GITHUB_TOKEN."""

    DEFAULT_CSS = BaseScreen.BASE_CSS + f"""
    SettingsScreen {{
        padding: 0 4;
        layout: vertical;
    }}

    #form-container {{
        width: 100%;
        max-width: 60;
        height: auto;
        padding: 0 2;
    }}

    .token-status {{
        color: {Colors.TEXT_MUTED};
        margin-bottom: 1;
    }}

    .token-status-set {{
        color: {Colors.SUCCESS};
    }}

    .token-status-notset {{
        color: {Colors.ERROR};
    }}

    .label {{
        color: {Colors.TEXT_PRIMARY};
        margin-top: 1;
        margin-bottom: 0;
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

    .hint {{
        color: {Colors.TEXT_DISABLED};
        margin-top: 0;
        margin-bottom: 1;
    }}

    #validation-status {{
        margin-top: 1;
        height: auto;
    }}

    .validation-success {{
        color: {Colors.SUCCESS};
    }}

    .validation-error {{
        color: {Colors.ERROR};
    }}

    .validation-pending {{
        color: {Colors.TEXT_MUTED};
    }}
    """

    validation_state = reactive(ValidationState.IDLE)

    def __init__(self) -> None:
        """Initialize settings screen."""
        super().__init__()
        self.env_manager = EnvManager()
        self._validation_message = ""

    def compose(self) -> ComposeResult:
        has_token = self.env_manager.has_key("GITHUB_TOKEN")
        status_text = f"[{Colors.SUCCESS}]set[/]" if has_token else f"[{Colors.ERROR}]not set[/]"

        yield Static("settings", id="header")
        with Container(id="form-container"):
            with Vertical():
                yield Static(
                    f"GITHUB_TOKEN: {status_text}",
                    id="token-status",
                    markup=True,
                )
                yield Static(
                    "enter GitHub Personal Access Token:",
                    classes="label",
                )
                yield Input(
                    placeholder="ghp_xxxxxxxxxxxxxxxxxxxx",
                    password=True,
                    id="token-input",
                )
                yield Static(
                    "token requires 'repo' scope for PR creation",
                    classes="hint",
                )
                yield Static(
                    "get a token at: github.com/settings/tokens",
                    classes="hint",
                )
                yield Static("", id="validation-status")
        yield Static("", id="spacer")
        yield Static(
            f"[{Colors.TEXT_MUTED}]enter[/] save [{Colors.BORDER}]|[/] [{Colors.TEXT_MUTED}]q[/] back",
            id="footer",
            markup=True,
        )

    def on_mount(self) -> None:
        """Focus the input field on mount."""
        self.query_one("#token-input", Input).focus()

    def watch_validation_state(self, state: ValidationState) -> None:
        """Update UI when validation state changes."""
        status_widget = self.query_one("#validation-status", Static)

        if state == ValidationState.IDLE:
            status_widget.update("")
            return
        if state == ValidationState.VALIDATING:
            status_widget.update(f"[{Colors.TEXT_MUTED}]validating token...[/]")
            return
        if state == ValidationState.SUCCESS:
            status_widget.update(f"[{Colors.SUCCESS}]{self._validation_message}[/]")
            return
        if state == ValidationState.ERROR:
            status_widget.update(f"[{Colors.ERROR}]{self._validation_message}[/]")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle token submission."""
        token = event.value.strip()

        if not token:
            self.notify("token cannot be empty", severity="error")
            return

        # Save token first
        success = self.env_manager.set("GITHUB_TOKEN", token)
        if not success:
            self.notify("failed to save token", severity="error")
            return

        # Update status display
        token_status = self.query_one("#token-status", Static)
        token_status.update(f"GITHUB_TOKEN: [{Colors.SUCCESS}]set[/]")

        # Validate the token
        self.validation_state = ValidationState.VALIDATING
        self._validate_token(token)

    def _build_github_headers(self, token: str) -> dict[str, str]:
        """Build headers for GitHub API requests."""
        return {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _try_github_request(self, url: str, headers: dict[str, str]) -> requests.Response | None:
        """Make a GitHub API request. Returns None on network failure."""
        try:
            return requests.get(url, headers=headers, timeout=30)
        except requests.Timeout:
            self._report_validation_error("validation timed out")
            return None
        except requests.RequestException as e:
            self._report_validation_error(f"network error: {e}")
            return None

    def _validate_user_token(self, headers: dict[str, str]) -> str | None:
        """Validate token and get username. Returns None on failure."""
        response = self._try_github_request("https://api.github.com/user", headers)
        if response is None:
            return None

        if response.status_code == 401:
            self._report_validation_error("invalid token (401 unauthorized)")
            return None
        if response.status_code != 200:
            self._report_validation_error(f"token check failed ({response.status_code})")
            return None

        return response.json().get("login", "unknown")

    def _validate_repo_access(self, headers: dict[str, str], user: str) -> bool:
        """Validate access to the leaderboard repo. Returns False on failure."""
        response = self._try_github_request(
            "https://api.github.com/repos/gsma-research/ot_leaderboard", headers
        )
        if response is None:
            return False

        if response.status_code == 404:
            self._report_validation_error(f"user {user}: cannot access gsma-research/ot_leaderboard")
            return False
        if response.status_code != 200:
            self._report_validation_error(f"repo check failed ({response.status_code})")
            return False

        return True

    def _report_validation_error(self, message: str) -> None:
        """Report a validation error to the UI."""
        self._validation_message = message
        self.app.call_from_thread(self._set_validation_state, ValidationState.ERROR)

    def _report_validation_success(self, user: str) -> None:
        """Report validation success to the UI."""
        self._validation_message = f"token valid for {user}. can create PRs via fork."
        self.app.call_from_thread(self._set_validation_state, ValidationState.SUCCESS)

    @work(exclusive=True, thread=True)
    def _validate_token(self, token: str) -> None:
        """Validate the GitHub token in background."""
        headers = self._build_github_headers(token)

        user = self._validate_user_token(headers)
        if user is None:
            return

        if not self._validate_repo_access(headers, user):
            return

        self._report_validation_success(user)

    def _set_validation_state(self, state: ValidationState) -> None:
        """Set validation state (called from thread)."""
        self.validation_state = state
