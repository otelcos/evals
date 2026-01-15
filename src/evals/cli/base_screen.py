"""Base screen class with shared CSS and navigation patterns."""

from __future__ import annotations

from textual.binding import Binding
from textual.screen import Screen

from evals.cli.constants import Colors


class BaseScreen(Screen[None]):
    """Base screen with common CSS and navigation.

    Provides:
    - Standard header/footer/spacer styling
    - Common key bindings (q, escape for back)
    - Default action_go_back implementation
    """

    BASE_CSS = f"""
    #header {{
        color: {Colors.RED};
        text-style: bold;
        padding: 0 0 2 0;
        height: auto;
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

    #error-message {{
        color: {Colors.ERROR};
        padding: 1 2;
        height: auto;
    }}

    #status {{
        color: {Colors.TEXT_MUTED};
        padding: 0 0 1 0;
        height: auto;
    }}
    """

    BINDINGS = [
        Binding("q", "go_back", "Back"),
        Binding("escape", "go_back", "Back"),
    ]

    def action_go_back(self) -> None:
        """Navigate back."""
        self.app.pop_screen()
