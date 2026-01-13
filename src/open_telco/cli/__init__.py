"""Open Telco CLI package."""

# Eager load inspect_ai.analysis BEFORE Textual starts
# This prevents terminal interference during the session
from inspect_ai.analysis import evals_df  # noqa: F401

from open_telco.cli.app import OpenTelcoApp


def main() -> None:
    """Run the Open Telco CLI application."""
    app = OpenTelcoApp()
    app.run()


__all__ = ["OpenTelcoApp", "main"]
