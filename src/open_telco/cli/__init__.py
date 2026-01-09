"""Open Telco CLI package."""

from open_telco.cli.app import OpenTelcoApp


def main() -> None:
    """Run the Open Telco CLI application."""
    app = OpenTelcoApp()
    app.run()


__all__ = ["OpenTelcoApp", "main"]
