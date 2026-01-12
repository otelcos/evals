"""Reusable menu widgets for Open Telco CLI."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static

from open_telco.cli.constants import Colors


class MenuItem(Static):
    """A selectable menu item with optional disabled state."""

    DEFAULT_CSS = f"""
    MenuItem {{
        height: 1;
        padding: 0;
        background: transparent;
    }}

    MenuItem:hover {{
        background: {Colors.HOVER};
    }}
    """

    highlighted = reactive(False)

    def __init__(
        self, label: str, action: str, disabled: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.action = action
        self.disabled = disabled

    def render(self) -> str:
        prefix = f"[{Colors.RED}]>[/] " if self.highlighted else "  "
        style = self._compute_item_style()
        return f"{prefix}[{style}]{self.label}[/]"

    def _compute_item_style(self) -> str:
        if self.disabled:
            return Colors.TEXT_DISABLED
        if self.highlighted:
            return f"bold {Colors.TEXT_PRIMARY}"
        return Colors.TEXT_MUTED


class Menu(Vertical):
    """Container for menu items with keyboard navigation.

    Supports both 2-tuple (label, action) and 3-tuple (label, action, disabled) items.
    """

    DEFAULT_CSS = """
    Menu {
        height: auto;
        padding: 0;
    }
    """

    selected_index = reactive(0)

    def __init__(
        self, *items: tuple[str, str] | tuple[str, str, bool], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # Normalize items to 3-tuples using list comprehension
        self.items: list[tuple[str, str, bool]] = [
            (item[0], item[1], item[2] if len(item) > 2 else False) for item in items
        ]
        self._cached_items: list[MenuItem] | None = None

    def compose(self) -> ComposeResult:
        for label, action, disabled in self.items:
            yield MenuItem(label, action, disabled)

    def on_mount(self) -> None:
        # Cache widget references for performance
        self._cached_items = list(self.query(MenuItem))
        self._update_highlight()

    def watch_selected_index(self) -> None:
        self._update_highlight()

    def _update_highlight(self) -> None:
        items = self._cached_items or list(self.query(MenuItem))
        for i, item in enumerate(items):
            item.highlighted = i == self.selected_index

    def move_up(self) -> None:
        self.selected_index = (self.selected_index - 1) % len(self.items)

    def move_down(self) -> None:
        self.selected_index = (self.selected_index + 1) % len(self.items)

    def get_selected(self) -> tuple[str, str, bool]:
        """Get the currently selected item as (label, action, disabled)."""
        return self.items[self.selected_index]
