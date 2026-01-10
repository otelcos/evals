"""Reusable menu widgets for Open Telco CLI."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static

# Shared color constants
GSMA_RED = "#a61d2d"
GSMA_BACKGROUND = "#0d1117"
GSMA_TEXT_MUTED = "#8b949e"
GSMA_TEXT_PRIMARY = "#f0f6fc"
GSMA_TEXT_DISABLED = "#484f58"
GSMA_BORDER = "#30363d"
GSMA_HOVER = "#21262d"


class MenuItem(Static):
    """A selectable menu item with optional disabled state."""

    DEFAULT_CSS = """
    MenuItem {
        height: 1;
        padding: 0;
        background: transparent;
    }

    MenuItem:hover {
        background: #21262d;
    }
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
        if self.disabled:
            if self.highlighted:
                return f"[{GSMA_RED}]>[/] [{GSMA_TEXT_DISABLED}]{self.label}[/]"
            return f"  [{GSMA_TEXT_DISABLED}]{self.label}[/]"
        if self.highlighted:
            return f"[{GSMA_RED}]>[/] [bold {GSMA_TEXT_PRIMARY}]{self.label}[/]"
        return f"  [{GSMA_TEXT_MUTED}]{self.label}[/]"


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
        # Normalize items to 3-tuples
        self.items: list[tuple[str, str, bool]] = []
        for item in items:
            if len(item) == 2:
                self.items.append((item[0], item[1], False))
            else:
                self.items.append(item)  # type: ignore
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
