"""Preview leaderboard screen showing rankings with user models highlighted."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from inspect_ai.analysis import evals_df
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.screen import Screen
from textual.widgets import DataTable, Static

from open_telco.cli.services.huggingface_client import (
    HuggingFaceError,
    fetch_and_transform_leaderboard,
    parse_model_provider,
)
from open_telco.cli.services.tci_calculator import (
    LeaderboardEntry,
    sort_by_tci,
    with_tci,
)

# Map provider prefixes to display names
_PROVIDER_NAMES = {
    "openai": "Openai",
    "anthropic": "Anthropic",
    "google": "Google",
    "mistralai": "Mistral",
    "deepseek": "Deepseek",
    "meta-llama": "Meta",
    "cohere": "Cohere",
    "together": "Together",
    "openrouter": "Openrouter",
    "groq": "Groq",
    "fireworks": "Fireworks",
    "allenai": "Allenai",
}

# Minimum array length for score extraction (score, stderr)
_MIN_SCORE_ARRAY_LEN = 2

# Number of top entries to display (user models outside this range are also shown)
_TOP_N_DISPLAY = 10


def _extract_score_tuple(
    val: list[float] | None,
) -> tuple[float | None, float | None]:
    """Extract score and stderr from [score, stderr, ...] format."""
    if val is None:
        return None, None
    if hasattr(val, "__len__") and len(val) >= _MIN_SCORE_ARRAY_LEN:
        return float(val[0]), float(val[1])
    return None, None


class PreviewLeaderboardScreen(Screen[None]):
    """Screen for previewing leaderboard with user models highlighted."""

    DEFAULT_CSS = """
    PreviewLeaderboardScreen {
        padding: 0 4;
        layout: vertical;
    }

    #header {
        color: #a61d2d;
        text-style: bold;
        padding: 0 0 1 0;
        height: auto;
    }

    #status {
        color: #8b949e;
        padding: 0 0 1 0;
        height: auto;
    }

    #table-container {
        width: 100%;
        height: 1fr;
        padding: 0;
    }

    DataTable {
        height: 100%;
        width: 100%;
    }

    DataTable > .datatable--header {
        background: #21262d;
        color: #f0f6fc;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: #21262d;
    }

    DataTable > .datatable--header-cursor {
        background: #21262d;
    }

    #footer {
        dock: bottom;
        height: 1;
        color: #484f58;
    }
    """

    BINDINGS = [
        Binding("q", "go_back", "Back"),
        Binding("escape", "go_back", "Back"),
        Binding("r", "retry", "Retry", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._user_models: set[str] = set()
        self._all_entries: list[LeaderboardEntry] = []
        self._error: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("preview-leaderboard", id="header")
        yield Static("loading...", id="status")
        with ScrollableContainer(id="table-container"):
            yield DataTable(id="leaderboard-table")
        yield Static("[#8b949e]q[/] back", id="footer", markup=True)

    def on_mount(self) -> None:
        """Start loading data."""
        table = self.query_one("#leaderboard-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self._load_and_display()

    @work(exclusive=True, thread=True)
    def _load_and_display(self) -> None:
        """Load leaderboard data and display."""
        try:
            # Load user's local models (optional)
            user_entries = self._load_local_results()

            # Fetch remote leaderboard
            remote_entries = self._fetch_remote_leaderboard()

            # Calculate TCI for all entries
            for entry in remote_entries:
                with_tci(entry)

            for entry in user_entries:
                with_tci(entry)

            # Track user model names for highlighting
            user_model_names = {e.model for e in user_entries}
            user_models_map = {e.model: e for e in user_entries}

            # Merge: replace remote with local when model matches (use local scores)
            merged = []
            remote_model_names = set()
            for entry in remote_entries:
                remote_model_names.add(entry.model)
                if entry.model in user_models_map:
                    # Use local scores instead of remote
                    merged.append(user_models_map[entry.model])
                else:
                    merged.append(entry)

            # Add user models that don't exist in remote leaderboard
            for entry in user_entries:
                if entry.model not in remote_model_names:
                    merged.append(entry)

            # Sort by TCI
            sorted_entries = sort_by_tci(merged)

            # Filter to top N + any user models outside top N
            display_entries: list[tuple[int, LeaderboardEntry]] = []
            for i, entry in enumerate(sorted_entries):
                rank = i + 1
                if rank <= _TOP_N_DISPLAY or entry.model in user_model_names:
                    display_entries.append((rank, entry))

            # Display
            self.app.call_from_thread(
                self._show_leaderboard, display_entries, user_model_names
            )

        except HuggingFaceError as e:
            self.app.call_from_thread(self._show_error, str(e), can_retry=True)
        except Exception as e:
            self.app.call_from_thread(self._show_error, f"error: {e}", can_retry=False)

    def _get_open_telco_dir(self) -> Path:
        """Get the open_telco source directory path."""
        return Path(__file__).parent.parent.parent.parent

    def _load_local_results(self) -> list[LeaderboardEntry]:
        """Load user results from logs/leaderboard/ JSON files.

        Uses inspect_ai.analysis.evals_df() to parse JSON trajectory logs directly.
        This allows viewing partial results when only some evals completed.
        Falls back to results.parquet if no JSON logs found.

        Returns empty list if no local results exist (not an error).
        """
        open_telco_dir = self._get_open_telco_dir()
        log_dir = open_telco_dir / "logs" / "leaderboard"

        if not log_dir.exists():
            return []

        # Check if there are any JSON files
        json_files = list(log_dir.glob("*.json"))

        if json_files:
            # Use evals_df to parse JSON logs directly
            return self._load_from_json_logs(log_dir)

        # Fallback to parquet if it exists
        parquet_path = log_dir / "results.parquet"
        if parquet_path.exists():
            return self._load_from_parquet(parquet_path)

        return []

    def _load_from_json_logs(self, log_dir: Path) -> list[LeaderboardEntry]:
        """Load results from JSON trajectory logs using evals_df.

        Args:
            log_dir: Directory containing JSON log files

        Returns:
            List of LeaderboardEntry objects with scores from JSON logs
        """
        try:
            df = evals_df(str(log_dir))
        except Exception:
            return []

        if df.empty:
            return []

        # Group by model and aggregate scores
        entries: dict[str, LeaderboardEntry] = {}

        for _, row in df.iterrows():
            model_str = row.get("model", "Unknown")
            model, provider = self._format_model_display(model_str)

            # Create or get entry
            if model not in entries:
                entries[model] = LeaderboardEntry(
                    model=model,
                    provider=provider,
                    is_user=True,
                )

            entry = entries[model]

            # Extract task and score
            task_name = row.get("task_name", "")
            score = row.get("score_headline_value")
            stderr = row.get("score_headline_stderr")

            if pd.notna(score):
                # Normalize score to 0-100 scale
                score_val = float(score) * 100 if float(score) <= 1.0 else float(score)
                stderr_val = (
                    float(stderr) * 100
                    if pd.notna(stderr) and float(stderr) <= 1.0
                    else (float(stderr) if pd.notna(stderr) else None)
                )

                # Map task to entry field
                task_lower = task_name.lower()
                if "teleqna" in task_lower:
                    entry.teleqna = score_val
                    entry.teleqna_stderr = stderr_val
                elif "telelogs" in task_lower:
                    entry.telelogs = score_val
                    entry.telelogs_stderr = stderr_val
                elif "telemath" in task_lower:
                    entry.telemath = score_val
                    entry.telemath_stderr = stderr_val
                elif "three_gpp" in task_lower or "3gpp" in task_lower:
                    entry.tsg = score_val
                    entry.tsg_stderr = stderr_val

        return list(entries.values())

    def _format_model_display(self, model_str: str) -> tuple[str, str]:
        """Format model string to (model_name, provider) tuple.

        Handles formats like:
        - openrouter/openai/gpt-4o -> (gpt-4o, Openai)
        - openai/gpt-4o -> (gpt-4o, Openai)
        """
        parts = model_str.split("/")

        if len(parts) >= 3:
            # Format: router/provider/model
            provider = parts[1]
            model_name = "/".join(parts[2:])
        elif len(parts) == 2:
            # Format: provider/model
            provider = parts[0]
            model_name = parts[1]
        else:
            provider = "Unknown"
            model_name = model_str

        provider_display = _PROVIDER_NAMES.get(provider.lower(), provider.title())
        return model_name, provider_display

    def _load_from_parquet(self, parquet_path: Path) -> list[LeaderboardEntry]:
        """Load results from parquet file (legacy fallback).

        Args:
            parquet_path: Path to results.parquet

        Returns:
            List of LeaderboardEntry objects
        """
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            return []

        if df.empty:
            return []

        entries = []
        for _, row in df.iterrows():
            model_str = row.get("model", "Unknown")
            model, provider = parse_model_provider(model_str)

            teleqna, teleqna_stderr = _extract_score_tuple(row.get("teleqna"))
            telelogs, telelogs_stderr = _extract_score_tuple(row.get("telelogs"))
            telemath, telemath_stderr = _extract_score_tuple(row.get("telemath"))
            tsg, tsg_stderr = _extract_score_tuple(row.get("3gpp_tsg"))

            entries.append(
                LeaderboardEntry(
                    model=model,
                    provider=provider,
                    teleqna=teleqna,
                    teleqna_stderr=teleqna_stderr,
                    telelogs=telelogs,
                    telelogs_stderr=telelogs_stderr,
                    telemath=telemath,
                    telemath_stderr=telemath_stderr,
                    tsg=tsg,
                    tsg_stderr=tsg_stderr,
                    is_user=True,
                )
            )

        return entries

    def _fetch_remote_leaderboard(self) -> list[LeaderboardEntry]:
        """Fetch leaderboard from HuggingFace."""
        return fetch_and_transform_leaderboard(timeout=30)

    def _show_leaderboard(
        self,
        display_entries: list[tuple[int, LeaderboardEntry]],
        user_models: set[str],
    ) -> None:
        """Display the leaderboard table with user models highlighted."""
        self._user_models = user_models

        # Update status
        status = self.query_one("#status", Static)
        if user_models:
            model_count = len(user_models)
            status.update(
                f"[#3fb950]{model_count} model{'s' if model_count > 1 else ''} from your evaluations highlighted[/]"
            )
        else:
            status.update("[#8b949e]run evaluations to see your models ranked[/]")

        # Build table
        table = self.query_one("#leaderboard-table", DataTable)
        table.clear(columns=True)

        # Add columns
        table.add_column("#", key="rank", width=4)
        table.add_column("Model", key="model", width=30)
        table.add_column("Provider", key="provider", width=12)
        table.add_column("TCI", key="tci", width=8)
        table.add_column("TeleQnA", key="teleqna", width=10)
        table.add_column("TeleLogs", key="telelogs", width=10)
        table.add_column("TeleMath", key="telemath", width=10)
        table.add_column("3GPP-TSG", key="tsg", width=10)

        # Track first user model row for auto-scroll
        first_user_row: int | None = None

        # Add rows
        for row_idx, (rank, entry) in enumerate(display_entries):
            is_user = entry.model in user_models

            # Track first user model for auto-scroll
            if is_user and first_user_row is None:
                first_user_row = row_idx

            # Format values
            tci = f"{entry.tci:.1f}" if entry.tci is not None else "--"
            teleqna = f"{entry.teleqna:.1f}" if entry.teleqna is not None else "--"
            telelogs = f"{entry.telelogs:.1f}" if entry.telelogs is not None else "--"
            telemath = f"{entry.telemath:.1f}" if entry.telemath is not None else "--"
            tsg = f"{entry.tsg:.1f}" if entry.tsg is not None else "--"

            if is_user:
                # Highlight user's models with green color and arrow
                table.add_row(
                    f"[#3fb950]{rank}[/]",
                    f"[#3fb950 bold]{entry.model}[/]",
                    f"[#3fb950]{entry.provider}[/]",
                    f"[#3fb950 bold]{tci}[/]",
                    f"[#3fb950]{teleqna}[/]",
                    f"[#3fb950]{telelogs}[/]",
                    f"[#3fb950]{telemath}[/]",
                    f"[#3fb950]{tsg}[/]",
                    key=f"row_{row_idx}",
                )
            else:
                table.add_row(
                    str(rank),
                    entry.model,
                    entry.provider,
                    tci,
                    teleqna,
                    telelogs,
                    telemath,
                    tsg,
                    key=f"row_{row_idx}",
                )

        # Auto-scroll to first user model
        if first_user_row is not None:
            table.move_cursor(row=first_user_row)

        # Update footer
        self.query_one("#footer", Static).update(
            "[#8b949e]↑↓[/] navigate [#30363d]|[/] [#8b949e]q[/] back"
        )

    def _show_error(self, message: str, can_retry: bool = False) -> None:
        """Show error message."""
        self._error = message
        status = self.query_one("#status", Static)
        status.update(f"[#f85149]{message}[/]")

        if can_retry:
            self.query_one("#footer", Static).update(
                "[#8b949e]r[/] retry [#30363d]|[/] [#8b949e]q[/] back"
            )

    def action_go_back(self) -> None:
        """Go back to main menu."""
        self.app.pop_screen()

    def action_retry(self) -> None:
        """Retry fetching data after error."""
        self.query_one("#status", Static).update("loading...")
        self._load_and_display()
