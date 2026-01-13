"""Preview leaderboard screen showing rankings with user models highlighted."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from inspect_ai.analysis import evals_df
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.widgets import DataTable, Static

from open_telco.cli.base_screen import BaseScreen
from open_telco.cli.constants import MIN_SCORE_ARRAY_LEN, TOP_N_DISPLAY, Colors
from open_telco.cli.services.huggingface_client import (
    HuggingFaceError,
    fetch_and_transform_leaderboard,
    parse_model_provider,
)
from open_telco.cli.services.tci_calculator import (
    LeaderboardEntry,
    calculate_all_tci,
    sort_by_tci,
)
from open_telco.cli.utils.model_parser import parse_model_string

OPEN_TELCO_DIR = Path(__file__).parent.parent.parent.parent


def _extract_score_tuple(val: list[float] | None) -> tuple[float | None, float | None]:
    """Extract score and stderr from [score, stderr, ...] format."""
    if val is None:
        return None, None
    if not hasattr(val, "__len__"):
        return None, None
    if len(val) < MIN_SCORE_ARRAY_LEN:
        return None, None
    return float(val[0]), float(val[1])


class PreviewLeaderboardScreen(BaseScreen):
    """Screen for previewing leaderboard with user models highlighted."""

    DEFAULT_CSS = (
        BaseScreen.BASE_CSS
        + f"""
    PreviewLeaderboardScreen {{
        padding: 0 4;
        layout: vertical;
    }}

    #status {{
        color: {Colors.TEXT_MUTED};
        padding: 0 0 1 0;
        height: auto;
    }}

    #table-container {{
        width: 100%;
        height: 1fr;
        padding: 0;
    }}

    DataTable {{
        height: 100%;
        width: 100%;
    }}

    DataTable > .datatable--header {{
        background: {Colors.HOVER};
        color: {Colors.TEXT_PRIMARY};
        text-style: bold;
    }}

    DataTable > .datatable--cursor {{
        background: {Colors.HOVER};
    }}

    DataTable > .datatable--header-cursor {{
        background: {Colors.HOVER};
    }}
    """
    )

    BINDINGS = BaseScreen.BINDINGS + [
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
        yield Static(f"[{Colors.TEXT_MUTED}]q[/] back", id="footer", markup=True)

    def on_mount(self) -> None:
        table = self.query_one("#leaderboard-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self._load_and_display()

    def _try_fetch_remote(self) -> list[LeaderboardEntry] | None:
        """Try to fetch remote leaderboard. Returns None on failure."""
        try:
            return fetch_and_transform_leaderboard(timeout=30)
        except HuggingFaceError:
            return None

    @work(exclusive=True, thread=True)
    def _load_and_display(self) -> None:
        user_entries = self._load_local_results()
        remote_entries = self._try_fetch_remote()

        if remote_entries is None:
            self.app.call_from_thread(
                self._show_error, "failed to fetch leaderboard", can_retry=True
            )
            return

        user_model_names = {e.model for e in user_entries}
        merged = self._merge_entries(remote_entries, user_entries)

        # Check if remote entries already have TCI from HuggingFace
        remote_has_tci = any(e.tci is not None for e in remote_entries)

        if remote_has_tci:
            # Remote models have TCI from HuggingFace - only calculate for local models
            # We still fit IRT on all data for proper calibration
            entries_needing_tci = [e for e in merged if e.tci is None]
            if entries_needing_tci:
                # Fit IRT on full dataset, then copy TCI only to entries that need it
                all_entries_copy, _irt_params = calculate_all_tci(merged.copy())
                tci_map = {e.model: e.tci for e in all_entries_copy}
                for entry in merged:
                    if entry.tci is None and entry.model in tci_map:
                        entry.tci = tci_map[entry.model]
        else:
            # Fallback: HuggingFace doesn't have TCI yet, calculate for all
            merged, _irt_params = calculate_all_tci(merged)

        sorted_entries = sort_by_tci(merged)
        display_entries = self._filter_display_entries(sorted_entries, user_model_names)

        self.app.call_from_thread(
            self._show_leaderboard, display_entries, user_model_names
        )

    def _merge_entries(
        self,
        remote_entries: list[LeaderboardEntry],
        user_entries: list[LeaderboardEntry],
    ) -> list[LeaderboardEntry]:
        user_models_map = {e.model: e for e in user_entries}
        remote_model_names: set[str] = set()
        merged: list[LeaderboardEntry] = []

        for entry in remote_entries:
            remote_model_names.add(entry.model)
            if entry.model in user_models_map:
                merged.append(user_models_map[entry.model])
                continue
            merged.append(entry)

        for entry in user_entries:
            if entry.model in remote_model_names:
                continue
            merged.append(entry)

        return merged

    def _filter_display_entries(
        self, sorted_entries: list[LeaderboardEntry], user_model_names: set[str]
    ) -> list[tuple[int, LeaderboardEntry]]:
        display_entries: list[tuple[int, LeaderboardEntry]] = []
        for i, entry in enumerate(sorted_entries):
            rank = i + 1
            if rank <= TOP_N_DISPLAY or entry.model in user_model_names:
                display_entries.append((rank, entry))
        return display_entries

    def _load_local_results(self) -> list[LeaderboardEntry]:
        log_dir = OPEN_TELCO_DIR / "logs" / "leaderboard"

        if not log_dir.exists():
            return []

        json_files = list(log_dir.glob("*.json"))

        if json_files:
            return self._load_from_json_logs(log_dir)

        parquet_path = log_dir / "results.parquet"
        if parquet_path.exists():
            return self._load_from_parquet(parquet_path)

        return []

    def _load_from_json_logs(self, log_dir: Path) -> list[LeaderboardEntry]:
        try:
            df = evals_df(str(log_dir), quiet=True)
        except Exception:
            return []

        if df.empty:
            return []

        entries: dict[str, LeaderboardEntry] = {}

        for _, row in df.iterrows():
            model_str = row.get("model", "Unknown")
            model_info = parse_model_string(model_str)

            if model_info.model_name not in entries:
                entries[model_info.model_name] = LeaderboardEntry(
                    model=model_info.model_name,
                    provider=model_info.provider,
                    is_user=True,
                )

            entry = entries[model_info.model_name]
            self._update_entry_score(entry, row)

        return list(entries.values())

    def _update_entry_score(self, entry: LeaderboardEntry, row: pd.Series) -> None:
        task_name = row.get("task_name", "")
        score = row.get("score_headline_value")
        stderr = row.get("score_headline_stderr")

        if pd.isna(score):
            return

        score_val = float(score) * 100 if float(score) <= 1.0 else float(score)
        stderr_val = None
        if pd.notna(stderr):
            stderr_val = float(stderr) * 100 if float(stderr) <= 1.0 else float(stderr)

        task_lower = task_name.lower()
        if "teleqna" in task_lower:
            entry.teleqna = score_val
            entry.teleqna_stderr = stderr_val
            return
        if "telelogs" in task_lower:
            entry.telelogs = score_val
            entry.telelogs_stderr = stderr_val
            return
        if "telemath" in task_lower:
            entry.telemath = score_val
            entry.telemath_stderr = stderr_val
            return
        if "three_gpp" in task_lower or "3gpp" in task_lower:
            entry.tsg = score_val
            entry.tsg_stderr = stderr_val

    def _load_from_parquet(self, parquet_path: Path) -> list[LeaderboardEntry]:
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

    def _show_leaderboard(
        self,
        display_entries: list[tuple[int, LeaderboardEntry]],
        user_models: set[str],
    ) -> None:
        self._user_models = user_models
        self._update_status(user_models)
        self._build_table(display_entries, user_models)
        self._update_footer_navigate()

    def _update_status(self, user_models: set[str]) -> None:
        status = self.query_one("#status", Static)
        if not user_models:
            status.update(
                f"[{Colors.TEXT_MUTED}]run evaluations to see your models ranked[/]"
            )
            return
        model_count = len(user_models)
        plural = "s" if model_count > 1 else ""
        status.update(
            f"[{Colors.SUCCESS}]{model_count} model{plural} from your evaluations highlighted[/]"
        )

    def _build_table(
        self, display_entries: list[tuple[int, LeaderboardEntry]], user_models: set[str]
    ) -> None:
        table = self.query_one("#leaderboard-table", DataTable)
        table.clear(columns=True)

        table.add_column("#", key="rank", width=4)
        table.add_column("Model", key="model", width=30)
        table.add_column("Provider", key="provider", width=12)
        table.add_column("TCI", key="tci", width=8)
        table.add_column("TeleQnA", key="teleqna", width=10)
        table.add_column("TeleLogs", key="telelogs", width=10)
        table.add_column("TeleMath", key="telemath", width=10)
        table.add_column("3GPP-TSG", key="tsg", width=10)

        first_user_row: int | None = None

        for row_idx, (rank, entry) in enumerate(display_entries):
            is_user = entry.model in user_models

            if is_user and first_user_row is None:
                first_user_row = row_idx

            row_values = self._build_row_values(rank, entry, is_user)
            table.add_row(*row_values, key=f"row_{row_idx}")

        if first_user_row is not None:
            table.move_cursor(row=first_user_row)

    def _build_row_values(
        self, rank: int, entry: LeaderboardEntry, is_user: bool
    ) -> tuple[str, str, str, str, str, str, str, str]:
        tci = f"{entry.tci:.1f}" if entry.tci is not None else "--"
        teleqna = f"{entry.teleqna:.1f}" if entry.teleqna is not None else "--"
        telelogs = f"{entry.telelogs:.1f}" if entry.telelogs is not None else "--"
        telemath = f"{entry.telemath:.1f}" if entry.telemath is not None else "--"
        tsg = f"{entry.tsg:.1f}" if entry.tsg is not None else "--"

        if not is_user:
            return (
                str(rank),
                entry.model,
                entry.provider,
                tci,
                teleqna,
                telelogs,
                telemath,
                tsg,
            )

        return (
            f"[{Colors.SUCCESS}]{rank}[/]",
            f"[{Colors.SUCCESS} bold]{entry.model}[/]",
            f"[{Colors.SUCCESS}]{entry.provider}[/]",
            f"[{Colors.SUCCESS} bold]{tci}[/]",
            f"[{Colors.SUCCESS}]{teleqna}[/]",
            f"[{Colors.SUCCESS}]{telelogs}[/]",
            f"[{Colors.SUCCESS}]{telemath}[/]",
            f"[{Colors.SUCCESS}]{tsg}[/]",
        )

    def _update_footer_navigate(self) -> None:
        self.query_one("#footer", Static).update(
            f"[{Colors.TEXT_MUTED}]↑↓[/] navigate [{Colors.BORDER}]|[/] [{Colors.TEXT_MUTED}]q[/] back"
        )

    def _show_error(self, message: str, can_retry: bool = False) -> None:
        self._error = message
        self.query_one("#status", Static).update(f"[{Colors.ERROR}]{message}[/]")

        if not can_retry:
            return

        self.query_one("#footer", Static).update(
            f"[{Colors.TEXT_MUTED}]r[/] retry [{Colors.BORDER}]|[/] [{Colors.TEXT_MUTED}]q[/] back"
        )

    def action_retry(self) -> None:
        self.query_one("#status", Static).update("loading...")
        self._load_and_display()
