"""Submit screen for submitting evaluation results to GSMA leaderboard."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import evals_df
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Static

from open_telco.cli.base_screen import BaseScreen
from open_telco.cli.config import EnvManager
from open_telco.cli.constants import Colors
from open_telco.cli.services.huggingface_client import (
    HuggingFaceError,
    fetch_and_transform_leaderboard,
    parse_model_provider,
)
from open_telco.cli.services.tci_calculator import LeaderboardEntry, calculate_all_tci
from open_telco.cli.utils.model_parser import parse_model_string

OPEN_TELCO_DIR = Path(__file__).parent.parent.parent.parent


class Stage(Enum):
    """Current stage of the submit flow."""

    LOADING = "loading"
    SELECT_MODELS = "select_models"
    CONFIRMING = "confirming"
    SUBMITTING = "submitting"
    SUCCESS = "success"
    ERROR = "error"


class ModelChecklistItem(Static):
    """A selectable model checklist item."""

    selected = reactive(False)
    highlighted = reactive(False)

    def __init__(
        self, model: str, provider: str, tci: float | None, item_id: str
    ) -> None:
        super().__init__(id=item_id)
        self.model = model
        self.provider = provider
        self.tci = tci

    def render(self) -> str:
        checkbox = (
            f"[{Colors.SUCCESS}][x][/]"
            if self.selected
            else f"[{Colors.TEXT_DISABLED}][ ][/]"
        )
        style = f"bold {Colors.TEXT_PRIMARY}" if self.highlighted else Colors.TEXT_MUTED
        tci_str = f" TCI: {self.tci:.1f}" if self.tci else ""
        return f"  {checkbox} [{style}]{self.model}[/] [{Colors.TEXT_DISABLED}]({self.provider}){tci_str}[/]"

    def toggle(self) -> bool:
        self.selected = not self.selected
        return self.selected


class SubmitScreen(BaseScreen):
    """Screen for submitting evaluation results."""

    DEFAULT_CSS = (
        BaseScreen.BASE_CSS
        + f"""
    SubmitScreen {{
        padding: 0 4;
        layout: vertical;
    }}

    #status {{
        color: {Colors.TEXT_MUTED};
        padding: 0 0 1 0;
        height: auto;
    }}

    #model-list-container {{
        width: 100%;
        max-width: 80;
        height: auto;
        max-height: 15;
        padding: 0;
    }}

    #model-list {{
        height: auto;
        padding: 0;
    }}

    ModelChecklistItem {{
        height: 1;
        padding: 0;
        background: transparent;
    }}

    #confirm-message {{
        color: {Colors.TEXT_PRIMARY};
        padding: 1 0;
        height: auto;
    }}

    #progress-container {{
        width: 100%;
        max-width: 60;
        height: auto;
        padding: 1 0;
    }}
    """
    )

    BINDINGS = BaseScreen.BINDINGS + [
        Binding("space", "toggle_selection", "Toggle", show=True),
        Binding("enter", "submit", "Submit", show=True),
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("j", "move_down", "Down", show=False),
    ]

    stage = reactive(Stage.LOADING)
    selected_index = reactive(0)

    def __init__(self) -> None:
        super().__init__()
        self._models: list[dict] = []
        self._github_token: str | None = None
        self._pr_url: str | None = None
        self.env_manager = EnvManager()

    def _get_open_telco_dir(self) -> Path:
        """Return the open_telco root directory. Mockable for tests."""
        return OPEN_TELCO_DIR

    def compose(self) -> ComposeResult:
        yield Static("submit", id="header")
        yield Static("loading...", id="status")
        with ScrollableContainer(id="model-list-container"):
            yield Vertical(id="model-list")
        yield Static("", id="confirm-message", markup=True)
        yield Container(id="progress-container")
        yield Static("", id="spacer")
        yield Static(f"[{Colors.TEXT_MUTED}]q[/] back", id="footer", markup=True)

    def on_mount(self) -> None:
        self._load_models()

    def _get_log_dir(self) -> Path:
        """Get the leaderboard log directory path."""
        return self._get_open_telco_dir() / "logs" / "leaderboard"

    def _try_load_models(self, log_dir: Path) -> list[dict] | None:
        """Try to load models from sources. Returns None on failure."""
        try:
            parquet_path = log_dir / "results.parquet"
            return self._load_models_from_sources(log_dir, parquet_path)
        except Exception:
            return None

    @work(exclusive=True, thread=True)
    def _load_models(self) -> None:
        log_dir = self._get_log_dir()

        if not log_dir.exists():
            self.app.call_from_thread(
                self._transition_to_error, "no-results-found. run-evals first."
            )
            return

        models = self._try_load_models(log_dir)
        if models is None:
            self.app.call_from_thread(
                self._transition_to_error, "failed-to-load-models"
            )
            return

        if not models:
            self.app.call_from_thread(
                self._transition_to_error, "no-results-found. run-evals first."
            )
            return

        self._models = models
        self.app.call_from_thread(self._show_models)

    def _load_models_from_sources(
        self, log_dir: Path, parquet_path: Path
    ) -> list[dict]:
        json_files = list(log_dir.glob("*.json"))

        # Load local entries
        local_entries: list[LeaderboardEntry] = []
        local_models_data: dict[str, dict] = {}

        if json_files:
            local_entries, local_models_data = self._load_entries_from_json(log_dir)
        elif parquet_path.exists():
            local_entries, local_models_data = self._load_entries_from_parquet(
                parquet_path
            )

        if not local_entries:
            return []

        # Fetch remote leaderboard for IRT fitting (so TCI is computed correctly)
        try:
            remote_entries = fetch_and_transform_leaderboard(timeout=30)
        except HuggingFaceError:
            remote_entries = []  # Continue without remote data if fetch fails

        # Merge: local entries take priority
        local_model_names = {e.model for e in local_entries}
        all_entries = local_entries + [
            e for e in remote_entries if e.model not in local_model_names
        ]

        # Fit IRT on full dataset and calculate TCI
        all_entries, _irt_params = calculate_all_tci(all_entries)

        # Extract TCI for local models only
        tci_lookup = {e.model: e.tci for e in all_entries}

        # Build model list with TCI values
        models = []
        for model_data in local_models_data.values():
            models.append(
                {
                    "model": model_data["model"],
                    "provider": model_data["provider"],
                    "tci": tci_lookup.get(model_data["model"]),
                    "model_str": model_data.get("model_str", model_data["model"]),
                    "raw_model": model_data.get("raw_model"),
                }
            )

        return models

    def _load_entries_from_json(
        self, log_dir: Path
    ) -> tuple[list[LeaderboardEntry], dict[str, dict]]:
        """Load local entries from JSON logs.

        Returns:
            Tuple of (LeaderboardEntry list, models_data dict for building final model list)
        """
        try:
            df = evals_df(str(log_dir), quiet=True)
        except Exception:
            return [], {}

        if df.empty:
            return [], {}

        models_data: dict[str, dict] = {}

        for _, row in df.iterrows():
            model_str = row.get("model", "Unknown")
            model_info = parse_model_string(model_str)
            model_key = f"{model_info.model_name}_{model_info.provider}"

            if model_key not in models_data:
                models_data[model_key] = {
                    "model": model_info.model_name,
                    "provider": model_info.provider,
                    "model_str": model_info.display_name,
                    "raw_model": model_str,
                    "scores": {},
                }

            task_name = row.get("task_name", "")
            score = row.get("score_headline_value")

            if pd.isna(score):
                continue

            score_val = float(score) * 100 if float(score) <= 1.0 else float(score)
            task_key = self._get_task_key(task_name)
            if task_key is not None:
                models_data[model_key]["scores"][task_key] = score_val

        # Build LeaderboardEntry list for IRT fitting
        entries = []
        for model_data in models_data.values():
            scores = model_data["scores"]
            entries.append(
                LeaderboardEntry(
                    model=model_data["model"],
                    provider=model_data["provider"],
                    teleqna=scores.get("teleqna"),
                    telelogs=scores.get("telelogs"),
                    telemath=scores.get("telemath"),
                    tsg=scores.get("tsg"),
                    is_user=True,
                )
            )

        return entries, models_data

    def _get_task_key(self, task_name: str) -> str | None:
        task_lower = task_name.lower()
        if "teleqna" in task_lower:
            return "teleqna"
        if "telelogs" in task_lower:
            return "telelogs"
        if "telemath" in task_lower:
            return "telemath"
        if "three_gpp" in task_lower or "3gpp" in task_lower:
            return "tsg"
        return None

    def _load_entries_from_parquet(
        self, parquet_path: Path
    ) -> tuple[list[LeaderboardEntry], dict[str, dict]]:
        """Load local entries from parquet file.

        Returns:
            Tuple of (LeaderboardEntry list, models_data dict for building final model list)
        """
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            return [], {}

        if df.empty:
            return [], {}

        entries = []
        models_data: dict[str, dict] = {}

        for _, row in df.iterrows():
            model_str = row.get("model", "Unknown")
            model, provider = parse_model_provider(model_str)

            teleqna = self._extract_score(row.get("teleqna"))
            telelogs = self._extract_score(row.get("telelogs"))
            telemath = self._extract_score(row.get("telemath"))
            tsg = self._extract_score(row.get("3gpp_tsg"))

            entry = LeaderboardEntry(
                model=model,
                provider=provider,
                teleqna=teleqna,
                telelogs=telelogs,
                telemath=telemath,
                tsg=tsg,
                is_user=True,
            )
            entries.append(entry)

            models_data[model] = {
                "model": model,
                "provider": provider,
                "model_str": model_str,
            }

        return entries, models_data

    def _extract_score(self, val: list | None) -> float | None:
        if val is None:
            return None
        if not hasattr(val, "__len__"):
            return None
        if len(val) < 1:
            return None
        return float(val[0])

    def _show_models(self) -> None:
        self.stage = Stage.SELECT_MODELS

        status = self.query_one("#status", Static)
        status.update("select models to submit:")

        model_list = self.query_one("#model-list", Vertical)
        model_list.remove_children()

        for i, model_data in enumerate(self._models):
            item = ModelChecklistItem(
                model=model_data["model"],
                provider=model_data["provider"],
                tci=model_data["tci"],
                item_id=f"model_{i}",
            )
            item.highlighted = i == 0
            model_list.mount(item)

        self.query_one("#footer", Static).update(
            f"[{Colors.TEXT_MUTED}]space[/] toggle [{Colors.BORDER}]|[/] "
            f"[{Colors.TEXT_MUTED}]enter[/] submit [{Colors.BORDER}]|[/] "
            f"[{Colors.TEXT_MUTED}]q[/] back"
        )

    def _transition_to_error(self, message: str) -> None:
        self.stage = Stage.ERROR
        self.query_one("#status", Static).update(f"[{Colors.ERROR}]{message}[/]")
        self.query_one("#footer", Static).update(f"[{Colors.TEXT_MUTED}]q[/] back")

    def _get_checklist_items(self) -> list[ModelChecklistItem]:
        return list(self.query(ModelChecklistItem))

    def _update_highlight(self) -> list[ModelChecklistItem]:
        items = self._get_checklist_items()
        for i, item in enumerate(items):
            item.highlighted = i == self.selected_index
        return items

    def action_move_up(self) -> None:
        if self.stage != Stage.SELECT_MODELS:
            return
        if self.selected_index <= 0:
            return
        self.selected_index -= 1
        self._update_highlight()

    def action_move_down(self) -> None:
        if self.stage != Stage.SELECT_MODELS:
            return
        items = self._get_checklist_items()
        if self.selected_index >= len(items) - 1:
            return
        self.selected_index += 1
        self._update_highlight()

    def action_toggle_selection(self) -> None:
        if self.stage != Stage.SELECT_MODELS:
            return
        items = self._get_checklist_items()
        if not items:
            return
        if self.selected_index < 0 or self.selected_index >= len(items):
            return
        items[self.selected_index].toggle()

    def _get_selected_models(self) -> list[dict]:
        items = self._get_checklist_items()
        selected = []
        for i, item in enumerate(items):
            if not item.selected:
                continue
            selected.append(self._models[i])
        return selected

    def action_submit(self) -> None:
        handlers = {
            Stage.SELECT_MODELS: self._handle_model_selection,
            Stage.CONFIRMING: self._start_submission,
            Stage.SUCCESS: self.app.pop_screen,
        }
        handler = handlers.get(self.stage)
        if handler is None:
            return
        handler()

    def _handle_model_selection(self) -> None:
        selected = self._get_selected_models()
        if not selected:
            self.notify("select at least one model", title="warning")
            return
        self._show_confirmation(selected)

    def _show_confirmation(self, selected: list[dict]) -> None:
        self.stage = Stage.CONFIRMING

        model_list = "\n".join(f"  - {m['model']} ({m['provider']})" for m in selected)

        self.query_one("#confirm-message", Static).update(
            f"[{Colors.TEXT_PRIMARY}]submit {len(selected)} model(s) to GSMA/open_telco?[/]\n\n"
            f"[{Colors.TEXT_MUTED}]{model_list}[/]"
        )

        self.query_one("#footer", Static).update(
            f"[{Colors.TEXT_MUTED}]enter[/] confirm [{Colors.BORDER}]|[/] [{Colors.TEXT_MUTED}]q[/] cancel"
        )

    def _start_submission(self) -> None:
        self._github_token = self.env_manager.get("GITHUB_TOKEN")
        if not self._github_token:
            self._transition_to_error(
                "GITHUB_TOKEN not set. use settings to configure."
            )
            return

        self.stage = Stage.SUBMITTING
        self.query_one("#confirm-message", Static).update("")
        self.query_one("#status", Static).update(
            f"[{Colors.TEXT_MUTED}]submitting...[/]"
        )
        self.query_one("#footer", Static).update(
            f"[{Colors.TEXT_MUTED}]please wait...[/]"
        )

        self._do_submission()

    @work(exclusive=True, thread=True)
    def _do_submission(self) -> None:
        from open_telco.cli.screens.submit.github_service import GitHubService
        from open_telco.cli.screens.submit.trajectory_bundler import (
            create_submission_bundle,
        )

        try:
            selected = self._get_selected_models()
            base_dir = self._get_open_telco_dir()
            parquet_path = base_dir / "logs" / "leaderboard" / "results.parquet"
            log_dir = base_dir / "logs" / "leaderboard"

            github = GitHubService(self._github_token)

            model_data = selected[0]

            self.app.call_from_thread(self._update_progress, "bundling trajectories...")

            bundle = create_submission_bundle(
                model_name=model_data["model"],
                provider=model_data["provider"],
                results_parquet_path=parquet_path if parquet_path.exists() else None,
                log_dir=log_dir,
                raw_model=model_data.get("raw_model"),
            )

            self.app.call_from_thread(self._update_progress, "creating pull request...")

            result = github.create_submission_pr(
                model_name=bundle.model_name,
                provider=bundle.provider,
                parquet_content=bundle.parquet_content,
                trajectory_files=bundle.trajectory_files,
            )

            if not result.success:
                self.app.call_from_thread(
                    self._transition_to_error, f"PR creation failed: {result.error}"
                )
                return

            self._pr_url = result.pr_url
            self.app.call_from_thread(self._show_success, result.pr_url)

        except Exception as e:
            self.app.call_from_thread(
                self._transition_to_error, f"submission failed: {e}"
            )

    def _update_progress(self, message: str) -> None:
        self.query_one("#status", Static).update(f"[{Colors.TEXT_MUTED}]{message}[/]")

    def _show_success(self, pr_url: str) -> None:
        self.stage = Stage.SUCCESS

        self.query_one("#status", Static).update(
            f"[{Colors.SUCCESS}]submission complete![/]"
        )

        self.query_one("#confirm-message", Static).update(
            f"[{Colors.TEXT_PRIMARY}]PR: {pr_url}[/]\n\n"
            f"[{Colors.TEXT_MUTED}]validation will run automatically.\n"
            "once approved, scores sync to HuggingFace.[/]"
        )

        self.query_one("#footer", Static).update(
            f"[{Colors.TEXT_MUTED}]enter[/] done [{Colors.BORDER}]|[/] [{Colors.TEXT_MUTED}]q[/] back"
        )

    def action_go_back(self) -> None:
        if self.stage == Stage.CONFIRMING:
            self._return_to_selection()
            return
        self.app.pop_screen()

    def _return_to_selection(self) -> None:
        self.stage = Stage.SELECT_MODELS
        self.query_one("#confirm-message", Static).update("")
        self.query_one("#footer", Static).update(
            f"[{Colors.TEXT_MUTED}]space[/] toggle [{Colors.BORDER}]|[/] "
            f"[{Colors.TEXT_MUTED}]enter[/] submit [{Colors.BORDER}]|[/] "
            f"[{Colors.TEXT_MUTED}]q[/] back"
        )
