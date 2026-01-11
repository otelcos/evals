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
from textual.screen import Screen
from textual.widgets import Static

from open_telco.cli.config import EnvManager
from open_telco.cli.services.huggingface_client import parse_model_provider
from open_telco.cli.services.tci_calculator import LeaderboardEntry, with_tci

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

    def __init__(self, model: str, provider: str, tci: float | None, item_id: str) -> None:
        super().__init__(id=item_id)
        self.model = model
        self.provider = provider
        self.tci = tci

    def render(self) -> str:
        checkbox = "[#3fb950][x][/]" if self.selected else "[#484f58][ ][/]"
        if self.highlighted:
            highlight_start = "[bold #f0f6fc]"
            highlight_end = "[/]"
        else:
            highlight_start = "[#8b949e]"
            highlight_end = "[/]"
        tci_str = f" TCI: {self.tci:.1f}" if self.tci else ""
        return f"  {checkbox} {highlight_start}{self.model}{highlight_end} [#484f58]({self.provider}){tci_str}[/]"

    def toggle(self) -> None:
        self.selected = not self.selected


class SubmitScreen(Screen[None]):
    """Screen for submitting evaluation results."""

    DEFAULT_CSS = """
    SubmitScreen {
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

    #model-list-container {
        width: 100%;
        max-width: 80;
        height: auto;
        max-height: 15;
        padding: 0;
    }

    #model-list {
        height: auto;
        padding: 0;
    }

    ModelChecklistItem {
        height: 1;
        padding: 0;
        background: transparent;
    }

    #confirm-message {
        color: #f0f6fc;
        padding: 1 0;
        height: auto;
    }

    #progress-container {
        width: 100%;
        max-width: 60;
        height: auto;
        padding: 1 0;
    }

    #spacer {
        height: 1fr;
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

    def compose(self) -> ComposeResult:
        yield Static("submit", id="header")
        yield Static("loading...", id="status")
        with ScrollableContainer(id="model-list-container"):
            yield Vertical(id="model-list")
        yield Static("", id="confirm-message", markup=True)
        yield Container(id="progress-container")
        yield Static("", id="spacer")
        yield Static("[#8b949e]q[/] back", id="footer", markup=True)

    def on_mount(self) -> None:
        """Start loading data."""
        self._load_models()

    @work(exclusive=True, thread=True)
    def _load_models(self) -> None:
        """Load models from JSON logs or results.parquet."""
        try:
            open_telco_dir = self._get_open_telco_dir()
            log_dir = open_telco_dir / "logs" / "leaderboard"
            parquet_path = log_dir / "results.parquet"

            if not log_dir.exists():
                self.app.call_from_thread(
                    self._show_error, "no-results-found. run-evals first."
                )
                return

            # Try JSON logs first (supports partial results)
            json_files = list(log_dir.glob("*.json"))

            if json_files:
                models = self._load_models_from_json(log_dir)
            elif parquet_path.exists():
                models = self._load_models_from_parquet(parquet_path)
            else:
                self.app.call_from_thread(
                    self._show_error, "no-results-found. run-evals first."
                )
                return

            if not models:
                self.app.call_from_thread(
                    self._show_error, "no-results-found. run-evals first."
                )
                return

            self._models = models
            self.app.call_from_thread(self._show_models)

        except Exception as e:
            self.app.call_from_thread(self._show_error, f"failed-to-load: {e}")

    def _load_models_from_json(self, log_dir: Path) -> list[dict]:
        """Load model data from JSON trajectory logs.

        Args:
            log_dir: Directory containing JSON log files

        Returns:
            List of model dictionaries with scores
        """
        try:
            df = evals_df(str(log_dir))
        except Exception:
            return []

        if df.empty:
            return []

        # Group by model
        models_data: dict[str, dict] = {}

        for _, row in df.iterrows():
            model_str = row.get("model", "Unknown")
            model, provider = self._format_model_display(model_str)

            model_key = f"{model}_{provider}"

            if model_key not in models_data:
                models_data[model_key] = {
                    "model": model,
                    "provider": provider,
                    "model_str": f"{model} ({provider})",
                    "raw_model": model_str,  # Store raw for trajectory matching
                    "scores": {},
                }

            # Extract task and score
            task_name = row.get("task_name", "")
            score = row.get("score_headline_value")

            if pd.notna(score):
                score_val = float(score) * 100 if float(score) <= 1.0 else float(score)

                task_lower = task_name.lower()
                if "teleqna" in task_lower:
                    models_data[model_key]["scores"]["teleqna"] = score_val
                elif "telelogs" in task_lower:
                    models_data[model_key]["scores"]["telelogs"] = score_val
                elif "telemath" in task_lower:
                    models_data[model_key]["scores"]["telemath"] = score_val
                elif "three_gpp" in task_lower or "3gpp" in task_lower:
                    models_data[model_key]["scores"]["tsg"] = score_val

        # Calculate TCI for each model
        models = []
        for model_data in models_data.values():
            scores = model_data["scores"]
            entry = LeaderboardEntry(
                model=model_data["model"],
                provider=model_data["provider"],
                teleqna=scores.get("teleqna"),
                telelogs=scores.get("telelogs"),
                telemath=scores.get("telemath"),
                tsg=scores.get("tsg"),
            )
            with_tci(entry)

            models.append({
                "model": model_data["model"],
                "provider": model_data["provider"],
                "tci": entry.tci,
                "model_str": model_data["model_str"],
                "raw_model": model_data["raw_model"],
            })

        return models

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

    def _load_models_from_parquet(self, parquet_path: Path) -> list[dict]:
        """Load models from parquet file (legacy fallback).

        Args:
            parquet_path: Path to results.parquet

        Returns:
            List of model dictionaries
        """
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            return []

        if df.empty:
            return []

        models = []
        for _, row in df.iterrows():
            model_str = row.get("model", "Unknown")
            model, provider = parse_model_provider(model_str)

            # Extract scores for TCI calculation
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
            )
            with_tci(entry)

            models.append({
                "model": model,
                "provider": provider,
                "tci": entry.tci,
                "model_str": model_str,
            })

        return models

    def _extract_score(self, val: list | None) -> float | None:
        """Extract score value from [score, stderr, n_samples] format."""
        if val is not None and hasattr(val, "__len__") and len(val) >= 1:
            return float(val[0])
        return None

    def _get_open_telco_dir(self) -> Path:
        """Get the open_telco source directory path."""
        return Path(__file__).parent.parent.parent.parent

    def _show_models(self) -> None:
        """Display loaded models in checklist."""
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
            if i == 0:
                item.highlighted = True
            model_list.mount(item)

        self.query_one("#footer", Static).update(
            "[#8b949e]space[/] toggle [#30363d]|[/] "
            "[#8b949e]enter[/] submit [#30363d]|[/] "
            "[#8b949e]q[/] back"
        )

    def _show_error(self, message: str) -> None:
        """Show error message."""
        self.stage = Stage.ERROR
        status = self.query_one("#status", Static)
        status.update(f"[#f85149]{message}[/]")
        self.query_one("#footer", Static).update("[#8b949e]q[/] back")

    def _get_checklist_items(self) -> list[ModelChecklistItem]:
        """Get all checklist items."""
        return list(self.query(ModelChecklistItem))

    def _update_highlight(self) -> None:
        """Update which item is highlighted."""
        items = self._get_checklist_items()
        for i, item in enumerate(items):
            item.highlighted = i == self.selected_index

    def action_move_up(self) -> None:
        """Move selection up."""
        if self.stage != Stage.SELECT_MODELS:
            return
        items = self._get_checklist_items()
        if items and self.selected_index > 0:
            self.selected_index -= 1
            self._update_highlight()

    def action_move_down(self) -> None:
        """Move selection down."""
        if self.stage != Stage.SELECT_MODELS:
            return
        items = self._get_checklist_items()
        if items and self.selected_index < len(items) - 1:
            self.selected_index += 1
            self._update_highlight()

    def action_toggle_selection(self) -> None:
        """Toggle selection of current item."""
        if self.stage != Stage.SELECT_MODELS:
            return
        items = self._get_checklist_items()
        if items and 0 <= self.selected_index < len(items):
            items[self.selected_index].toggle()

    def _get_selected_models(self) -> list[dict]:
        """Get list of selected model data."""
        items = self._get_checklist_items()
        selected = []
        for i, item in enumerate(items):
            if item.selected:
                selected.append(self._models[i])
        return selected

    def action_submit(self) -> None:
        """Handle submit action based on current stage."""
        if self.stage == Stage.SELECT_MODELS:
            selected = self._get_selected_models()
            if not selected:
                self.notify("select at least one model", title="warning")
                return
            self._show_confirmation(selected)
        elif self.stage == Stage.CONFIRMING:
            self._start_submission()
        elif self.stage == Stage.SUCCESS:
            self.app.pop_screen()

    def _show_confirmation(self, selected: list[dict]) -> None:
        """Show confirmation before submitting."""
        self.stage = Stage.CONFIRMING

        model_list = "\n".join(
            f"  - {m['model']} ({m['provider']})" for m in selected
        )

        confirm_msg = self.query_one("#confirm-message", Static)
        confirm_msg.update(
            f"[#f0f6fc]submit {len(selected)} model(s) to GSMA/open_telco?[/]\n\n"
            f"[#8b949e]{model_list}[/]"
        )

        self.query_one("#footer", Static).update(
            "[#8b949e]enter[/] confirm [#30363d]|[/] [#8b949e]q[/] cancel"
        )

    def _start_submission(self) -> None:
        """Start the submission process."""
        # Check for GitHub token from .env file
        self._github_token = self.env_manager.get("GITHUB_TOKEN")
        if not self._github_token:
            self._show_error(
                "GITHUB_TOKEN not set. use settings to configure."
            )
            return

        self.stage = Stage.SUBMITTING
        self.query_one("#confirm-message", Static).update("")
        self.query_one("#status", Static).update("[#8b949e]submitting...[/]")
        self.query_one("#footer", Static).update("[#8b949e]please wait...[/]")

        self._do_submission()

    @work(exclusive=True, thread=True)
    def _do_submission(self) -> None:
        """Perform the actual submission in background."""
        from open_telco.cli.screens.submit.github_service import GitHubService
        from open_telco.cli.screens.submit.trajectory_bundler import create_submission_bundle

        try:
            selected = self._get_selected_models()
            open_telco_dir = self._get_open_telco_dir()
            parquet_path = open_telco_dir / "logs" / "leaderboard" / "results.parquet"
            log_dir = open_telco_dir / "logs" / "leaderboard"

            github = GitHubService(self._github_token)

            # For now, submit the first selected model
            # In the future, we could batch multiple models
            model_data = selected[0]

            self.app.call_from_thread(
                self._update_progress, "bundling trajectories..."
            )

            # Pass raw_model for accurate trajectory matching if available
            bundle = create_submission_bundle(
                model_name=model_data["model"],
                provider=model_data["provider"],
                results_parquet_path=parquet_path if parquet_path.exists() else None,
                log_dir=log_dir,
                raw_model=model_data.get("raw_model"),
            )

            self.app.call_from_thread(
                self._update_progress, "creating pull request..."
            )

            result = github.create_submission_pr(
                model_name=bundle.model_name,
                provider=bundle.provider,
                parquet_content=bundle.parquet_content,
                trajectory_files=bundle.trajectory_files,
            )

            if result.success:
                self._pr_url = result.pr_url
                self.app.call_from_thread(self._show_success, result.pr_url)
            else:
                self.app.call_from_thread(
                    self._show_error, f"PR creation failed: {result.error}"
                )

        except Exception as e:
            self.app.call_from_thread(self._show_error, f"submission failed: {e}")

    def _update_progress(self, message: str) -> None:
        """Update progress message."""
        self.query_one("#status", Static).update(f"[#8b949e]{message}[/]")

    def _show_success(self, pr_url: str) -> None:
        """Show success message with PR URL."""
        self.stage = Stage.SUCCESS

        self.query_one("#status", Static).update("[#3fb950]submission complete![/]")

        confirm_msg = self.query_one("#confirm-message", Static)
        confirm_msg.update(
            f"[#f0f6fc]PR: {pr_url}[/]\n\n"
            "[#8b949e]validation will run automatically.\n"
            "once approved, scores sync to HuggingFace.[/]"
        )

        self.query_one("#footer", Static).update(
            "[#8b949e]enter[/] done [#30363d]|[/] [#8b949e]q[/] back"
        )

    def action_go_back(self) -> None:
        """Go back based on current stage."""
        if self.stage == Stage.CONFIRMING:
            # Go back to selection
            self.stage = Stage.SELECT_MODELS
            self.query_one("#confirm-message", Static).update("")
            self.query_one("#footer", Static).update(
                "[#8b949e]space[/] toggle [#30363d]|[/] "
                "[#8b949e]enter[/] submit [#30363d]|[/] "
                "[#8b949e]q[/] back"
            )
        else:
            self.app.pop_screen()
