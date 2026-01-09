"""Run-evals screen with checklist UI."""

from datetime import date
from enum import Enum
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import evals_df
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Static
from textual import work

from open_telco.cli.config import EnvManager

GSMA_RED = "#a61d2d"

# All available tasks for full evaluation
ALL_TASKS = [
    "telelogs/telelogs.py",
    "telemath/telemath.py",
    "teleqna/teleqna.py",
    "three_gpp/three_gpp.py",
]

# Map task file names to GSMA column names
TASK_TO_COLUMN = {
    "teleqna": "teleqna",
    "telelogs": "telelogs",
    "telemath": "telemath",
    "three_gpp": "3gpp_tsg",
}

# Map provider prefixes to display names
PROVIDER_NAMES = {
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
}


class Stage(Enum):
    """Current stage of the run-evals flow."""

    INIT = "init"
    MINI_TEST = "mini_test"
    STRESS_TEST = "stress_test"
    READY = "ready"
    RUNNING_EVAL = "running_eval"
    EXPORTING = "exporting"
    ERROR = "error"
    COMPLETE = "complete"


class ChecklistItem(Static):
    """A checklist item with animated progress indicator."""

    status = reactive("pending")  # pending, running, passed, failed
    dot_count = reactive(0)  # For "cooking." animation: 0, 1, 2
    score: reactive[float | None] = reactive(None)  # Optional score to display

    def __init__(self, label: str, step_id: str) -> None:
        super().__init__()
        self.label = label
        self.step_id = step_id

    def render(self) -> str:
        score_text = ""
        if self.score is not None:
            score_text = f"  [#8b949e]score: {self.score:.2f}[/]"

        if self.status == "pending":
            return f"  [#484f58][ ][/] [#8b949e]{self.label}[/]"
        elif self.status == "running":
            dots = "." * (self.dot_count + 1)
            padding = " " * (3 - self.dot_count - 1)
            return f"  [#f0883e][◐][/] [#f0f6fc]{self.label}[/]  [#f0883e]cooking{dots}{padding}[/]"
        elif self.status == "passed":
            return f"  [#3fb950][✓][/] [#f0f6fc]{self.label}[/]{score_text}"
        elif self.status == "failed":
            return f"  [#f85149][✗][/] [#f0f6fc]{self.label}[/]"
        return f"  [#484f58][ ][/] [#8b949e]{self.label}[/]"


class RunEvalsScreen(Screen[None]):
    """Screen for running evaluations with 3-step checklist."""

    DEFAULT_CSS = """
    RunEvalsScreen {
        background: #0d1117;
        padding: 2 4;
        layout: vertical;
    }

    #header {
        color: #a61d2d;
        text-style: bold;
        padding: 1 0 2 0;
        height: auto;
    }

    #model-info {
        color: #8b949e;
        padding: 0 2 1 2;
        height: auto;
    }

    #checklist-container {
        width: 100%;
        max-width: 60;
        height: auto;
        padding: 1 2;
    }

    #checklist {
        height: auto;
        padding: 0;
    }

    ChecklistItem {
        height: 1;
        padding: 0;
        background: transparent;
    }

    #error-message {
        color: #f85149;
        padding: 1 2;
        height: auto;
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
        Binding("q", "cancel", "Cancel/Back"),
        Binding("escape", "cancel", "Cancel/Back"),
        Binding("enter", "confirm", "Confirm", show=False),
    ]

    stage = reactive(Stage.INIT)

    def __init__(self) -> None:
        super().__init__()
        self.env_manager = EnvManager()
        self.model = self.env_manager.get("INSPECT_EVAL_MODEL") or ""
        self.tasks = ALL_TASKS
        self._animation_timer = None

    def compose(self) -> ComposeResult:
        yield Static("run-evals", id="header")
        model_display = self.model if self.model else "not-configured"
        yield Static(
            f"model: [#f0f6fc]{model_display}[/]", id="model-info", markup=True
        )
        with Container(id="checklist-container"):
            with Vertical(id="checklist"):
                yield ChecklistItem("mini-open-telco", "mini_test")
                yield ChecklistItem("stress-testing", "stress_test")
                yield ChecklistItem("go", "ready")
        yield Static("", id="error-message", markup=True)
        yield Static("", id="spacer")
        yield Static("[#8b949e]q[/] cancel", id="footer", markup=True)

    def on_mount(self) -> None:
        """Start the preflight checks."""
        # Start animation timer
        self._animation_timer = self.set_interval(0.4, self._animate_dots)

        if not self.model:
            self._show_error("no-model-configured. use set-model first.")
            return

        # Start the step sequence
        self._run_steps()

    def _animate_dots(self) -> None:
        """Animate the cooking... dots for running items."""
        for item in self.query(ChecklistItem):
            if item.status == "running":
                item.dot_count = (item.dot_count + 1) % 3

    def _set_step_status(self, step_id: str, status: str) -> None:
        """Set the status of a checklist step."""
        for item in self.query(ChecklistItem):
            if item.step_id == step_id:
                item.status = status
                break

    def _set_step_score(self, step_id: str, score: float) -> None:
        """Set the score for a checklist step."""
        for item in self.query(ChecklistItem):
            if item.step_id == step_id:
                item.score = score
                break

    def _show_error(self, message: str) -> None:
        """Show error message and update UI."""
        self.stage = Stage.ERROR
        self.query_one("#error-message", Static).update(f"[#f85149]{message}[/]")
        self.query_one("#footer", Static).update("[#8b949e]q[/] back")

    def _show_ready(self) -> None:
        """Show ready state - all checks passed."""
        self.stage = Stage.READY
        self.query_one("#footer", Static).update(
            "[#8b949e]enter[/] run-full-eval [#30363d]|[/] [#8b949e]q[/] cancel"
        )

    @work(exclusive=True, thread=True)
    def _run_steps(self) -> None:
        """Run all preflight steps sequentially."""
        # Step 1: Mini Open Telco test
        self.app.call_from_thread(self._set_step_status, "mini_test", "running")
        self.app.call_from_thread(self._set_stage, Stage.MINI_TEST)

        result = self._run_mini_test()
        if not result["passed"]:
            self.app.call_from_thread(self._set_step_status, "mini_test", "failed")
            self.app.call_from_thread(self._show_error, result["error"])
            return

        self.app.call_from_thread(self._set_step_status, "mini_test", "passed")
        if result.get("score") is not None:
            self.app.call_from_thread(self._set_step_score, "mini_test", result["score"])

        # Step 2: Stress-testing
        self.app.call_from_thread(self._set_step_status, "stress_test", "running")
        self.app.call_from_thread(self._set_stage, Stage.STRESS_TEST)

        result = self._run_stress_test()
        if not result["passed"]:
            self.app.call_from_thread(self._set_step_status, "stress_test", "failed")
            self.app.call_from_thread(self._show_error, result["error"])
            return

        self.app.call_from_thread(self._set_step_status, "stress_test", "passed")

        # Step 3: Ready for full benchmark
        self.app.call_from_thread(self._set_step_status, "ready", "passed")
        self.app.call_from_thread(self._show_ready)

    def _set_stage(self, stage: Stage) -> None:
        """Set the current stage."""
        self.stage = stage

    def _run_mini_test(self) -> dict:
        """Run mini Open Telco test via subprocess with --limit 1."""
        import subprocess
        from pathlib import Path

        # Find src/open_telco directory (where tasks are located)
        # Path: cli/screens/run_evals/run_evals_screen.py -> src/open_telco
        open_telco_dir = Path(__file__).parent.parent.parent.parent

        cmd = [
            "uv",
            "run",
            "inspect",
            "eval",
            *self.tasks,  # All 4 tasks
            "--model",
            self.model,
            "--limit",
            "1",
            "--log-dir",
            "logs/preflight",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=open_telco_dir,
            )

            if result.returncode == 0:
                # Parse score from output
                score = self._parse_score(result.stdout + result.stderr)
                return {"passed": True, "score": score}
            else:
                error_output = result.stderr or result.stdout or "Eval failed"
                # Get last meaningful line for error display
                error_lines = [
                    line for line in error_output.strip().split("\n") if line.strip()
                ]
                error_msg = error_lines[-1] if error_lines else "Eval failed"
                return {"passed": False, "error": error_msg}

        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Mini test timed out after 120s"}
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _parse_score(self, output: str) -> float | None:
        """Parse average accuracy score from inspect output."""
        import re

        # Match patterns like "accuracy: 0.85" or "accuracy=0.85" or "accuracy 0.85"
        matches = re.findall(r"accuracy[=:\s]+([0-9.]+)", output, re.IGNORECASE)
        if matches:
            return sum(float(m) for m in matches) / len(matches)
        return None

    def _run_stress_test(self) -> dict:
        """Run stress tests with edge case prompts."""
        from open_telco.cli.preflight.stress_test import run_stress_tests_sync

        try:
            result = run_stress_tests_sync(self.model)

            if result.passed:
                return {"passed": True}
            else:
                return {"passed": False, "error": result.error or "Stress test failed"}
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _start_full_eval(self) -> None:
        """Start the full evaluation."""
        self.stage = Stage.RUNNING_EVAL
        self.query_one("#error-message", Static).update(
            "[#8b949e]running-full-evaluation...[/]"
        )
        self.query_one("#footer", Static).update("[#8b949e]q[/] cancel-unsafe")
        self._run_full_eval()

    def _get_open_telco_dir(self) -> Path:
        """Get the open_telco source directory path."""
        return Path(__file__).parent.parent.parent.parent

    @work(exclusive=True, thread=True)
    def _run_full_eval(self) -> None:
        """Run full evaluation via subprocess in background."""
        import subprocess

        open_telco_dir = self._get_open_telco_dir()

        cmd = [
            "uv",
            "run",
            "inspect",
            "eval",
            *self.tasks,
            "--model",
            self.model,
            "--limit",
            "1",
            "--log-dir",
            "logs/leaderboard",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for full eval
                cwd=open_telco_dir,
            )

            if result.returncode == 0:
                self.app.call_from_thread(self._export_and_show_results)
            else:
                error_output = result.stderr or result.stdout or "Eval failed"
                error_lines = [
                    line for line in error_output.strip().split("\n") if line.strip()
                ]
                error_msg = error_lines[-1] if error_lines else "Eval failed"
                self.app.call_from_thread(self._show_error, f"Evaluation failed: {error_msg}")

        except subprocess.TimeoutExpired:
            self.app.call_from_thread(self._show_error, "Evaluation timed out after 10 minutes")
        except Exception as e:
            self.app.call_from_thread(self._show_error, f"Evaluation failed: {e}")

    def _export_and_show_results(self) -> None:
        """Export results to parquet and show preview."""
        self.stage = Stage.EXPORTING
        self.query_one("#error-message", Static).update(
            "[#8b949e]exporting-results...[/]"
        )
        self._do_export()

    @work(exclusive=True, thread=True)
    def _do_export(self) -> None:
        """Export results to parquet in background thread."""
        try:
            open_telco_dir = self._get_open_telco_dir()
            log_dir = open_telco_dir / "logs" / "leaderboard"
            output_path = log_dir / "results.parquet"

            if not log_dir.exists():
                raise FileNotFoundError(f"Log directory not found: {log_dir}")

            df = self._export_to_leaderboard_parquet(str(log_dir), str(output_path))

            preview = self._format_results_preview(df)
            self.app.call_from_thread(
                self._on_export_success, preview, str(output_path.relative_to(open_telco_dir))
            )

        except Exception as e:
            self.app.call_from_thread(self._show_error, f"Export failed: {e}")

    def _parse_model_display_name(self, model_str: str) -> str:
        """Parse model string to display format: 'model_name (Provider)'."""
        parts = model_str.split("/")

        if len(parts) >= 3:
            # Format: router/provider/model (e.g., openrouter/openai/gpt-5.2)
            provider = parts[1]
            model_name = "/".join(parts[2:])
        elif len(parts) == 2:
            # Format: provider/model (e.g., openai/gpt-4o)
            provider = parts[0]
            model_name = parts[1]
        else:
            provider = "unknown"
            model_name = model_str

        provider_display = PROVIDER_NAMES.get(provider.lower(), provider.title())
        return f"{model_name} ({provider_display})"

    def _export_to_leaderboard_parquet(self, log_dir: str, output_path: str) -> pd.DataFrame:
        """Convert inspect eval logs to GSMA leaderboard parquet format."""
        df = evals_df(log_dir)

        if df.empty:
            raise ValueError(f"No eval logs found in {log_dir}")

        results = []
        models = df["model"].unique()

        for model in models:
            model_df = df[df["model"] == model]

            row = {
                "model": self._parse_model_display_name(model),
                "teleqna": None,
                "telelogs": None,
                "telemath": None,
                "3gpp_tsg": None,
                "date": date.today().isoformat(),
            }

            for _, eval_row in model_df.iterrows():
                task_name = eval_row.get("task_name", "")

                task_id = None
                for key in TASK_TO_COLUMN:
                    if key in task_name.lower():
                        task_id = key
                        break

                if task_id is None:
                    continue

                column_name = TASK_TO_COLUMN[task_id]

                score = eval_row.get("score_headline_value")
                stderr = eval_row.get("score_headline_stderr")
                n_samples = eval_row.get("samples_completed", eval_row.get("samples", 0))

                if score is not None:
                    score_val = float(score) * 100 if float(score) <= 1.0 else float(score)
                    stderr_val = (
                        float(stderr) * 100 if stderr and float(stderr) <= 1.0 else (float(stderr) if stderr else 0.0)
                    )
                    n_samples_val = float(n_samples) if n_samples else 0.0
                    row[column_name] = [score_val, stderr_val, n_samples_val]

            results.append(row)

        result_df = pd.DataFrame(results)
        column_order = ["model", "teleqna", "telelogs", "telemath", "3gpp_tsg", "date"]
        result_df = result_df[column_order]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output_path, index=False)

        return result_df

    def _format_results_preview(self, df: pd.DataFrame) -> str:
        """Format results DataFrame as a human-readable preview string."""
        if df.empty:
            return "No results to display"

        lines = []
        for _, row in df.iterrows():
            lines.append(f"model: {row['model']}")
            lines.append("")

            for col, display in [
                ("teleqna", "teleqna"),
                ("telelogs", "telelogs"),
                ("telemath", "telemath"),
                ("3gpp_tsg", "3gpp_tsg"),
            ]:
                val = row.get(col)
                if val is not None and isinstance(val, list) and len(val) >= 2:
                    score, stderr = val[0], val[1]
                    lines.append(f"  {display:10} {score:6.2f} ± {stderr:.2f}")
                else:
                    lines.append(f"  {display:10} --")

            lines.append("")

        return "\n".join(lines)

    def _on_export_success(self, preview: str, output_path: str) -> None:
        """Handle successful export completion."""
        self.stage = Stage.COMPLETE

        # Format the preview with colors
        preview_lines = preview.strip().split("\n")
        formatted_lines = []
        for line in preview_lines:
            if line.startswith("model:"):
                formatted_lines.append(f"[#f0f6fc]{line}[/]")
            elif "±" in line:
                formatted_lines.append(f"[#8b949e]{line}[/]")
            else:
                formatted_lines.append(line)

        formatted_preview = "\n".join(formatted_lines)

        self.query_one("#error-message", Static).update(
            f"[#3fb950]evaluation-complete![/]\n\n{formatted_preview}\n"
            f"[#8b949e]saved: {output_path}[/]"
        )
        self.query_one("#footer", Static).update(
            "[#8b949e]enter[/] done [#30363d]|[/] [#8b949e]q[/] back"
        )

    def _on_eval_success(self) -> None:
        """Handle successful evaluation completion (legacy, now uses _export_and_show_results)."""
        self.stage = Stage.COMPLETE
        self.query_one("#error-message", Static).update(
            "[#3fb950]evaluation-complete! results-saved-to logs/leaderboard/[/]"
        )
        self.query_one("#footer", Static).update(
            "[#8b949e]enter[/] done [#30363d]|[/] [#8b949e]q[/] back"
        )

    def action_cancel(self) -> None:
        """Cancel and go back."""
        if self._animation_timer:
            self._animation_timer.stop()
        self.app.pop_screen()

    def action_confirm(self) -> None:
        """Handle enter key based on current stage."""
        if self.stage == Stage.READY:
            self._start_full_eval()
        elif self.stage == Stage.COMPLETE:
            if self._animation_timer:
                self._animation_timer.stop()
            self.app.pop_screen()
