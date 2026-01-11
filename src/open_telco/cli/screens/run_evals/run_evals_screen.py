"""Run-evals screen with checklist UI."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from datetime import date
from enum import Enum
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import evals_df
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Static

from open_telco.cli.config import EnvManager

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

# Map task paths to display names for selection UI
TASK_DISPLAY_NAMES = {
    "telelogs/telelogs.py": "telelogs",
    "telemath/telemath.py": "telemath",
    "teleqna/teleqna.py": "teleqna",
    "three_gpp/three_gpp.py": "3gpp_tsg",
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

    # Progress circle animation frames: empty → quarter → half → three-quarter → full
    PROGRESS_FRAMES = ["○", "◔", "◑", "◕", "●"]

    status = reactive("pending")  # pending, running, passed, failed
    dot_count = reactive(0)  # For animation cycle: 0-4 (circle frames)
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
            frame = self.PROGRESS_FRAMES[self.dot_count % 5]
            dots = "." * ((self.dot_count % 3) + 1)
            padding = " " * (2 - (self.dot_count % 3))
            return f"  [#f0883e][{frame}][/] [#f0f6fc]{self.label}[/]  [#f0883e]cooking{dots}{padding}[/]"
        elif self.status == "passed":
            return f"  [#3fb950][✓][/] [#f0f6fc]{self.label}[/]{score_text}"
        elif self.status == "failed":
            return f"  [#f85149][✗][/] [#f0f6fc]{self.label}[/]"
        return f"  [#484f58][ ][/] [#8b949e]{self.label}[/]"


class TaskChecklistItem(Static):
    """A selectable task checklist item with [X] checkbox in GSMA_RED."""

    selected = reactive(True)  # Default all selected
    highlighted = reactive(False)

    def __init__(self, task_name: str, display_name: str, item_id: str) -> None:
        super().__init__(id=item_id)
        self.task_name = task_name  # e.g., "telelogs/telelogs.py"
        self.display_name = display_name  # e.g., "telelogs"

    def render(self) -> str:
        # Use GSMA_RED (#a61d2d) for the [X] checkbox
        checkbox = "[#a61d2d][X][/]" if self.selected else "[#484f58][ ][/]"
        if self.highlighted:
            return f"  {checkbox} [bold #f0f6fc]{self.display_name}[/]"
        return f"  {checkbox} [#8b949e]{self.display_name}[/]"

    def toggle(self) -> None:
        self.selected = not self.selected


class TaskSelectScreen(Screen[list[str] | None]):
    """Screen for selecting which tasks to run."""

    DEFAULT_CSS = """
    TaskSelectScreen {
        padding: 0 4;
        layout: vertical;
    }

    #header {
        color: #a61d2d;
        text-style: bold;
        padding: 0 0 2 0;
        height: auto;
    }

    #model-info {
        color: #8b949e;
        padding: 0 2 1 2;
        height: auto;
    }

    #task-header {
        color: #8b949e;
        padding: 1 2 0 2;
        height: auto;
    }

    #task-list {
        height: auto;
        padding: 0 2;
    }

    TaskChecklistItem {
        height: 1;
        padding: 0;
        background: transparent;
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
        Binding("space", "toggle_task", "Toggle Task", show=False),
        Binding("up", "move_up", "Move Up", show=False),
        Binding("down", "move_down", "Move Down", show=False),
        Binding("k", "move_up", "Move Up", show=False),
        Binding("j", "move_down", "Move Down", show=False),
    ]

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model
        self._selected_index: int = 0

    def compose(self) -> ComposeResult:
        yield Static("run-evals", id="header")
        yield Static(
            f"model: [#f0f6fc]{self.model}[/]", id="model-info", markup=True
        )
        yield Static("[#8b949e]select tasks to run:[/]", id="task-header", markup=True)
        with Vertical(id="task-list"):
            for i, task in enumerate(ALL_TASKS):
                display_name = TASK_DISPLAY_NAMES.get(task, task)
                item = TaskChecklistItem(task, display_name, f"task_{i}")
                if i == 0:
                    item.highlighted = True
                yield item
        yield Static("", id="spacer")
        yield Static(
            "[#8b949e]space[/] toggle [#30363d]|[/] "
            "[#8b949e]enter[/] run-selected [#30363d]|[/] "
            "[#8b949e]q[/] cancel",
            id="footer",
            markup=True,
        )

    def _get_task_items(self) -> list[TaskChecklistItem]:
        """Get all task checklist items."""
        return list(self.query(TaskChecklistItem))

    def _update_highlight(self) -> None:
        """Update which task is highlighted."""
        items = self._get_task_items()
        for i, item in enumerate(items):
            item.highlighted = i == self._selected_index

    def action_move_up(self) -> None:
        """Move task selection up."""
        items = self._get_task_items()
        if items and self._selected_index > 0:
            self._selected_index -= 1
            self._update_highlight()

    def action_move_down(self) -> None:
        """Move task selection down."""
        items = self._get_task_items()
        if items and self._selected_index < len(items) - 1:
            self._selected_index += 1
            self._update_highlight()

    def action_toggle_task(self) -> None:
        """Toggle selection of current task."""
        items = self._get_task_items()
        if items and 0 <= self._selected_index < len(items):
            items[self._selected_index].toggle()

    def action_confirm(self) -> None:
        """Confirm selection and return selected tasks."""
        items = self._get_task_items()
        selected = [item.task_name for item in items if item.selected]
        if not selected:
            self.notify("select at least one task", title="warning")
            return
        self.dismiss(selected)

    def action_cancel(self) -> None:
        """Cancel and return None."""
        self.dismiss(None)


class RunEvalsScreen(Screen[None]):
    """Screen for running evaluations with 3-step checklist."""

    DEFAULT_CSS = """
    RunEvalsScreen {
        padding: 0 4;
        layout: vertical;
    }

    #header {
        color: #a61d2d;
        text-style: bold;
        padding: 0 0 2 0;
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

    #viewer-url {
        color: #58a6ff;
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
        self._animation_timer: Timer | None = None
        self._current_process: subprocess.Popen[str] | None = None
        self._viewer_process: subprocess.Popen[str] | None = None
        self._cancelled: bool = False
        self._full_eval_dot_count: int = 0

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
        yield Static("", id="viewer-url", markup=True)
        yield Static("", id="spacer")
        yield Static("[#8b949e]q[/] cancel", id="footer", markup=True)

    def on_mount(self) -> None:
        """Start the preflight checks."""
        # Start animation timer
        self._animation_timer = self.set_interval(0.6, self._animate_dots)

        if not self.model:
            self._show_error("no-model-configured. use set-model first.")
            return

        # Start the step sequence
        self._run_steps()

    def on_unmount(self) -> None:
        """Clean up when screen is removed."""
        if self._animation_timer:
            self._animation_timer.stop()
            self._animation_timer = None
        # Kill any running subprocess
        self._kill_current_process()
        # Stop the viewer subprocess
        self._stop_viewer()

    def _kill_current_process(self) -> None:
        """Terminate and kill any running subprocess and its process group."""
        if self._current_process is not None:
            try:
                # Try to kill the entire process group first
                try:
                    pgid = os.getpgid(self._current_process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    # Process group doesn't exist or already dead, try direct terminate
                    self._current_process.terminate()

                self._current_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Process didn't respond to SIGTERM, escalate to SIGKILL
                try:
                    pgid = os.getpgid(self._current_process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    self._current_process.kill()
                self._current_process.wait()
            except Exception:
                pass  # Process may have already exited
            finally:
                self._current_process = None

    def _start_viewer(self) -> None:
        """Start inspect view subprocess for live monitoring."""
        import time

        if self._viewer_process is not None:
            return  # Already running

        open_telco_dir = self._get_open_telco_dir()

        # Ensure log directory exists
        log_dir = open_telco_dir / "logs" / "leaderboard"
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv",
            "run",
            "inspect",
            "view",
            "start",
            "--log-dir",
            "logs/leaderboard",
            "--host",
            "127.0.0.1",
            "--port",
            "7575",
        ]

        try:
            self._viewer_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=open_telco_dir,
                start_new_session=True,
            )

            # Brief delay to check if process started successfully
            time.sleep(0.3)

            # Check if process exited immediately (indicates failure)
            if self._viewer_process.poll() is not None:
                # Process already exited - read error
                _, stderr = self._viewer_process.communicate()
                error_msg = stderr.strip().split("\n")[-1] if stderr else "Unknown error"
                self._viewer_process = None
                self.query_one("#viewer-url", Static).update(
                    f"[#f85149]viewer failed: {error_msg}[/]"
                )
                return

            # Update UI with viewer URL
            self.query_one("#viewer-url", Static).update(
                "[#58a6ff]view live at:[/] [#f0f6fc]http://127.0.0.1:7575[/]"
            )
        except Exception as e:
            self._viewer_process = None
            self.query_one("#viewer-url", Static).update(
                f"[#f85149]viewer failed: {e}[/]"
            )

    def _stop_viewer(self) -> None:
        """Stop the inspect view subprocess."""
        if self._viewer_process is not None:
            try:
                try:
                    pgid = os.getpgid(self._viewer_process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    self._viewer_process.terminate()

                try:
                    self._viewer_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        pgid = os.getpgid(self._viewer_process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        self._viewer_process.kill()
                    self._viewer_process.wait()
            except (ProcessLookupError, OSError):
                pass
            finally:
                self._viewer_process = None

    def _render_full_eval_status(self) -> str:
        """Render the animated running-full-evaluation status text."""
        dots = "." * ((self._full_eval_dot_count % 3) + 1)
        padding = " " * (3 - len(dots))
        return f"[#8b949e]running-full-evaluation{dots}{padding}[/]"

    def _animate_dots(self) -> None:
        """Animate the cooking... dots for running items and full eval status."""
        has_running = False
        for item in self.query(ChecklistItem):
            if item.status == "running":
                has_running = True
                item.dot_count = (item.dot_count + 1) % 5

        # Animate full eval status when in RUNNING_EVAL stage
        if self.stage == Stage.RUNNING_EVAL:
            has_running = True
            self._full_eval_dot_count = (self._full_eval_dot_count + 1) % 3
            self.query_one("#error-message", Static).update(
                self._render_full_eval_status()
            )

        # Pause timer when nothing is animating to save CPU
        if not has_running and self._animation_timer:
            self._animation_timer.pause()

    def _set_step_status(self, step_id: str, status: str) -> None:
        """Set the status of a checklist step."""
        for item in self.query(ChecklistItem):
            if item.step_id == step_id:
                item.status = status
                # Resume timer when a step starts running
                if status == "running" and self._animation_timer:
                    self._animation_timer.resume()
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
        """Show ready state - push task selection screen."""
        self.stage = Stage.READY
        self.app.push_screen(TaskSelectScreen(self.model), self._on_task_selection)

    def _on_task_selection(self, selected: list[str] | None) -> None:
        """Handle task selection result from TaskSelectScreen."""
        if selected is None:
            # User cancelled - go back to main menu
            self.app.pop_screen()
            return
        self.tasks = selected
        self._start_full_eval()

    def _check_preflight_passed(self) -> bool:
        """Check if model has already passed mini-open-telco preflights.

        Scans logs/preflight/*.json for files where:
        - eval.model matches current INSPECT_EVAL_MODEL
        - status == "success"
        - All 4 tasks have passing logs

        Returns:
            True if all 4 task preflights passed for this model
        """
        open_telco_dir = self._get_open_telco_dir()
        preflight_dir = open_telco_dir / "logs" / "preflight"

        if not preflight_dir.exists():
            return False

        # Required tasks to find
        required_tasks = {"telelogs", "telemath", "teleqna", "three_gpp"}
        found_tasks: set[str] = set()

        for json_file in preflight_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Check if this log matches our model and status is success
                if data.get("status") != "success":
                    continue

                eval_info = data.get("eval", {})
                log_model = eval_info.get("model", "")

                if log_model != self.model:
                    continue

                # Extract task name from task field
                task_name = eval_info.get("task", "")
                if task_name in required_tasks:
                    found_tasks.add(task_name)

            except (json.JSONDecodeError, OSError):
                continue

        return found_tasks == required_tasks

    @work(exclusive=True, thread=True)
    def _run_steps(self) -> None:
        """Run all preflight steps sequentially."""
        try:
            if self._cancelled:
                return

            # Check if preflights already passed for this model
            preflight_passed = self._check_preflight_passed()

            if preflight_passed:
                # Skip mini-test and stress-test, mark them as passed
                self.app.call_from_thread(self._set_step_status, "mini_test", "passed")
                self.app.call_from_thread(self._set_step_status, "stress_test", "passed")
                self.app.call_from_thread(self._set_step_status, "ready", "passed")
                self.app.call_from_thread(self._show_ready)
                return

            # Step 1: Mini Open Telco test
            self.app.call_from_thread(self._set_step_status, "mini_test", "running")
            self.app.call_from_thread(self._set_stage, Stage.MINI_TEST)

            result = self._run_mini_test()
            if self._cancelled:
                return
            if not result["passed"]:
                self.app.call_from_thread(self._set_step_status, "mini_test", "failed")
                self.app.call_from_thread(self._show_error, result["error"])
                return

            self.app.call_from_thread(self._set_step_status, "mini_test", "passed")
            if result.get("score") is not None:
                self.app.call_from_thread(self._set_step_score, "mini_test", result["score"])

            if self._cancelled:
                return

            # Step 2: Stress-testing
            self.app.call_from_thread(self._set_step_status, "stress_test", "running")
            self.app.call_from_thread(self._set_stage, Stage.STRESS_TEST)

            result = self._run_stress_test()
            if self._cancelled:
                return
            if not result["passed"]:
                self.app.call_from_thread(self._set_step_status, "stress_test", "failed")
                self.app.call_from_thread(self._show_error, result["error"])
                return

            self.app.call_from_thread(self._set_step_status, "stress_test", "passed")

            if self._cancelled:
                return

            # Step 3: Ready for full benchmark
            self.app.call_from_thread(self._set_step_status, "ready", "passed")
            self.app.call_from_thread(self._show_ready)
        except Exception:
            # App may have been closed while worker was running
            pass

    def _set_stage(self, stage: Stage) -> None:
        """Set the current stage."""
        self.stage = stage

    def _run_mini_test(self) -> dict:
        """Run mini Open Telco test via subprocess with --limit 1."""
        if self._cancelled:
            return {"passed": False, "error": "cancelled"}

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
            "--log-format",
            "json",
        ]

        try:
            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=open_telco_dir,
                start_new_session=True,  # Create new process group for clean termination
            )

            try:
                stdout, stderr = self._current_process.communicate(timeout=120)
                returncode = self._current_process.returncode
            except subprocess.TimeoutExpired:
                self._current_process.kill()
                self._current_process.wait()
                return {"passed": False, "error": "Mini test timed out after 120s"}
            finally:
                self._current_process = None

            if self._cancelled:
                return {"passed": False, "error": "cancelled"}

            if returncode == 0:
                # Parse score from output
                score = self._parse_score(stdout + stderr)
                return {"passed": True, "score": score}
            else:
                error_output = stderr or stdout or "Eval failed"
                # Get last meaningful line for error display
                error_lines = [
                    line for line in error_output.strip().split("\n") if line.strip()
                ]
                error_msg = error_lines[-1] if error_lines else "Eval failed"
                return {"passed": False, "error": error_msg}

        except Exception as e:
            self._current_process = None
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

        # Ensure log directory exists for viewer
        open_telco_dir = self._get_open_telco_dir()
        log_dir = open_telco_dir / "logs" / "leaderboard"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Start viewer immediately so URL is visible
        self._start_viewer()

        # Reset animation counter and resume timer for full eval animation
        self._full_eval_dot_count = 0
        if self._animation_timer:
            self._animation_timer.resume()

        self.query_one("#error-message", Static).update(
            self._render_full_eval_status()
        )
        self.query_one("#footer", Static).update("[#8b949e]q[/] cancel-unsafe")
        self._run_full_eval()

    def _get_open_telco_dir(self) -> Path:
        """Get the open_telco source directory path."""
        return Path(__file__).parent.parent.parent.parent

    @work(exclusive=True, thread=True)
    def _run_full_eval(self) -> None:
        """Run full evaluation via subprocess in background."""
        if self._cancelled:
            return

        open_telco_dir = self._get_open_telco_dir()

        cmd = [
            "uv",
            "run",
            "inspect",
            "eval",
            *self.tasks,
            "--model",
            self.model,
            "--log-dir",
            "logs/leaderboard",
            "--log-format",
            "json",
        ]

        try:
            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=open_telco_dir,
                start_new_session=True,  # Create new process group for clean termination
            )

            try:
                stdout, stderr = self._current_process.communicate(timeout=3600)
                returncode = self._current_process.returncode
            except subprocess.TimeoutExpired:
                self._current_process.kill()
                self._current_process.wait()
                try:
                    self.app.call_from_thread(
                        self._show_error, "Evaluation timed out after 1 hour"
                    )
                except Exception:
                    pass
                return
            finally:
                self._current_process = None

            if self._cancelled:
                return

            if returncode == 0:
                self.app.call_from_thread(self._export_and_show_results)
            else:
                error_output = stderr or stdout or "Eval failed"
                error_lines = [
                    line for line in error_output.strip().split("\n") if line.strip()
                ]
                error_msg = error_lines[-1] if error_lines else "Eval failed"
                self.app.call_from_thread(
                    self._show_error, f"Evaluation failed: {error_msg}"
                )

        except Exception as e:
            self._current_process = None
            try:
                self.app.call_from_thread(self._show_error, f"Evaluation failed: {e}")
            except Exception:
                pass  # App may have been closed

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
            try:
                self.app.call_from_thread(self._show_error, f"Export failed: {e}")
            except Exception:
                pass  # App may have been closed

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

                # Get n_samples from dataset_sample_ids (the actual samples evaluated)
                # dataset_sample_ids is stored as a JSON string like "[1, 2, 3, ...]"
                dataset_sample_ids = eval_row.get("dataset_sample_ids", "[]")
                if isinstance(dataset_sample_ids, str):
                    import json
                    try:
                        sample_ids = json.loads(dataset_sample_ids)
                        n_samples = len(sample_ids) if isinstance(sample_ids, list) else 0
                    except (json.JSONDecodeError, TypeError):
                        n_samples = 0
                elif hasattr(dataset_sample_ids, "__len__"):
                    n_samples = len(dataset_sample_ids)
                else:
                    n_samples = eval_row.get("completed_samples", eval_row.get("total_samples", 0))

                if pd.notna(score):
                    score_val = float(score) * 100 if float(score) <= 1.0 else float(score)
                    stderr_val = (
                        float(stderr) * 100 if pd.notna(stderr) and float(stderr) <= 1.0 else (float(stderr) if pd.notna(stderr) else 0.0)
                    )
                    n_samples_val = float(n_samples) if pd.notna(n_samples) else 0.0
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
        """Cancel and go back, killing any running process."""
        self._cancelled = True
        self._kill_current_process()
        self.app.pop_screen()

    def action_confirm(self) -> None:
        """Handle enter key based on current stage."""
        if self.stage == Stage.COMPLETE:
            # Timer cleanup is handled by on_unmount
            self.app.pop_screen()
