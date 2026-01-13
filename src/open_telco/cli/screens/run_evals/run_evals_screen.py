"""Run-evals screen with checklist UI."""

from __future__ import annotations

import json
import re
import signal
import subprocess
from datetime import date
from enum import Enum
from pathlib import Path

import pandas as pd
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Static

from open_telco.cli.config import EnvManager
from open_telco.cli.constants import (
    ALL_TASKS,
    PROVIDER_DISPLAY_NAMES,
    TASK_DISPLAY_NAMES,
    TASK_TO_COLUMN,
    Animation,
    Colors,
    Ports,
    Timeouts,
)
from open_telco.cli.types import FindKResult, StepResult
from open_telco.cli.utils.model_parser import parse_model_string
from open_telco.cli.utils.process import (
    communicate_with_timeout,
    get_process_group,
    kill_process_group,
    start_process,
)

# Re-export for test compatibility
PROVIDER_NAMES = PROVIDER_DISPLAY_NAMES

OPEN_TELCO_DIR = Path(__file__).parent.parent.parent.parent
DEFAULT_K = 1
REQUIRED_PREFLIGHT_TASKS = frozenset({"telelogs", "telemath", "teleqna", "three_gpp"})


class Stage(Enum):
    """Current stage of the run-evals flow."""

    INIT = "init"
    MINI_TEST = "mini_test"
    FIND_K = "find_k"
    READY = "ready"
    RUNNING_EVAL = "running_eval"
    EXPORTING = "exporting"
    ERROR = "error"
    COMPLETE = "complete"


class ChecklistItem(Static):
    """A checklist item with animated progress indicator."""

    PROGRESS_FRAMES = ("○", "◔", "◑", "◕", "●")

    status = reactive("pending")
    dot_count = reactive(0)
    score: reactive[float | None] = reactive(None)
    stderr: reactive[float | None] = reactive(None)
    variance_reduction: reactive[float | None] = reactive(None)

    def __init__(self, label: str, step_id: str) -> None:
        super().__init__()
        self.label = label
        self.step_id = step_id

    def render(self) -> str:
        if self.status == "running":
            return self._render_running()
        return self._render_static()

    def _render_running(self) -> str:
        frame = self.PROGRESS_FRAMES[self.dot_count % Animation.PROGRESS_CYCLE_LENGTH]
        dots = "." * ((self.dot_count % Animation.DOT_CYCLE_LENGTH) + 1)
        padding = " " * (2 - (self.dot_count % Animation.DOT_CYCLE_LENGTH))
        return f"  [{Colors.RED}][{frame}][/] [{Colors.TEXT_PRIMARY}]{self.label}[/]  [{Colors.RED}]cooking{dots}{padding}[/]"

    def _render_static(self) -> str:
        score_text = self._format_score_text()
        icon, icon_color, label_color = self._get_status_style()
        return f"  [{icon_color}]{icon}[/] [{label_color}]{self.label}[/]{score_text}"

    def _get_status_style(self) -> tuple[str, str, str]:
        if self.status == "passed":
            return "[✓]", Colors.SUCCESS, Colors.TEXT_PRIMARY
        if self.status == "failed":
            return "[✗]", Colors.ERROR, Colors.TEXT_PRIMARY
        return "[ ]", Colors.TEXT_DISABLED, Colors.TEXT_MUTED

    def _format_score_text(self) -> str:
        if self.score is None:
            return ""
        if self.step_id == "find_k" and self.variance_reduction is not None:
            return f"  [{Colors.TEXT_MUTED}]K={int(self.score)} for {self.variance_reduction:.0f}% variance reduction[/]"
        stderr_text = f" | std: {self.stderr:.2f}" if self.stderr is not None else ""
        return f"  [{Colors.TEXT_MUTED}]score: {self.score:.2f}{stderr_text}[/]"


class TaskChecklistItem(Static):
    """A selectable task checklist item."""

    selected = reactive(True)
    highlighted = reactive(False)

    def __init__(self, task_name: str, display_name: str, item_id: str) -> None:
        super().__init__(id=item_id)
        self.task_name = task_name
        self.display_name = display_name

    def render(self) -> str:
        checkbox = (
            f"[{Colors.RED}]●[/]" if self.selected else f"[{Colors.TEXT_DISABLED}]○[/]"
        )
        style = f"bold {Colors.TEXT_PRIMARY}" if self.highlighted else Colors.TEXT_MUTED
        return f"  {checkbox} [{style}]{self.display_name}[/]"

    def toggle(self) -> bool:
        self.selected = not self.selected
        return self.selected


class TaskSelectScreen(Screen[list[str] | None]):
    """Screen for selecting which tasks to run."""

    DEFAULT_CSS = f"""
    TaskSelectScreen {{
        padding: 0 4;
        layout: vertical;
    }}

    #header {{
        color: {Colors.RED};
        text-style: bold;
        padding: 0 0 1 0;
        height: auto;
    }}

    #model-info {{
        color: {Colors.TEXT_MUTED};
        padding: 0 2 0 2;
        height: auto;
    }}

    #task-header {{
        color: {Colors.TEXT_MUTED};
        padding: 0 2 0 2;
        height: auto;
    }}

    #task-list {{
        height: auto;
        padding: 0 2;
    }}

    TaskChecklistItem {{
        height: 1;
        padding: 0;
        background: transparent;
    }}

    #spacer {{
        height: 1fr;
    }}

    #footer {{
        dock: bottom;
        height: 1;
        color: {Colors.TEXT_DISABLED};
    }}
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
        self._selected_index = 0

    def compose(self) -> ComposeResult:
        yield Static("run-evals", id="header")
        yield Static(
            f"model: [{Colors.TEXT_PRIMARY}]{self.model}[/]",
            id="model-info",
            markup=True,
        )
        yield Static(
            f"[{Colors.TEXT_MUTED}]select evals to run:[/]",
            id="task-header",
            markup=True,
        )
        with Vertical(id="task-list"):
            for i, task in enumerate(ALL_TASKS):
                display_name = TASK_DISPLAY_NAMES.get(task, task)
                item = TaskChecklistItem(task, display_name, f"task_{i}")
                item.highlighted = i == 0
                yield item
        yield Static("", id="spacer")
        yield Static(
            f"[{Colors.TEXT_MUTED}]space[/] toggle [{Colors.BORDER}]|[/] "
            f"[{Colors.TEXT_MUTED}]enter[/] run-selected [{Colors.BORDER}]|[/] "
            f"[{Colors.TEXT_MUTED}]q[/] cancel",
            id="footer",
            markup=True,
        )

    def _get_task_items(self) -> list[TaskChecklistItem]:
        return list(self.query(TaskChecklistItem))

    def _update_highlight(self) -> list[TaskChecklistItem]:
        items = self._get_task_items()
        for i, item in enumerate(items):
            item.highlighted = i == self._selected_index
        return items

    def action_move_up(self) -> None:
        if self._selected_index <= 0:
            return
        self._selected_index -= 1
        self._update_highlight()

    def action_move_down(self) -> None:
        items = self._get_task_items()
        if self._selected_index >= len(items) - 1:
            return
        self._selected_index += 1
        self._update_highlight()

    def action_toggle_task(self) -> None:
        items = self._get_task_items()
        if not items:
            return
        if self._selected_index < 0 or self._selected_index >= len(items):
            return
        items[self._selected_index].toggle()

    def action_confirm(self) -> None:
        items = self._get_task_items()
        selected = [item.task_name for item in items if item.selected]
        if not selected:
            self.notify("select at least one task", title="warning")
            return
        self.dismiss(selected)

    def action_cancel(self) -> None:
        self.dismiss(None)


class EvalRunningScreen(Screen[None]):
    """Screen for running selected evaluations with animated progress."""

    DEFAULT_CSS = f"""
    EvalRunningScreen {{
        padding: 0 4;
        layout: vertical;
    }}

    #header {{
        color: {Colors.RED};
        text-style: bold;
        padding: 0 0 1 0;
        height: auto;
    }}

    #model-info {{
        color: {Colors.TEXT_MUTED};
        padding: 0 2 0 2;
        height: auto;
    }}

    #eval-header {{
        color: {Colors.TEXT_MUTED};
        padding: 0 2 0 2;
        height: auto;
    }}

    #eval-list {{
        height: auto;
        padding: 0 2;
    }}

    ChecklistItem {{
        height: 1;
        padding: 0;
        background: transparent;
    }}

    #viewer-url {{
        color: {Colors.LINK};
        padding: 1 2;
        height: auto;
    }}

    #error-message {{
        color: {Colors.ERROR};
        padding: 1 2;
        height: auto;
    }}

    #spacer {{
        height: 1fr;
    }}

    #footer {{
        dock: bottom;
        height: 1;
        color: {Colors.TEXT_DISABLED};
    }}
    """

    BINDINGS = [
        Binding("q", "cancel", "Cancel/Back"),
        Binding("escape", "cancel", "Cancel/Back"),
        Binding("enter", "confirm", "Confirm", show=False),
    ]

    def __init__(self, model: str, tasks: list[str], selected_k: int) -> None:
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.selected_k = selected_k
        self._animation_timer: Timer | None = None
        self._current_process: subprocess.Popen[str] | None = None
        self._viewer_process: subprocess.Popen[str] | None = None
        self._cancelled = False
        self._current_task_index = 0
        self._completed = False

    def compose(self) -> ComposeResult:
        yield Static("run-evals", id="header")
        yield Static(
            f"model: [{Colors.TEXT_PRIMARY}]{self.model}[/]",
            id="model-info",
            markup=True,
        )
        yield Static(
            f"[{Colors.TEXT_MUTED}]running evals:[/]", id="eval-header", markup=True
        )
        with Vertical(id="eval-list"):
            for task in self.tasks:
                display_name = TASK_DISPLAY_NAMES.get(task, task)
                yield ChecklistItem(display_name, task)
        yield Static("", id="viewer-url", markup=True)
        yield Static("", id="error-message", markup=True)
        yield Static("", id="spacer")
        yield Static(
            f"[{Colors.TEXT_MUTED}]q[/] cancel-unsafe", id="footer", markup=True
        )

    def on_mount(self) -> None:
        self._animation_timer = self.set_interval(
            Animation.INTERVAL_SECONDS, self._animate_dots
        )
        self._start_viewer()
        self._run_next_task()

    def on_unmount(self) -> None:
        self._stop_animation_timer()
        self._kill_current_process()
        self._stop_viewer()

    def _stop_animation_timer(self) -> None:
        if self._animation_timer is None:
            return
        self._animation_timer.stop()
        self._animation_timer = None

    def _kill_current_process(self) -> None:
        if self._current_process is None:
            return
        self._terminate_process(self._current_process)
        self._current_process = None

    def _stop_viewer(self) -> None:
        if self._viewer_process is None:
            return
        self._terminate_process(self._viewer_process)
        self._viewer_process = None

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        self._send_sigterm(process)
        if not self._wait_for_process(process, timeout=2):
            self._send_sigkill(process)
            self._wait_for_process(process, timeout=None)

    def _send_sigterm(self, process: subprocess.Popen[str]) -> None:
        pgid = get_process_group(process.pid)
        if pgid is None:
            process.terminate()
            return
        kill_process_group(pgid, signal.SIGTERM)

    def _send_sigkill(self, process: subprocess.Popen[str]) -> None:
        pgid = get_process_group(process.pid)
        if pgid is None:
            process.kill()
            return
        kill_process_group(pgid, signal.SIGKILL)

    def _wait_for_process(
        self, process: subprocess.Popen[str], timeout: int | None
    ) -> bool:
        try:
            process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False
        except (ProcessLookupError, OSError):
            return True

    def _start_viewer(self) -> bool:
        import time

        if self._viewer_process is not None:
            return True

        log_dir = OPEN_TELCO_DIR / "logs" / "leaderboard"
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
            str(Ports.INSPECT_VIEWER),
        ]

        try:
            self._viewer_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                cwd=OPEN_TELCO_DIR,
            )
            time.sleep(0.3)

            if self._viewer_process.poll() is not None:
                self._viewer_process = None
                return False

            self.query_one("#viewer-url", Static).update(
                f"[{Colors.LINK}]view live at:[/] [{Colors.TEXT_PRIMARY}]http://127.0.0.1:{Ports.INSPECT_VIEWER}[/]"
            )
            return True
        except Exception:
            self._viewer_process = None
            return False

    def _animate_dots(self) -> bool:
        animated = False
        for item in self.query(ChecklistItem):
            if item.status != "running":
                continue
            animated = True
            item.dot_count = (item.dot_count + 1) % Animation.PROGRESS_CYCLE_LENGTH
        return animated

    def _find_checklist_item(self, step_id: str) -> ChecklistItem | None:
        for item in self.query(ChecklistItem):
            if item.step_id == step_id:
                return item
        return None

    def _run_next_task(self) -> None:
        if self._cancelled:
            return
        if self._current_task_index >= len(self.tasks):
            self._on_all_tasks_complete()
            return

        task = self.tasks[self._current_task_index]
        item = self._find_checklist_item(task)
        if item:
            item.status = "running"
        self._run_single_task(task)

    @work(exclusive=True, thread=True)
    def _run_single_task(self, task: str) -> None:
        if self._cancelled:
            return

        log_dir = OPEN_TELCO_DIR / "logs" / "leaderboard"
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv",
            "run",
            "inspect",
            "eval",
            task,
            "--model",
            self.model,
            "--epochs",
            str(self.selected_k),
            "--log-dir",
            "logs/leaderboard",
            "--log-format",
            "json",
        ]

        process = start_process(cmd, cwd=OPEN_TELCO_DIR)
        if process is None:
            self.app.call_from_thread(
                self._on_task_failed, task, "Failed to start process"
            )
            return

        self._current_process = process
        stdout, stderr, timed_out = communicate_with_timeout(
            process, Timeouts.FULL_EVAL
        )
        self._current_process = None

        if timed_out:
            self.app.call_from_thread(self._on_task_failed, task, "Task timed out")
            return
        if self._cancelled:
            return
        if process.returncode != 0:
            error_msg = self._extract_last_error_line(stderr or stdout)
            self.app.call_from_thread(self._on_task_failed, task, error_msg)
            return

        score, stderr_val = self._parse_score_and_stderr(stdout + stderr)
        self.app.call_from_thread(self._on_task_complete, task, score, stderr_val)

    def _parse_score_and_stderr(self, output: str) -> tuple[float | None, float | None]:
        score_matches = re.findall(r"accuracy[=:\s]+([0-9.]+)", output, re.IGNORECASE)
        stderr_matches = re.findall(r"stderr[=:\s]+([0-9.]+)", output, re.IGNORECASE)

        score = None
        if score_matches:
            score = sum(float(m) for m in score_matches) / len(score_matches)

        stderr_val = None
        if stderr_matches:
            stderr_val = sum(float(m) for m in stderr_matches) / len(stderr_matches)

        return score, stderr_val

    def _extract_last_error_line(self, output: str) -> str:
        if not output:
            return "Eval failed"
        lines = [line for line in output.strip().split("\n") if line.strip()]
        if not lines:
            return "Eval failed"
        return lines[-1]

    def _on_task_complete(
        self, task: str, score: float | None, stderr_val: float | None = None
    ) -> None:
        item = self._find_checklist_item(task)
        if item:
            item.status = "passed"
            if score is not None:
                item.score = score
            if stderr_val is not None:
                item.stderr = stderr_val
        self._current_task_index += 1
        self._run_next_task()

    def _on_task_failed(self, task: str, error: str) -> None:
        item = self._find_checklist_item(task)
        if item:
            item.status = "failed"
        self.query_one("#error-message", Static).update(f"[{Colors.ERROR}]{error}[/]")
        self._current_task_index += 1
        self._run_next_task()

    def _on_all_tasks_complete(self) -> None:
        self._completed = True
        self._do_export()

    @work(exclusive=True, thread=True)
    def _do_export(self) -> None:
        try:
            log_dir = OPEN_TELCO_DIR / "logs" / "leaderboard"
            output_path = log_dir / "results.parquet"

            if not log_dir.exists():
                raise FileNotFoundError(f"Log directory not found: {log_dir}")

            self._export_to_leaderboard_parquet(str(log_dir), str(output_path))
            self.app.call_from_thread(
                self._on_export_success, str(output_path.relative_to(OPEN_TELCO_DIR))
            )

        except Exception as e:
            self.app.call_from_thread(self._on_export_error, str(e))

    def _export_to_leaderboard_parquet(
        self, log_dir: str, output_path: str
    ) -> pd.DataFrame:
        from inspect_ai.analysis import evals_df

        df = evals_df(log_dir)

        if df.empty:
            raise ValueError(f"No eval logs found in {log_dir}")

        results = []
        for model in df["model"].unique():
            row = self._build_model_row(df[df["model"] == model], model)
            results.append(row)

        result_df = pd.DataFrame(results)
        column_order = ["model", "teleqna", "telelogs", "telemath", "3gpp_tsg", "date"]
        result_df = result_df[column_order]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output_path, index=False)

        return result_df

    def _build_model_row(self, model_df: pd.DataFrame, model: str) -> dict:
        model_info = parse_model_string(model)
        row = {
            "model": model_info.display_name,
            "teleqna": None,
            "telelogs": None,
            "telemath": None,
            "3gpp_tsg": None,
            "date": date.today().isoformat(),
        }

        for _, eval_row in model_df.iterrows():
            task_name = eval_row.get("task_name", "")
            task_id = self._find_task_id(task_name)
            if task_id is None:
                continue

            column_name = TASK_TO_COLUMN[task_id]
            score_array = self._build_score_array(eval_row)
            if score_array is not None:
                row[column_name] = score_array

        return row

    def _find_task_id(self, task_name: str) -> str | None:
        task_lower = task_name.lower()
        for key in TASK_TO_COLUMN:
            if key in task_lower:
                return key
        return None

    def _build_score_array(self, eval_row: pd.Series) -> list[float] | None:
        score = eval_row.get("score_headline_value")
        if pd.isna(score):
            return None

        stderr = eval_row.get("score_headline_stderr")
        n_samples = self._extract_sample_count(eval_row)

        score_val = float(score) * 100 if float(score) <= 1.0 else float(score)
        stderr_val = 0.0
        if pd.notna(stderr):
            stderr_val = float(stderr) * 100 if float(stderr) <= 1.0 else float(stderr)
        n_samples_val = float(n_samples) if pd.notna(n_samples) else 0.0

        return [score_val, stderr_val, n_samples_val]

    def _extract_sample_count(self, eval_row: pd.Series) -> int:
        dataset_sample_ids = eval_row.get("dataset_sample_ids", "[]")

        if isinstance(dataset_sample_ids, str):
            try:
                sample_ids = json.loads(dataset_sample_ids)
                if isinstance(sample_ids, list):
                    return len(sample_ids)
            except (json.JSONDecodeError, TypeError):
                pass
            return 0

        if hasattr(dataset_sample_ids, "__len__"):
            return len(dataset_sample_ids)

        return eval_row.get("completed_samples", eval_row.get("total_samples", 0))

    def _format_results_preview(self, df: pd.DataFrame) -> str:
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
                lines.append(self._format_score_line(row.get(col), display))
            lines.append("")

        return "\n".join(lines)

    def _format_score_line(self, val: list | None, display: str) -> str:
        if val is None:
            return f"  {display:10} --"
        if not isinstance(val, list):
            return f"  {display:10} --"
        if len(val) < 2:
            return f"  {display:10} --"
        return f"  {display:10} {val[0]:6.2f} ± {val[1]:.2f}"

    def _colorize_preview_line(self, line: str) -> str:
        if line.startswith("model:"):
            return f"[{Colors.TEXT_PRIMARY}]{line}[/]"
        if "±" in line:
            return f"[{Colors.TEXT_MUTED}]{line}[/]"
        return line

    def _on_export_success(self, output_path: str) -> None:
        self.query_one("#error-message", Static).update(
            f"[{Colors.SUCCESS}]evaluation-complete![/]\n\n"
            f"[{Colors.TEXT_MUTED}]saved: {output_path}[/]"
        )
        self.query_one("#footer", Static).update(
            f"[{Colors.TEXT_MUTED}]enter[/] done [{Colors.BORDER}]|[/] [{Colors.TEXT_MUTED}]q[/] back"
        )

    def _on_export_error(self, error: str) -> None:
        self.query_one("#error-message", Static).update(
            f"[{Colors.ERROR}]Export failed: {error}[/]"
        )
        self.query_one("#footer", Static).update(f"[{Colors.TEXT_MUTED}]q[/] back")

    def action_cancel(self) -> None:
        self._cancelled = True
        self._kill_current_process()
        self.app.pop_screen()

    def action_confirm(self) -> None:
        if not self._completed:
            return
        # Pop EvalRunningScreen and RunEvalsScreen to return to main menu
        self.app.pop_screen()
        self.app.pop_screen()


class KSelectionScreen(Screen[int | None]):
    """Screen for confirming or selecting K value for epochs."""

    DEFAULT_CSS = f"""
    KSelectionScreen {{
        padding: 0 4;
        layout: vertical;
    }}

    #header {{
        color: {Colors.RED};
        text-style: bold;
        padding: 0 0 2 0;
        height: auto;
    }}

    #model-info {{
        color: {Colors.TEXT_MUTED};
        padding: 0 2 1 2;
        height: auto;
    }}

    #k-info {{
        color: {Colors.TEXT_PRIMARY};
        padding: 1 2;
        height: auto;
    }}

    #variance-info {{
        color: {Colors.SUCCESS};
        padding: 0 2 1 2;
        height: auto;
    }}

    #consistency-header {{
        color: {Colors.TEXT_MUTED};
        padding: 1 2 0 2;
        height: auto;
    }}

    #consistency-list {{
        height: auto;
        padding: 0 2;
    }}

    .consistency-item {{
        height: 1;
        padding: 0;
        background: transparent;
    }}

    #k-options {{
        color: {Colors.TEXT_MUTED};
        padding: 1 2;
        height: auto;
    }}

    #spacer {{
        height: 1fr;
    }}

    #footer {{
        dock: bottom;
        height: 1;
        color: {Colors.TEXT_DISABLED};
    }}
    """

    BINDINGS = [
        Binding("q", "cancel", "Cancel/Back"),
        Binding("escape", "cancel", "Cancel/Back"),
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("1", "select_k('1')", "K=1", show=False),
        Binding("2", "select_k('2')", "K=2", show=False),
        Binding("3", "select_k('3')", "K=3", show=False),
        Binding("4", "select_k('4')", "K=4", show=False),
        Binding("5", "select_k('5')", "K=5", show=False),
    ]

    def __init__(
        self,
        model: str,
        optimal_k: int,
        variance_reduction: float,
        task_consistency: dict[str, list[bool]],
        observed_variance: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimal_k = optimal_k
        self.selected_k = optimal_k
        self.variance_reduction = variance_reduction
        self.task_consistency = task_consistency
        self.observed_variance = observed_variance

    def compose(self) -> ComposeResult:
        yield Static("find-k", id="header")
        yield Static(
            f"model: [{Colors.TEXT_PRIMARY}]{self.model}[/]",
            id="model-info",
            markup=True,
        )
        yield Static(
            f"[{Colors.TEXT_PRIMARY}]recommended K: [bold]{self.optimal_k}[/bold][/]",
            id="k-info",
            markup=True,
        )
        yield Static(
            f"[{Colors.SUCCESS}]variance reduction: {self.variance_reduction:.0f}%[/]",
            id="variance-info",
            markup=True,
        )

        if self.task_consistency:
            yield Static(
                f"[{Colors.TEXT_MUTED}]task consistency (across 5 epochs):[/]",
                id="consistency-header",
                markup=True,
            )
            with Vertical(id="consistency-list"):
                for task, results in self.task_consistency.items():
                    yield self._render_consistency_item(task, results)

        yield Static(
            f"[{Colors.TEXT_MUTED}]press 1-5 to select different K, or enter to confirm[/]",
            id="k-options",
            markup=True,
        )
        yield Static("", id="spacer")
        yield Static(
            f"[{Colors.TEXT_MUTED}]enter[/] confirm [{Colors.BORDER}]|[/] "
            f"[{Colors.TEXT_MUTED}]1-5[/] select-k [{Colors.BORDER}]|[/] "
            f"[{Colors.TEXT_MUTED}]q[/] cancel",
            id="footer",
            markup=True,
        )

    def _render_consistency_item(self, task: str, results: list[bool]) -> Static:
        consistency_str = "".join(
            f"[{Colors.SUCCESS}]✓[/]" if r else f"[{Colors.ERROR}]✗[/]" for r in results
        )
        is_consistent = len(set(results)) == 1
        status = (
            f"[{Colors.SUCCESS}]consistent[/]"
            if is_consistent
            else f"[{Colors.WARNING}]varies[/]"
        )
        return Static(
            f"  {task:12} {consistency_str}  {status}",
            markup=True,
            classes="consistency-item",
        )

    def _update_k_display(self) -> None:
        from open_telco.cli.preflight.find_k import calculate_variance_reduction

        new_reduction = calculate_variance_reduction(
            self.selected_k, self.observed_variance
        )
        self.query_one("#k-info", Static).update(
            f"[{Colors.TEXT_PRIMARY}]selected K: [bold]{self.selected_k}[/bold][/]"
        )
        self.query_one("#variance-info", Static).update(
            f"[{Colors.SUCCESS}]variance reduction: {new_reduction:.0f}%[/]"
        )

    def action_select_k(self, k: str) -> None:
        self.selected_k = int(k)
        self._update_k_display()

    def action_confirm(self) -> None:
        self.dismiss(self.selected_k)

    def action_cancel(self) -> None:
        self.dismiss(None)


class RunEvalsScreen(Screen[None]):
    """Screen for running evaluations with preflight checklist."""

    DEFAULT_CSS = f"""
    RunEvalsScreen {{
        padding: 0 4;
        layout: vertical;
    }}

    #header {{
        color: {Colors.RED};
        text-style: bold;
        padding: 0 0 1 0;
        height: auto;
    }}

    #model-info {{
        color: {Colors.TEXT_MUTED};
        padding: 0 2 0 2;
        height: auto;
    }}

    #checklist-container {{
        width: 100%;
        max-width: 60;
        height: auto;
        padding: 0 2;
    }}

    #checklist {{
        height: auto;
        padding: 0;
    }}

    ChecklistItem {{
        height: 1;
        padding: 0;
        background: transparent;
    }}

    #error-message {{
        color: {Colors.ERROR};
        padding: 1 2;
        height: auto;
    }}

    #viewer-url {{
        color: {Colors.LINK};
        padding: 1 2;
        height: auto;
    }}

    #spacer {{
        height: 1fr;
    }}

    #footer {{
        dock: bottom;
        height: 1;
        color: {Colors.TEXT_DISABLED};
    }}
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
        self.tasks = list(ALL_TASKS)
        self._animation_timer: Timer | None = None
        self._current_process: subprocess.Popen[str] | None = None
        self._viewer_process: subprocess.Popen[str] | None = None
        self._cancelled = False
        self._full_eval_dot_count = 0
        self._selected_k = DEFAULT_K
        self._find_k_result: FindKResult | None = None

    def compose(self) -> ComposeResult:
        yield Static("run-evals", id="header")
        model_display = self.model if self.model else "not-configured"
        yield Static(
            f"model: [{Colors.TEXT_PRIMARY}]{model_display}[/]",
            id="model-info",
            markup=True,
        )
        with Container(id="checklist-container"):
            with Vertical(id="checklist"):
                yield ChecklistItem("mini-open-telco", "mini_test")
                yield ChecklistItem("find-k", "find_k")
                yield ChecklistItem("go", "ready")
        yield Static("", id="error-message", markup=True)
        yield Static("", id="viewer-url", markup=True)
        yield Static("", id="spacer")
        yield Static(f"[{Colors.TEXT_MUTED}]q[/] cancel", id="footer", markup=True)

    def on_mount(self) -> None:
        self._animation_timer = self.set_interval(
            Animation.INTERVAL_SECONDS, self._animate_dots
        )
        if not self.model:
            self._transition_to_error("no-model-configured. use set-model first.")
            return
        self._run_steps()

    def on_unmount(self) -> None:
        self._stop_animation_timer()
        self._kill_current_process()
        self._stop_viewer()

    def _stop_animation_timer(self) -> None:
        if self._animation_timer is None:
            return
        self._animation_timer.stop()
        self._animation_timer = None

    def _kill_current_process(self) -> None:
        if self._current_process is None:
            return
        self._terminate_process(self._current_process)
        self._current_process = None

    def _stop_viewer(self) -> None:
        if self._viewer_process is None:
            return
        self._terminate_process(self._viewer_process)
        self._viewer_process = None

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        self._send_sigterm(process)
        if not self._wait_for_process(process, timeout=2):
            self._send_sigkill(process)
            self._wait_for_process(process, timeout=None)

    def _send_sigterm(self, process: subprocess.Popen[str]) -> None:
        pgid = get_process_group(process.pid)
        if pgid is None:
            process.terminate()
            return
        kill_process_group(pgid, signal.SIGTERM)

    def _send_sigkill(self, process: subprocess.Popen[str]) -> None:
        pgid = get_process_group(process.pid)
        if pgid is None:
            process.kill()
            return
        kill_process_group(pgid, signal.SIGKILL)

    def _wait_for_process(
        self, process: subprocess.Popen[str], timeout: int | None
    ) -> bool:
        try:
            process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False
        except (ProcessLookupError, OSError):
            return True

    def _start_viewer(self) -> bool:
        import time

        if self._viewer_process is not None:
            return True

        log_dir = OPEN_TELCO_DIR / "logs" / "leaderboard"
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
            str(Ports.INSPECT_VIEWER),
        ]

        try:
            self._viewer_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                cwd=OPEN_TELCO_DIR,
            )
            time.sleep(0.3)

            if self._viewer_process.poll() is not None:
                _, stderr = self._viewer_process.communicate()
                error_msg = (
                    stderr.strip().split("\n")[-1] if stderr else "Unknown error"
                )
                self._viewer_process = None
                self._update_viewer_url_error(error_msg)
                return False

            self._update_viewer_url_success()
            return True
        except Exception as e:
            self._viewer_process = None
            self._update_viewer_url_error(str(e))
            return False

    def _update_viewer_url_success(self) -> None:
        self.query_one("#viewer-url", Static).update(
            f"[{Colors.LINK}]view live at:[/] [{Colors.TEXT_PRIMARY}]http://127.0.0.1:{Ports.INSPECT_VIEWER}[/]"
        )

    def _update_viewer_url_error(self, error: str) -> None:
        self.query_one("#viewer-url", Static).update(
            f"[{Colors.ERROR}]viewer failed: {error}[/]"
        )

    def _animate_dots(self) -> bool:
        has_running = self._animate_checklist_items()
        has_running = self._animate_full_eval_status() or has_running
        self._pause_timer_if_idle(has_running)
        return has_running

    def _animate_checklist_items(self) -> bool:
        animated = False
        for item in self.query(ChecklistItem):
            if item.status != "running":
                continue
            animated = True
            item.dot_count = (item.dot_count + 1) % Animation.PROGRESS_CYCLE_LENGTH
        return animated

    def _animate_full_eval_status(self) -> bool:
        if self.stage != Stage.RUNNING_EVAL:
            return False
        self._full_eval_dot_count = (
            self._full_eval_dot_count + 1
        ) % Animation.DOT_CYCLE_LENGTH
        self.query_one("#error-message", Static).update(self._render_full_eval_status())
        return True

    def _pause_timer_if_idle(self, has_running: bool) -> None:
        if has_running:
            return
        if self._animation_timer is None:
            return
        self._animation_timer.pause()

    def _resume_timer(self) -> None:
        if self._animation_timer is None:
            return
        self._animation_timer.resume()

    def _render_full_eval_status(self) -> str:
        dots = "." * ((self._full_eval_dot_count % Animation.DOT_CYCLE_LENGTH) + 1)
        padding = " " * (Animation.DOT_CYCLE_LENGTH - len(dots))
        return f"[{Colors.TEXT_MUTED}]running-full-evaluation{dots}{padding}[/]"

    def _find_checklist_item(self, step_id: str) -> ChecklistItem | None:
        for item in self.query(ChecklistItem):
            if item.step_id == step_id:
                return item
        return None

    def _set_step_status(self, step_id: str, status: str) -> bool:
        item = self._find_checklist_item(step_id)
        if item is None:
            return False
        item.status = status
        if status == "running":
            self._resume_timer()
        return True

    def _set_step_score(self, step_id: str, score: float) -> bool:
        item = self._find_checklist_item(step_id)
        if item is None:
            return False
        item.score = score
        return True

    def _set_find_k_score(self, optimal_k: int, variance_reduction: float) -> bool:
        item = self._find_checklist_item("find_k")
        if item is None:
            return False
        item.score = float(optimal_k)
        item.variance_reduction = variance_reduction
        return True

    def _transition_to_error(self, message: str) -> None:
        self.stage = Stage.ERROR
        self.query_one("#error-message", Static).update(f"[{Colors.ERROR}]{message}[/]")
        self.query_one("#footer", Static).update(f"[{Colors.TEXT_MUTED}]q[/] back")

    def _transition_to_ready(self) -> None:
        self.stage = Stage.READY
        self.app.push_screen(TaskSelectScreen(self.model), self._on_task_selection)

    def _on_task_selection(self, selected: list[str] | None) -> None:
        if selected is None:
            self.app.pop_screen()
            return
        self.tasks = selected
        self.app.push_screen(EvalRunningScreen(self.model, selected, self._selected_k))

    def _show_k_selection(self, find_k_result: FindKResult) -> None:
        self.app.push_screen(
            KSelectionScreen(
                model=self.model,
                optimal_k=find_k_result.optimal_k,
                variance_reduction=find_k_result.variance_reduction,
                task_consistency=find_k_result.task_consistency or {},
                observed_variance=find_k_result.observed_variance,
            ),
            self._on_k_selection,
        )

    def _on_k_selection(self, selected_k: int | None) -> None:
        if selected_k is None:
            self.app.pop_screen()
            return
        self._selected_k = selected_k
        self._continue_after_k_selection()

    def _continue_after_k_selection(self) -> None:
        if self._cancelled:
            return
        self._set_step_status("ready", "passed")
        self._transition_to_ready()

    def _check_preflight_passed(self) -> bool:
        preflight_dir = OPEN_TELCO_DIR / "logs" / "preflight"
        if not preflight_dir.exists():
            return False

        found_tasks: set[str] = set()
        for json_file in preflight_dir.glob("*.json"):
            task = self._check_preflight_file(json_file)
            if task is not None:
                found_tasks.add(task)

        return found_tasks == REQUIRED_PREFLIGHT_TASKS

    def _check_preflight_file(self, json_file: Path) -> str | None:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        if data.get("status") != "success":
            return None

        eval_info = data.get("eval", {})
        if eval_info.get("model", "") != self.model:
            return None

        task_name = eval_info.get("task", "")
        if task_name in REQUIRED_PREFLIGHT_TASKS:
            return task_name
        return None

    @work(exclusive=True, thread=True)
    def _run_steps(self) -> None:
        if self._cancelled:
            return

        if self._check_preflight_passed():
            self._mark_all_steps_passed()
            self.app.call_from_thread(self._transition_to_ready)
            return

        mini_result = self._execute_mini_test()
        if not mini_result:
            return

        find_k_result = self._execute_find_k()
        if find_k_result is None:
            return

        self.app.call_from_thread(self._show_k_selection, find_k_result)

    def _mark_all_steps_passed(self) -> None:
        self.app.call_from_thread(self._set_step_status, "mini_test", "passed")
        self.app.call_from_thread(self._set_step_status, "find_k", "passed")
        self.app.call_from_thread(self._set_step_status, "ready", "passed")

    def _execute_mini_test(self) -> bool:
        self.app.call_from_thread(self._set_step_status, "mini_test", "running")
        self.app.call_from_thread(self._set_stage, Stage.MINI_TEST)

        result = self._run_mini_test()
        if self._cancelled:
            return False
        if not result.passed:
            self.app.call_from_thread(self._set_step_status, "mini_test", "failed")
            self.app.call_from_thread(
                self._transition_to_error, result.error or "Mini test failed"
            )
            return False

        self.app.call_from_thread(self._set_step_status, "mini_test", "passed")
        if result.score is not None:
            self.app.call_from_thread(self._set_step_score, "mini_test", result.score)
        return True

    def _execute_find_k(self) -> FindKResult | None:
        if self._cancelled:
            return None

        self.app.call_from_thread(self._set_step_status, "find_k", "running")
        self.app.call_from_thread(self._set_stage, Stage.FIND_K)

        find_k_result = self._run_find_k()
        if self._cancelled:
            return None
        if not find_k_result.passed:
            self.app.call_from_thread(self._set_step_status, "find_k", "failed")
            self.app.call_from_thread(
                self._transition_to_error, find_k_result.error or "Find-K failed"
            )
            return None

        self.app.call_from_thread(self._set_step_status, "find_k", "passed")
        self.app.call_from_thread(
            self._set_find_k_score,
            find_k_result.optimal_k,
            find_k_result.variance_reduction,
        )
        self._find_k_result = find_k_result
        return find_k_result

    def _set_stage(self, stage: Stage) -> None:
        self.stage = stage

    def _run_mini_test(self) -> StepResult:
        if self._cancelled:
            return StepResult(passed=False, error="cancelled")

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
            "logs/preflight",
            "--log-format",
            "json",
        ]

        process = start_process(cmd, cwd=OPEN_TELCO_DIR)
        if process is None:
            return StepResult(passed=False, error="Failed to start process")

        self._current_process = process
        stdout, stderr, timed_out = communicate_with_timeout(
            process, Timeouts.MINI_TEST
        )
        self._current_process = None

        if timed_out:
            return StepResult(passed=False, error="Mini test timed out after 5 minutes")
        if self._cancelled:
            return StepResult(passed=False, error="cancelled")
        if process.returncode != 0:
            return StepResult(
                passed=False, error=self._extract_last_error_line(stderr or stdout)
            )

        return StepResult(passed=True, score=self._parse_score(stdout + stderr))

    def _extract_last_error_line(self, output: str) -> str:
        if not output:
            return "Eval failed"
        lines = [line for line in output.strip().split("\n") if line.strip()]
        if not lines:
            return "Eval failed"
        return lines[-1]

    def _parse_score(self, output: str) -> float | None:
        matches = re.findall(r"accuracy[=:\s]+([0-9.]+)", output, re.IGNORECASE)
        if not matches:
            return None
        return sum(float(m) for m in matches) / len(matches)

    def _run_find_k(self) -> FindKResult:
        from open_telco.cli.preflight.find_k import run_find_k_sync

        if self._cancelled:
            return FindKResult(passed=False, error="cancelled")

        try:
            result = run_find_k_sync(
                model=self.model,
                epochs=5,
                tasks=self.tasks,
                open_telco_dir=OPEN_TELCO_DIR,
                timeout=Timeouts.FIND_K,
            )

            return FindKResult(
                passed=True,
                optimal_k=result.optimal_k,
                variance_reduction=result.variance_reduction_pct,
                task_consistency=result.task_consistency,
                observed_variance=result.observed_variance,
                error=result.error,
            )
        except Exception as e:
            return FindKResult(passed=False, error=str(e))

    def _start_full_eval(self) -> None:
        self.stage = Stage.RUNNING_EVAL

        log_dir = OPEN_TELCO_DIR / "logs" / "leaderboard"
        log_dir.mkdir(parents=True, exist_ok=True)

        self._start_viewer()
        self._full_eval_dot_count = 0
        self._resume_timer()

        self.query_one("#error-message", Static).update(self._render_full_eval_status())
        self.query_one("#footer", Static).update(
            f"[{Colors.TEXT_MUTED}]q[/] cancel-unsafe"
        )
        self._run_full_eval()

    @work(exclusive=True, thread=True)
    def _run_full_eval(self) -> None:
        if self._cancelled:
            return

        cmd = [
            "uv",
            "run",
            "inspect",
            "eval",
            *self.tasks,
            "--model",
            self.model,
            "--epochs",
            str(self._selected_k),
            "--log-dir",
            "logs/leaderboard",
            "--log-format",
            "json",
        ]

        process = start_process(cmd, cwd=OPEN_TELCO_DIR)
        if process is None:
            self.app.call_from_thread(
                self._transition_to_error, "Failed to start evaluation process"
            )
            return

        self._current_process = process
        stdout, stderr, timed_out = communicate_with_timeout(
            process, Timeouts.FULL_EVAL
        )
        self._current_process = None

        if timed_out:
            self.app.call_from_thread(
                self._transition_to_error, "Evaluation timed out after 1 hour"
            )
            return
        if self._cancelled:
            return
        if process.returncode != 0:
            error_msg = self._extract_last_error_line(stderr or stdout)
            self.app.call_from_thread(
                self._transition_to_error, f"Evaluation failed: {error_msg}"
            )
            return

        self.app.call_from_thread(self._export_and_show_results)

    def _export_and_show_results(self) -> None:
        self.stage = Stage.EXPORTING
        self.query_one("#error-message", Static).update(
            f"[{Colors.TEXT_MUTED}]exporting-results...[/]"
        )
        self._do_export()

    @work(exclusive=True, thread=True)
    def _do_export(self) -> None:
        try:
            log_dir = OPEN_TELCO_DIR / "logs" / "leaderboard"
            output_path = log_dir / "results.parquet"

            if not log_dir.exists():
                raise FileNotFoundError(f"Log directory not found: {log_dir}")

            df = self._export_to_leaderboard_parquet(str(log_dir), str(output_path))
            preview = self._format_results_preview(df)
            self.app.call_from_thread(
                self._on_export_success,
                preview,
                str(output_path.relative_to(OPEN_TELCO_DIR)),
            )

        except Exception as e:
            self.app.call_from_thread(self._transition_to_error, f"Export failed: {e}")

    def _export_to_leaderboard_parquet(
        self, log_dir: str, output_path: str
    ) -> pd.DataFrame:
        from inspect_ai.analysis import evals_df

        df = evals_df(log_dir)

        if df.empty:
            raise ValueError(f"No eval logs found in {log_dir}")

        results = []
        for model in df["model"].unique():
            row = self._build_model_row(df[df["model"] == model], model)
            results.append(row)

        result_df = pd.DataFrame(results)
        column_order = ["model", "teleqna", "telelogs", "telemath", "3gpp_tsg", "date"]
        result_df = result_df[column_order]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output_path, index=False)

        return result_df

    def _build_model_row(self, model_df: pd.DataFrame, model: str) -> dict:
        model_info = parse_model_string(model)
        row = {
            "model": model_info.display_name,
            "teleqna": None,
            "telelogs": None,
            "telemath": None,
            "3gpp_tsg": None,
            "date": date.today().isoformat(),
        }

        for _, eval_row in model_df.iterrows():
            task_name = eval_row.get("task_name", "")
            task_id = self._find_task_id(task_name)
            if task_id is None:
                continue

            column_name = TASK_TO_COLUMN[task_id]
            score_array = self._build_score_array(eval_row)
            if score_array is not None:
                row[column_name] = score_array

        return row

    def _find_task_id(self, task_name: str) -> str | None:
        task_lower = task_name.lower()
        for key in TASK_TO_COLUMN:
            if key in task_lower:
                return key
        return None

    def _build_score_array(self, eval_row: pd.Series) -> list[float] | None:
        score = eval_row.get("score_headline_value")
        if pd.isna(score):
            return None

        stderr = eval_row.get("score_headline_stderr")
        n_samples = self._extract_sample_count(eval_row)

        score_val = float(score) * 100 if float(score) <= 1.0 else float(score)
        stderr_val = 0.0
        if pd.notna(stderr):
            stderr_val = float(stderr) * 100 if float(stderr) <= 1.0 else float(stderr)
        n_samples_val = float(n_samples) if pd.notna(n_samples) else 0.0

        return [score_val, stderr_val, n_samples_val]

    def _extract_sample_count(self, eval_row: pd.Series) -> int:
        dataset_sample_ids = eval_row.get("dataset_sample_ids", "[]")

        if isinstance(dataset_sample_ids, str):
            try:
                sample_ids = json.loads(dataset_sample_ids)
                if isinstance(sample_ids, list):
                    return len(sample_ids)
            except (json.JSONDecodeError, TypeError):
                pass
            return 0

        if hasattr(dataset_sample_ids, "__len__"):
            return len(dataset_sample_ids)

        return eval_row.get("completed_samples", eval_row.get("total_samples", 0))

    def _format_results_preview(self, df: pd.DataFrame) -> str:
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
                lines.append(self._format_score_line(row.get(col), display))
            lines.append("")

        return "\n".join(lines)

    def _format_score_line(self, val: list | None, display: str) -> str:
        if val is None:
            return f"  {display:10} --"
        if not isinstance(val, list):
            return f"  {display:10} --"
        if len(val) < 2:
            return f"  {display:10} --"
        return f"  {display:10} {val[0]:6.2f} ± {val[1]:.2f}"

    def _colorize_preview_line(self, line: str) -> str:
        if line.startswith("model:"):
            return f"[{Colors.TEXT_PRIMARY}]{line}[/]"
        if "±" in line:
            return f"[{Colors.TEXT_MUTED}]{line}[/]"
        return line

    def _on_export_success(self, preview: str, output_path: str) -> None:
        self.stage = Stage.COMPLETE
        formatted_lines = [
            self._colorize_preview_line(line) for line in preview.strip().split("\n")
        ]
        formatted_preview = "\n".join(formatted_lines)

        self.query_one("#error-message", Static).update(
            f"[{Colors.SUCCESS}]evaluation-complete![/]\n\n{formatted_preview}\n"
            f"[{Colors.TEXT_MUTED}]saved: {output_path}[/]"
        )
        self.query_one("#footer", Static).update(
            f"[{Colors.TEXT_MUTED}]enter[/] done [{Colors.BORDER}]|[/] [{Colors.TEXT_MUTED}]q[/] back"
        )

    def action_cancel(self) -> None:
        self._cancelled = True
        self._kill_current_process()
        self.app.pop_screen()

    def action_confirm(self) -> None:
        if self.stage != Stage.COMPLETE:
            return
        self.app.pop_screen()

    def _parse_model_display_name(self, model_str: str) -> str:
        """Parse model string to display format for test compatibility."""
        model_info = parse_model_string(model_str)
        return model_info.display_name

    def _get_open_telco_dir(self) -> Path:
        """Get the open_telco source directory path for test compatibility."""
        return OPEN_TELCO_DIR
