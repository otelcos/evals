"""Tests for subprocess termination on run-evals cancellation.

These tests ensure that when the user cancels or exits the run-evals screen,
any running subprocess (inspect eval) is properly terminated and doesn't
continue running in the background.

Test matrix:
- Exit actions: action_cancel, on_unmount, SIGINT, direct_kill
- Process states: running, none, already_dead, zombie
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Callable
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from open_telco.cli.screens.run_evals import RunEvalsScreen

if TYPE_CHECKING:
    from subprocess import Popen


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def screen() -> RunEvalsScreen:
    """Create a RunEvalsScreen with model configured."""
    s = RunEvalsScreen()
    s.model = "test-model"
    s._animation_timer = None  # Prevent timer issues in tests
    return s


@pytest.fixture
def mock_app(screen: RunEvalsScreen) -> MagicMock:
    """Mock the app property for action_cancel tests."""
    with patch.object(type(screen), "app", new_callable=PropertyMock) as mock_app_prop:
        mock_app = MagicMock()
        mock_app_prop.return_value = mock_app
        yield mock_app


@pytest.fixture
def running_process(screen: RunEvalsScreen) -> Popen[str]:
    """Create a running subprocess attached to screen."""
    proc = subprocess.Popen(
        ["sleep", "60"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    screen._current_process = proc
    yield proc
    # Cleanup if test didn't kill it
    if proc.poll() is None:
        proc.kill()
        proc.wait()


@pytest.fixture
def dead_process(screen: RunEvalsScreen) -> Popen[str]:
    """Create an already-terminated subprocess attached to screen."""
    proc = subprocess.Popen(
        ["true"],  # Exits immediately with 0
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc.wait()  # Wait for it to finish
    screen._current_process = proc
    return proc


@pytest.fixture
def zombie_process(screen: RunEvalsScreen) -> int:
    """Create a zombie process (terminated but not reaped) attached to screen.

    Returns the PID of the zombie process.
    """
    # Create a subprocess that spawns a child and exits
    # The child becomes a zombie until we reap it
    proc = subprocess.Popen(
        ["sh", "-c", "exit 0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid = proc.pid
    # Don't call wait() - let it become a zombie briefly
    time.sleep(0.1)  # Give it time to exit
    screen._current_process = proc
    return pid


# =============================================================================
# Exit Action Helpers
# =============================================================================


def do_action_cancel(screen: RunEvalsScreen, mock_app: MagicMock) -> None:
    """Execute action_cancel on the screen."""
    screen.action_cancel()


def do_on_unmount(screen: RunEvalsScreen, mock_app: MagicMock) -> None:
    """Execute on_unmount on the screen."""
    screen.on_unmount()


def do_direct_kill(screen: RunEvalsScreen, mock_app: MagicMock) -> None:
    """Execute _kill_current_process directly."""
    screen._kill_current_process()


def do_sigint(screen: RunEvalsScreen, mock_app: MagicMock) -> None:
    """Simulate SIGINT by calling action_cancel (app-level handler would do this)."""
    # In a real app, SIGINT handler would trigger cleanup
    # We simulate by calling action_cancel which is what the app would do
    screen.action_cancel()


EXIT_ACTIONS: list[tuple[str, Callable[[RunEvalsScreen, MagicMock], None]]] = [
    ("action_cancel", do_action_cancel),
    ("on_unmount", do_on_unmount),
    ("direct_kill", do_direct_kill),
    ("sigint", do_sigint),
]


# =============================================================================
# Parametrized Tests: Process Cleanup Matrix
# =============================================================================


class TestProcessCleanupMatrix:
    """Parametrized tests for all exit action × process state combinations."""

    @pytest.mark.parametrize("action_name,action_fn", EXIT_ACTIONS)
    @pytest.mark.slow
    def test_running_process_terminated(
        self,
        screen: RunEvalsScreen,
        mock_app: MagicMock,
        running_process: Popen[str],
        action_name: str,
        action_fn: Callable[[RunEvalsScreen, MagicMock], None],
    ) -> None:
        """Running process should be terminated after exit action."""
        action_fn(screen, mock_app)
        time.sleep(0.1)  # Allow cleanup
        assert running_process.poll() is not None

    @pytest.mark.parametrize("action_name,action_fn", EXIT_ACTIONS)
    @pytest.mark.slow
    def test_running_process_reference_cleared(
        self,
        screen: RunEvalsScreen,
        mock_app: MagicMock,
        running_process: Popen[str],
        action_name: str,
        action_fn: Callable[[RunEvalsScreen, MagicMock], None],
    ) -> None:
        """_current_process should be None after cleanup."""
        action_fn(screen, mock_app)
        assert screen._current_process is None

    @pytest.mark.parametrize("action_name,action_fn", EXIT_ACTIONS)
    @pytest.mark.slow
    def test_running_process_no_orphan(
        self,
        screen: RunEvalsScreen,
        mock_app: MagicMock,
        running_process: Popen[str],
        action_name: str,
        action_fn: Callable[[RunEvalsScreen, MagicMock], None],
    ) -> None:
        """No orphan process should remain after exit action."""
        pid = running_process.pid
        action_fn(screen, mock_app)
        time.sleep(0.1)

        try:
            os.kill(pid, 0)  # Check if process exists
            pytest.fail(f"Process {pid} still running after {action_name}")
        except ProcessLookupError:
            pass  # Expected - process is gone

    @pytest.mark.parametrize("action_name,action_fn", EXIT_ACTIONS)
    def test_none_process_no_exception(
        self,
        screen: RunEvalsScreen,
        mock_app: MagicMock,
        action_name: str,
        action_fn: Callable[[RunEvalsScreen, MagicMock], None],
    ) -> None:
        """Exit action should not raise when _current_process is None."""
        screen._current_process = None
        action_fn(screen, mock_app)  # Should not raise

    @pytest.mark.parametrize("action_name,action_fn", EXIT_ACTIONS)
    def test_none_process_reference_stays_none(
        self,
        screen: RunEvalsScreen,
        mock_app: MagicMock,
        action_name: str,
        action_fn: Callable[[RunEvalsScreen, MagicMock], None],
    ) -> None:
        """_current_process should remain None after cleanup."""
        screen._current_process = None
        action_fn(screen, mock_app)
        assert screen._current_process is None

    @pytest.mark.parametrize("action_name,action_fn", EXIT_ACTIONS)
    def test_dead_process_no_exception(
        self,
        screen: RunEvalsScreen,
        mock_app: MagicMock,
        dead_process: Popen[str],
        action_name: str,
        action_fn: Callable[[RunEvalsScreen, MagicMock], None],
    ) -> None:
        """Exit action should not raise for already-dead process."""
        action_fn(screen, mock_app)  # Should not raise

    @pytest.mark.parametrize("action_name,action_fn", EXIT_ACTIONS)
    def test_dead_process_reference_cleared(
        self,
        screen: RunEvalsScreen,
        mock_app: MagicMock,
        dead_process: Popen[str],
        action_name: str,
        action_fn: Callable[[RunEvalsScreen, MagicMock], None],
    ) -> None:
        """_current_process should be None after cleanup of dead process."""
        action_fn(screen, mock_app)
        assert screen._current_process is None

    @pytest.mark.parametrize("action_name,action_fn", EXIT_ACTIONS)
    def test_zombie_process_no_exception(
        self,
        screen: RunEvalsScreen,
        mock_app: MagicMock,
        zombie_process: int,
        action_name: str,
        action_fn: Callable[[RunEvalsScreen, MagicMock], None],
    ) -> None:
        """Exit action should not raise for zombie process."""
        action_fn(screen, mock_app)  # Should not raise

    @pytest.mark.parametrize("action_name,action_fn", EXIT_ACTIONS)
    def test_zombie_process_reference_cleared(
        self,
        screen: RunEvalsScreen,
        mock_app: MagicMock,
        zombie_process: int,
        action_name: str,
        action_fn: Callable[[RunEvalsScreen, MagicMock], None],
    ) -> None:
        """_current_process should be None after cleanup of zombie process."""
        action_fn(screen, mock_app)
        assert screen._current_process is None


# =============================================================================
# SIGTERM -> SIGKILL Escalation Tests
# =============================================================================


class TestStubbornProcessKill:
    """Test SIGTERM -> SIGKILL escalation for stubborn processes."""

    @pytest.fixture
    def stubborn_process_screen(
        self, screen: RunEvalsScreen
    ) -> tuple[RunEvalsScreen, MagicMock]:
        """Create screen with mock process that times out on SIGTERM."""
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 2), None]
        mock_proc.pid = 12345
        screen._current_process = mock_proc
        return screen, mock_proc

    def test_terminate_called_first(self, screen: RunEvalsScreen) -> None:
        """SIGTERM (via terminate or killpg) should be sent first."""
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.pid = 12345
        screen._current_process = mock_proc

        with patch("os.getpgid", side_effect=ProcessLookupError):
            screen._kill_current_process()

        mock_proc.terminate.assert_called_once()

    def test_terminate_called_before_kill_escalation(
        self, stubborn_process_screen: tuple[RunEvalsScreen, MagicMock]
    ) -> None:
        """SIGTERM should be sent first before escalating to SIGKILL."""
        screen, mock_proc = stubborn_process_screen

        with patch("os.getpgid", side_effect=ProcessLookupError):
            screen._kill_current_process()

        mock_proc.terminate.assert_called_once()

    def test_kill_called_after_terminate_timeout(
        self, stubborn_process_screen: tuple[RunEvalsScreen, MagicMock]
    ) -> None:
        """SIGKILL should be sent after SIGTERM times out."""
        screen, mock_proc = stubborn_process_screen

        with patch("os.getpgid", side_effect=ProcessLookupError):
            screen._kill_current_process()

        mock_proc.kill.assert_called_once()

    def test_timeout_uses_2_seconds(self, screen: RunEvalsScreen) -> None:
        """Verify the hardcoded 2 second timeout is used."""
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.pid = 12345
        screen._current_process = mock_proc

        with patch("os.getpgid", side_effect=ProcessLookupError):
            screen._kill_current_process()

        mock_proc.wait.assert_called_with(timeout=2)


# =============================================================================
# Process Group Termination Tests
# =============================================================================


class TestProcessGroupTermination:
    """Test that child processes are also terminated via process group."""

    @pytest.mark.slow
    def test_child_processes_terminated(
        self, screen: RunEvalsScreen, mock_app: MagicMock
    ) -> None:
        """Subprocess children should be killed via process group."""
        # Spawn a process that spawns background children
        proc = subprocess.Popen(
            ["sh", "-c", "sleep 60 & sleep 60 & wait"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        pgid = os.getpgid(proc.pid)
        screen._current_process = proc

        screen.action_cancel()
        time.sleep(0.2)

        # Check that no processes in the group are running
        try:
            os.killpg(pgid, 0)  # Check if process group exists
            pytest.fail(f"Process group {pgid} still exists after cancel")
        except (ProcessLookupError, OSError):
            pass  # Expected

    def test_getpgid_called_for_process(self, screen: RunEvalsScreen) -> None:
        """os.getpgid should be called with process pid."""
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.pid = 12345
        screen._current_process = mock_proc

        with (
            patch("os.getpgid", return_value=12345) as mock_getpgid,
            patch("os.killpg"),
        ):
            screen._kill_current_process()

        mock_getpgid.assert_called_with(12345)

    def test_killpg_sends_sigterm_to_process_group(
        self, screen: RunEvalsScreen
    ) -> None:
        """os.killpg should send SIGTERM to the process group."""
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.pid = 12345
        screen._current_process = mock_proc

        with (
            patch("os.getpgid", return_value=12345),
            patch("os.killpg") as mock_killpg,
        ):
            screen._kill_current_process()

        mock_killpg.assert_called_with(12345, signal.SIGTERM)


# =============================================================================
# Race Condition Tests
# =============================================================================


class TestRaceConditions:
    """Test concurrent cleanup scenarios are safe."""

    @pytest.mark.slow
    def test_cancel_and_unmount_concurrent(
        self, screen: RunEvalsScreen, mock_app: MagicMock
    ) -> None:
        """Concurrent action_cancel and on_unmount should be safe."""
        proc = subprocess.Popen(
            ["sleep", "60"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        screen._current_process = proc

        def cancel() -> None:
            try:
                screen.action_cancel()
            except Exception:
                pass  # May fail if app mock is accessed concurrently

        def unmount() -> None:
            screen.on_unmount()

        t1 = threading.Thread(target=cancel)
        t2 = threading.Thread(target=unmount)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert screen._current_process is None

    @pytest.mark.slow
    def test_multiple_cancels_are_idempotent(
        self, screen: RunEvalsScreen, mock_app: MagicMock
    ) -> None:
        """Multiple cancel calls should be safe (idempotent)."""
        proc = subprocess.Popen(
            ["sleep", "60"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        screen._current_process = proc

        # Cancel multiple times
        screen.action_cancel()
        screen.action_cancel()
        screen.action_cancel()

        assert screen._current_process is None

    @pytest.mark.slow
    def test_multiple_cancels_no_exception(
        self, screen: RunEvalsScreen, mock_app: MagicMock
    ) -> None:
        """Multiple cancel calls should not raise exceptions."""
        proc = subprocess.Popen(
            ["sleep", "60"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        screen._current_process = proc

        # Should not raise
        for _ in range(5):
            screen.action_cancel()

    def test_unmount_after_cancel_is_safe(
        self, screen: RunEvalsScreen, mock_app: MagicMock
    ) -> None:
        """on_unmount after action_cancel should be safe."""
        screen._current_process = None
        screen._cancelled = True

        screen.action_cancel()
        screen.on_unmount()  # Should not raise

        assert screen._current_process is None


# =============================================================================
# Cancelled Flag Lifecycle Tests
# =============================================================================


class TestCancelledFlag:
    """Test _cancelled flag behavior."""

    def test_initially_false(self, screen: RunEvalsScreen) -> None:
        """_cancelled should be False on initialization."""
        assert screen._cancelled is False

    def test_set_true_on_cancel(
        self, screen: RunEvalsScreen, mock_app: MagicMock
    ) -> None:
        """_cancelled should be True after action_cancel."""
        screen.action_cancel()
        assert screen._cancelled is True

    def test_stays_true_after_cleanup(
        self, screen: RunEvalsScreen, mock_app: MagicMock
    ) -> None:
        """_cancelled should remain True forever (screen is disposable)."""
        screen.action_cancel()
        # Even after additional cleanup
        screen.on_unmount()
        assert screen._cancelled is True

    def test_prevents_mini_test(self, screen: RunEvalsScreen) -> None:
        """_run_mini_test should return immediately if cancelled."""
        screen._cancelled = True

        result = screen._run_mini_test()

        assert result["passed"] is False

    def test_prevents_mini_test_with_cancelled_error(
        self, screen: RunEvalsScreen
    ) -> None:
        """_run_mini_test should return 'cancelled' error when cancelled."""
        screen._cancelled = True

        result = screen._run_mini_test()

        assert result["error"] == "cancelled"


# =============================================================================
# Current Process Initialization Tests
# =============================================================================


class TestCurrentProcessInitialization:
    """Test _current_process initial state."""

    def test_initially_none(self, screen: RunEvalsScreen) -> None:
        """_current_process should be None on initialization."""
        assert screen._current_process is None


# =============================================================================
# Animation Timer Cleanup Tests
# =============================================================================


class TestAnimationTimerCleanup:
    """Test animation timer is stopped on cleanup."""

    def test_unmount_stops_timer(self, screen: RunEvalsScreen) -> None:
        """on_unmount should stop animation timer."""
        mock_timer = MagicMock()
        screen._animation_timer = mock_timer
        screen._current_process = None

        screen.on_unmount()

        mock_timer.stop.assert_called_once()

    def test_unmount_clears_timer_reference(self, screen: RunEvalsScreen) -> None:
        """on_unmount should set _animation_timer to None."""
        mock_timer = MagicMock()
        screen._animation_timer = mock_timer
        screen._current_process = None

        screen.on_unmount()

        assert screen._animation_timer is None

    def test_unmount_cleans_both_timer_and_process(
        self, screen: RunEvalsScreen
    ) -> None:
        """on_unmount should clean up both timer and process."""
        mock_timer = MagicMock()
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.pid = 12345

        screen._animation_timer = mock_timer
        screen._current_process = mock_proc

        with patch("os.getpgid", side_effect=ProcessLookupError):
            screen.on_unmount()

        mock_timer.stop.assert_called_once()
        assert screen._animation_timer is None
        assert screen._current_process is None


# =============================================================================
# Action Cancel Specific Tests
# =============================================================================


class TestActionCancel:
    """Tests specific to action_cancel behavior."""

    def test_sets_cancelled_flag(
        self, screen: RunEvalsScreen, mock_app: MagicMock
    ) -> None:
        """action_cancel should set _cancelled to True."""
        screen._cancelled = False
        screen.action_cancel()
        assert screen._cancelled is True

    def test_calls_pop_screen(
        self, screen: RunEvalsScreen, mock_app: MagicMock
    ) -> None:
        """action_cancel should call app.pop_screen()."""
        screen.action_cancel()
        mock_app.pop_screen.assert_called_once()

    def test_kills_before_pop(
        self, screen: RunEvalsScreen, mock_app: MagicMock
    ) -> None:
        """action_cancel should kill process before popping screen."""
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.pid = 12345
        screen._current_process = mock_proc

        call_order: list[str] = []
        mock_proc.terminate.side_effect = lambda: call_order.append("terminate")
        mock_app.pop_screen.side_effect = lambda: call_order.append("pop_screen")

        with patch("os.getpgid", side_effect=ProcessLookupError):
            screen.action_cancel()

        assert call_order.index("terminate") < call_order.index("pop_screen")
