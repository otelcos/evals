"""Subprocess utilities with single-responsibility error handling."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def start_process(
    cmd: list[str],
    cwd: Path | None = None,
    start_new_session: bool = True,
) -> subprocess.Popen | None:
    """Start a subprocess. Returns None on failure."""
    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            start_new_session=start_new_session,
        )
    except OSError:
        return None


def communicate_with_timeout(
    process: subprocess.Popen,
    timeout: int,
) -> tuple[str, str, bool]:
    """Communicate with process. Returns (stdout, stderr, timed_out)."""
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return stdout, stderr, False
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return "", "", True


def get_process_group(pid: int) -> int | None:
    """Get process group ID. Returns None if process doesn't exist."""
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, OSError):
        return None


def kill_process_group(pgid: int, sig: signal.Signals) -> bool:
    """Kill process group. Returns True on success."""
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, OSError):
        return False
