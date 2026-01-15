"""Shared fixtures for preflight tests."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from evals.cli.preflight.runner import (
    PreflightConfig,
    PreflightStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def mock_eval_func() -> MagicMock:
    """Mock eval function that returns success."""
    mock = MagicMock()
    mock.return_value = (True, {})
    return mock


@pytest.fixture
def mock_failing_eval_func() -> MagicMock:
    """Mock eval function that returns failure."""
    mock = MagicMock()
    mock.return_value = (False, {})
    return mock


@pytest.fixture
def mock_raising_eval_func() -> MagicMock:
    """Mock eval function that raises an exception."""
    mock = MagicMock()
    mock.side_effect = RuntimeError("API connection failed")
    return mock


@pytest.fixture
def mock_slow_eval_func() -> Callable[..., tuple[bool, Any]]:
    """Mock eval function that takes time (for timeout testing)."""

    def slow_eval(**kwargs: object) -> tuple[bool, Any]:
        time.sleep(5)
        return (True, {})

    return slow_eval


@pytest.fixture
def progress_tracker() -> dict[str, list[tuple[str, PreflightStatus]]]:
    """Track progress callback invocations."""
    return {"calls": []}


@pytest.fixture
def progress_callback(
    progress_tracker: dict[str, list[tuple[str, PreflightStatus]]],
) -> Callable[[str, PreflightStatus], None]:
    """Progress callback that records all calls."""

    def callback(model: str, status: PreflightStatus) -> None:
        progress_tracker["calls"].append((model, status))

    return callback


@pytest.fixture
def default_config() -> PreflightConfig:
    """Default preflight configuration for tests."""
    return PreflightConfig(
        sample_limit=2,
        timeout_seconds=10,
        tasks=("task1.py", "task2.py"),
        log_dir="/tmp/test_logs",
    )


@pytest.fixture
def short_timeout_config() -> PreflightConfig:
    """Configuration with very short timeout for testing timeouts."""
    return PreflightConfig(
        sample_limit=1,
        timeout_seconds=0.1,
        tasks=("task1.py",),
        log_dir="/tmp/test_logs",
    )
