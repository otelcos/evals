"""Pre-flight model testing runner."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

DEFAULT_SAMPLE_LIMIT = 5
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_TASKS = ("telelogs/telelogs.py", "teleqna/teleqna.py")
DEFAULT_LOG_DIR = "logs/preflight"


class PreflightStatus(Enum):
    """Status of a pre-flight test."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class PreflightConfig:
    """Configuration for pre-flight tests."""

    sample_limit: int = DEFAULT_SAMPLE_LIMIT
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    tasks: tuple[str, ...] = DEFAULT_TASKS
    log_dir: str = DEFAULT_LOG_DIR


EvalFunc = Callable[..., tuple[bool, Any]]


class PreflightRunner:
    """Runs pre-flight tests on models before full evaluation."""

    def __init__(
        self,
        eval_func: EvalFunc,
        config: PreflightConfig | None = None,
        on_progress: Callable[[str, PreflightStatus], None] | None = None,
    ) -> None:
        self._eval_func = eval_func
        self.config = config or PreflightConfig()
        self.on_progress = on_progress
        self.results: dict[str, bool | None] = {}
        self._cancelled = False

    def cancel(self) -> None:
        """Cancel the pre-flight tests."""
        self._cancelled = True

    def _notify(self, model: str, status: PreflightStatus) -> None:
        if self.on_progress:
            self.on_progress(model, status)

    def _execute_eval_sync(self, model: str) -> tuple[bool, Any]:
        """Execute evaluation synchronously in thread pool."""
        return self._eval_func(
            tasks=list(self.config.tasks),
            model=[model],
            limit=self.config.sample_limit,
            log_dir=self.config.log_dir,
            epochs=1,
            temperature=0.0,
        )

    async def run_model_test(self, model: str) -> bool | None:
        """Run pre-flight test for a single model.

        Returns:
            True: Test passed
            False: Test failed
            None: Test skipped or timed out

        Raises:
            Exception: Any exception from eval_func propagates up
        """
        if self._cancelled:
            self._notify(model, PreflightStatus.SKIPPED)
            return None

        self._notify(model, PreflightStatus.RUNNING)

        try:
            loop = asyncio.get_event_loop()
            success, _ = await asyncio.wait_for(
                loop.run_in_executor(None, self._execute_eval_sync, model),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._notify(model, PreflightStatus.TIMEOUT)
            return None

        if not success:
            self._notify(model, PreflightStatus.FAILED)
            return False

        self._notify(model, PreflightStatus.PASSED)
        return True

    async def run_all(self, models: list[str]) -> dict[str, bool | None]:
        """Run pre-flight tests for all models."""
        self.results = {}
        for model in models:
            self.results[model] = await self.run_model_test(model)
        return self.results

    def get_passed_models(self) -> list[str]:
        """Return list of models that passed pre-flight."""
        return [m for m, r in self.results.items() if r is True]

    def get_failed_models(self) -> list[str]:
        """Return list of models that failed pre-flight."""
        return [m for m, r in self.results.items() if r is False]

    def get_summary(self) -> str:
        """Get a summary of pre-flight results."""
        passed = len(self.get_passed_models())
        failed = len(self.get_failed_models())
        total = len(self.results)
        return f"Pre-flight complete: {passed}/{total} passed, {failed} failed"
