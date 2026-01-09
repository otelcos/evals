"""Pre-flight model testing runner."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PreflightStatus(Enum):
    """Status of a pre-flight test."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class PreflightResult:
    """Result of a pre-flight test."""

    model: str
    status: PreflightStatus
    error_message: str | None = None
    samples_completed: int = 0
    format_errors: int = 0
    duration_seconds: float = 0.0


@dataclass
class PreflightConfig:
    """Configuration for pre-flight tests."""

    sample_limit: int = 5
    timeout_seconds: int = 120  # 2 minutes per model
    tasks: list[str] = field(
        default_factory=lambda: [
            "telelogs/telelogs.py",
            "teleqna/teleqna.py",
        ]
    )
    log_dir: str = "logs/preflight"


class PreflightRunner:
    """Runs pre-flight tests on models before full evaluation."""

    DEFAULT_TASKS = [
        "telelogs/telelogs.py",
        "teleqna/teleqna.py",
    ]

    def __init__(
        self,
        config: PreflightConfig | None = None,
        on_progress: Callable[[str, PreflightStatus], None] | None = None,
    ) -> None:
        """Initialize the pre-flight runner.

        Args:
            config: Configuration for pre-flight tests.
            on_progress: Callback for progress updates (model, status).
        """
        self.config = config or PreflightConfig()
        self.on_progress = on_progress
        self.results: dict[str, PreflightResult] = {}
        self._cancelled = False

    def cancel(self) -> None:
        """Cancel the pre-flight tests."""
        self._cancelled = True

    def _notify_progress(self, model: str, status: PreflightStatus) -> None:
        """Notify progress callback if set."""
        if self.on_progress:
            self.on_progress(model, status)

    def _execute_eval_sync(self, model: str) -> tuple[bool, Any]:
        """Execute evaluation synchronously.

        This runs in a thread pool to avoid blocking the event loop.
        """
        from inspect_ai import eval_set

        success, logs = eval_set(
            tasks=self.config.tasks,
            model=[model],
            limit=self.config.sample_limit,
            log_dir=self.config.log_dir,
            epochs=1,
            temperature=0.0,
        )
        return success, logs

    async def run_model_test(self, model: str) -> PreflightResult:
        """Run pre-flight test for a single model with timeout.

        Args:
            model: The model identifier (e.g., "openrouter/openai/gpt-4o").

        Returns:
            PreflightResult with the test outcome.
        """
        if self._cancelled:
            return PreflightResult(
                model=model,
                status=PreflightStatus.SKIPPED,
                error_message="Test cancelled",
            )

        self._notify_progress(model, PreflightStatus.RUNNING)
        start_time = time.time()

        try:
            # Run eval_set in a thread pool with timeout
            loop = asyncio.get_event_loop()
            success, logs = await asyncio.wait_for(
                loop.run_in_executor(None, self._execute_eval_sync, model),
                timeout=self.config.timeout_seconds,
            )

            duration = time.time() - start_time

            if success:
                result = PreflightResult(
                    model=model,
                    status=PreflightStatus.PASSED,
                    samples_completed=self.config.sample_limit * len(self.config.tasks),
                    duration_seconds=duration,
                )
            else:
                result = PreflightResult(
                    model=model,
                    status=PreflightStatus.FAILED,
                    error_message="Evaluation returned failure",
                    duration_seconds=duration,
                )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            result = PreflightResult(
                model=model,
                status=PreflightStatus.TIMEOUT,
                error_message=f"Model timed out after {self.config.timeout_seconds}s",
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            result = PreflightResult(
                model=model,
                status=PreflightStatus.FAILED,
                error_message=str(e),
                duration_seconds=duration,
            )

        self._notify_progress(model, result.status)
        return result

    async def run_all(self, models: list[str]) -> dict[str, PreflightResult]:
        """Run pre-flight tests for all models sequentially.

        Args:
            models: List of model identifiers to test.

        Returns:
            Dictionary mapping model names to their results.
        """
        self.results = {}

        for model in models:
            if self._cancelled:
                self.results[model] = PreflightResult(
                    model=model,
                    status=PreflightStatus.SKIPPED,
                    error_message="Test cancelled",
                )
            else:
                result = await self.run_model_test(model)
                self.results[model] = result

        return self.results

    def get_passed_models(self) -> list[str]:
        """Return list of models that passed pre-flight.

        Returns:
            List of model identifiers that passed testing.
        """
        return [
            model
            for model, result in self.results.items()
            if result.status == PreflightStatus.PASSED
        ]

    def get_failed_models(self) -> list[str]:
        """Return list of models that failed pre-flight.

        Returns:
            List of model identifiers that failed testing.
        """
        return [
            model
            for model, result in self.results.items()
            if result.status in (PreflightStatus.FAILED, PreflightStatus.TIMEOUT)
        ]

    def get_summary(self) -> str:
        """Get a summary of pre-flight results.

        Returns:
            Human-readable summary string.
        """
        passed = len(self.get_passed_models())
        failed = len(self.get_failed_models())
        total = len(self.results)
        return f"Pre-flight complete: {passed}/{total} passed, {failed} failed"
