"""Type definitions for CLI operations.

Provides consistent return schemas for functions that need
to return multiple values or status information.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepResult:
    """Result of a preflight step execution.

    Supports dict-like access for backwards compatibility with tests.
    """

    passed: bool
    error: str | None = None
    score: float | None = None

    def __getitem__(self, key: str) -> bool | str | float | None:
        return getattr(self, key)


@dataclass(frozen=True)
class FindKResult:
    """Result of find-k step with variance reduction metadata.

    Supports dict-like access for backwards compatibility with tests.
    """

    passed: bool
    optimal_k: int = 1
    variance_reduction: float = 0.0
    task_consistency: dict[str, list[bool]] | None = None
    observed_variance: float = 0.0
    error: str | None = None

    def __getitem__(self, key: str) -> bool | int | float | dict | str | None:
        return getattr(self, key)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Parsed model information."""

    provider: str
    model_name: str
    display_name: str
