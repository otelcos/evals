"""Pre-flight testing package for model validation."""

from open_telco.cli.preflight.edge_cases import (
    BOXED_PATTERN,
    EDGE_CASE_PROMPTS,
    EdgeCasePrompt,
    run_edge_case_validation,
    validate_boxed_response,
)
from open_telco.cli.preflight.runner import (
    EvalFunc,
    PreflightConfig,
    PreflightRunner,
    PreflightStatus,
)

__all__ = [
    "BOXED_PATTERN",
    "EDGE_CASE_PROMPTS",
    "EdgeCasePrompt",
    "EvalFunc",
    "PreflightConfig",
    "PreflightRunner",
    "PreflightStatus",
    "run_edge_case_validation",
    "validate_boxed_response",
]
