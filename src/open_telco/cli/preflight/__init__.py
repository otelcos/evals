"""Pre-flight testing package for model validation."""

from open_telco.cli.preflight.edge_cases import (
    BOXED_PATTERN,
    EDGE_CASE_PROMPTS,
    EdgeCasePrompt,
    run_edge_case_validation,
    validate_boxed_response,
)
from open_telco.cli.preflight.runner import (
    PreflightConfig,
    PreflightResult,
    PreflightRunner,
    PreflightStatus,
)
from open_telco.cli.preflight.stress_test import (
    STRESS_TEST_PROMPTS,
    StressTestResult,
    run_stress_tests,
    run_stress_tests_sync,
)

__all__ = [
    "BOXED_PATTERN",
    "EDGE_CASE_PROMPTS",
    "EdgeCasePrompt",
    "PreflightConfig",
    "PreflightResult",
    "PreflightRunner",
    "PreflightStatus",
    "STRESS_TEST_PROMPTS",
    "StressTestResult",
    "run_edge_case_validation",
    "run_stress_tests",
    "run_stress_tests_sync",
    "validate_boxed_response",
]
