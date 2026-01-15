"""Tests for pre-flight edge case validation functionality."""

import pytest

from evals.cli.preflight import (
    EDGE_CASE_PROMPTS,
    EdgeCasePrompt,
    PreflightConfig,
    PreflightStatus,
    run_edge_case_validation,
    validate_boxed_response,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def boxed_format_prompt() -> EdgeCasePrompt:
    """EdgeCasePrompt configured for boxed_format validation."""
    return EdgeCasePrompt(
        name="test",
        prompt="Test prompt",
        expected_behavior="boxed_format",
    )


@pytest.fixture
def numeric_prompt() -> EdgeCasePrompt:
    """EdgeCasePrompt configured for numeric validation."""
    return EdgeCasePrompt(
        name="test",
        prompt="Test prompt",
        expected_behavior="numeric",
    )


@pytest.fixture
def boxed_format_with_content_required() -> EdgeCasePrompt:
    """EdgeCasePrompt requiring non-empty boxed content."""
    return EdgeCasePrompt(
        name="test",
        prompt="Test prompt",
        expected_behavior="boxed_format",
        should_have_content=True,
    )


@pytest.fixture
def boxed_format_empty_allowed() -> EdgeCasePrompt:
    """EdgeCasePrompt allowing empty boxed content."""
    return EdgeCasePrompt(
        name="test",
        prompt="Test prompt",
        expected_behavior="boxed_format",
        should_have_content=False,
    )


# ============================================================================
# TestBoxedValidation
# ============================================================================


class TestBoxedValidation:
    r"""Test \boxed{} format validation."""

    def test_basic_boxed_returns_valid_true(self) -> None:
        """Test basic boxed format returns valid=True."""
        response = "The answer is \\boxed{42}"
        valid, _ = validate_boxed_response(response)
        assert valid

    def test_basic_boxed_extracts_correct_content(self) -> None:
        """Test basic boxed format extracts the correct content."""
        response = "The answer is \\boxed{42}"
        _, content = validate_boxed_response(response)
        assert content == "42"

    def test_boxed_with_text_around_returns_valid_true(self) -> None:
        """Test boxed format with surrounding text returns valid=True."""
        response = "After calculation, I get \\boxed{100} as the result."
        valid, _ = validate_boxed_response(response)
        assert valid

    def test_boxed_with_text_around_extracts_correct_content(self) -> None:
        """Test boxed format with surrounding text extracts the correct content."""
        response = "After calculation, I get \\boxed{100} as the result."
        _, content = validate_boxed_response(response)
        assert content == "100"

    def test_nested_braces_returns_valid_true(self) -> None:
        """Test boxed format with nested braces returns valid=True."""
        response = "Result: \\boxed{{nested}}"
        valid, _ = validate_boxed_response(response)
        assert valid

    def test_nested_braces_extracts_inner_braces(self) -> None:
        """Test boxed format with nested braces extracts inner braces correctly."""
        response = "Result: \\boxed{{nested}}"
        _, content = validate_boxed_response(response)
        assert content == "{nested}"

    def test_multiple_boxed_returns_valid_true(self) -> None:
        """Test multiple boxed expressions returns valid=True."""
        response = "Step 1: \\boxed{5}, Final: \\boxed{10}"
        valid, _ = validate_boxed_response(response)
        assert valid

    def test_multiple_boxed_extracts_last_value(self) -> None:
        """Test multiple boxed expressions extracts the last value."""
        response = "Step 1: \\boxed{5}, Final: \\boxed{10}"
        _, content = validate_boxed_response(response)
        assert content == "10"

    def test_no_boxed_returns_valid_false(self) -> None:
        """Test response without boxed format returns valid=False."""
        response = "The answer is 42"
        valid, _ = validate_boxed_response(response)
        assert not valid

    def test_no_boxed_returns_error_message(self) -> None:
        """Test response without boxed format returns appropriate error message."""
        response = "The answer is 42"
        _, message = validate_boxed_response(response)
        assert "No \\boxed{}" in message

    def test_empty_response_returns_valid_false(self) -> None:
        """Test empty response returns valid=False."""
        response = ""
        valid, _ = validate_boxed_response(response)
        assert not valid

    def test_empty_response_returns_error_message(self) -> None:
        """Test empty response returns appropriate error message."""
        response = ""
        _, message = validate_boxed_response(response)
        assert "Empty response" in message

    def test_boxed_with_whitespace_returns_valid_true(self) -> None:
        """Test boxed with whitespace content returns valid=True."""
        response = "\\boxed{  answer  }"
        valid, _ = validate_boxed_response(response)
        assert valid

    def test_boxed_with_whitespace_trims_content(self) -> None:
        """Test boxed with whitespace content trims the whitespace."""
        response = "\\boxed{  answer  }"
        _, content = validate_boxed_response(response)
        assert content == "answer"


# ============================================================================
# TestEdgeCaseValidation
# ============================================================================


class TestEdgeCaseValidation:
    """Test edge case prompt validation."""

    def test_valid_boxed_format_returns_valid_true(
        self, boxed_format_prompt: EdgeCasePrompt
    ) -> None:
        """Test validation of valid boxed format response returns valid=True."""
        response = "The answer is \\boxed{42}"
        valid, _ = run_edge_case_validation(response, boxed_format_prompt)
        assert valid

    def test_valid_boxed_format_explanation_contains_valid(
        self, boxed_format_prompt: EdgeCasePrompt
    ) -> None:
        """Test validation of valid boxed format response explanation contains 'Valid'."""
        response = "The answer is \\boxed{42}"
        _, explanation = run_edge_case_validation(response, boxed_format_prompt)
        assert "Valid boxed format" in explanation

    def test_invalid_boxed_format(self, boxed_format_prompt: EdgeCasePrompt) -> None:
        """Test validation of invalid boxed format response."""
        response = "The answer is 42"
        valid, _ = run_edge_case_validation(response, boxed_format_prompt)
        assert not valid

    def test_numeric_validation_success_returns_valid_true(
        self, numeric_prompt: EdgeCasePrompt
    ) -> None:
        """Test numeric validation with valid response returns valid=True."""
        response = "\\boxed{42}"
        valid, _ = run_edge_case_validation(response, numeric_prompt)
        assert valid

    def test_numeric_validation_success_explanation_contains_valid(
        self, numeric_prompt: EdgeCasePrompt
    ) -> None:
        """Test numeric validation with valid response explanation contains 'Valid'."""
        response = "\\boxed{42}"
        _, explanation = run_edge_case_validation(response, numeric_prompt)
        assert "Valid numeric format" in explanation

    def test_numeric_validation_failure_returns_valid_false(
        self, numeric_prompt: EdgeCasePrompt
    ) -> None:
        """Test numeric validation with non-numeric response returns valid=False."""
        response = "\\boxed{abc}"
        valid, _ = run_edge_case_validation(response, numeric_prompt)
        assert not valid

    def test_numeric_validation_failure_explanation_contains_expected_numeric(
        self, numeric_prompt: EdgeCasePrompt
    ) -> None:
        """Test numeric validation failure explanation contains 'Expected numeric'."""
        response = "\\boxed{abc}"
        _, explanation = run_edge_case_validation(response, numeric_prompt)
        assert "Expected numeric" in explanation

    def test_empty_content_when_expected_returns_valid_false(
        self, boxed_format_with_content_required: EdgeCasePrompt
    ) -> None:
        """Test empty boxed content when should_have_content=True returns valid=False."""
        response = "\\boxed{}"
        valid, _ = run_edge_case_validation(
            response, boxed_format_with_content_required
        )
        assert not valid

    def test_empty_content_when_expected_explanation_contains_non_empty(
        self, boxed_format_with_content_required: EdgeCasePrompt
    ) -> None:
        """Test empty content when expected explanation contains 'non-empty'."""
        response = "\\boxed{}"
        _, explanation = run_edge_case_validation(
            response, boxed_format_with_content_required
        )
        assert "non-empty content" in explanation

    def test_empty_content_allowed(
        self, boxed_format_empty_allowed: EdgeCasePrompt
    ) -> None:
        """Test empty boxed content when should_have_content=False passes."""
        response = "\\boxed{}"
        valid, _ = run_edge_case_validation(response, boxed_format_empty_allowed)
        assert valid


# ============================================================================
# TestPreflightConfig
# ============================================================================


class TestPreflightConfig:
    """Test pre-flight configuration."""

    def test_default_config_sample_limit(self) -> None:
        """Test default configuration has sample_limit=5."""
        config = PreflightConfig()
        assert config.sample_limit == 5

    def test_default_config_timeout_seconds(self) -> None:
        """Test default configuration has timeout_seconds=120."""
        config = PreflightConfig()
        assert config.timeout_seconds == 120

    def test_default_config_task_count(self) -> None:
        """Test default configuration has 2 tasks."""
        config = PreflightConfig()
        assert len(config.tasks) == 2

    def test_default_config_contains_telelogs_task(self) -> None:
        """Test default configuration contains telelogs task."""
        config = PreflightConfig()
        assert "telelogs/telelogs.py" in config.tasks

    def test_custom_config_sample_limit(self) -> None:
        """Test custom configuration sample_limit is set correctly."""
        config = PreflightConfig(
            sample_limit=3,
            timeout_seconds=60,
            tasks=["task1.py", "task2.py", "task3.py"],
        )
        assert config.sample_limit == 3

    def test_custom_config_timeout_seconds(self) -> None:
        """Test custom configuration timeout_seconds is set correctly."""
        config = PreflightConfig(
            sample_limit=3,
            timeout_seconds=60,
            tasks=["task1.py", "task2.py", "task3.py"],
        )
        assert config.timeout_seconds == 60

    def test_custom_config_task_count(self) -> None:
        """Test custom configuration task count is set correctly."""
        config = PreflightConfig(
            sample_limit=3,
            timeout_seconds=60,
            tasks=["task1.py", "task2.py", "task3.py"],
        )
        assert len(config.tasks) == 3


# ============================================================================
# TestPreflightStatus
# ============================================================================


class TestPreflightStatus:
    """Test pre-flight status enum."""

    @pytest.mark.parametrize(
        ("status", "expected_value"),
        [
            pytest.param(PreflightStatus.PENDING, "pending", id="pending"),
            pytest.param(PreflightStatus.RUNNING, "running", id="running"),
            pytest.param(PreflightStatus.PASSED, "passed", id="passed"),
            pytest.param(PreflightStatus.FAILED, "failed", id="failed"),
            pytest.param(PreflightStatus.TIMEOUT, "timeout", id="timeout"),
            pytest.param(PreflightStatus.SKIPPED, "skipped", id="skipped"),
        ],
    )
    def test_status_has_correct_value(
        self, status: PreflightStatus, expected_value: str
    ) -> None:
        """Each PreflightStatus should have the correct string value."""
        assert status.value == expected_value


# ============================================================================
# TestEdgeCasePrompts
# ============================================================================


class TestEdgeCasePrompts:
    """Test the predefined edge case prompts."""

    def test_edge_case_prompts_exist(self) -> None:
        """Test that edge case prompts are defined."""
        assert len(EDGE_CASE_PROMPTS) > 0

    @pytest.mark.parametrize(
        "prompt",
        [pytest.param(p, id=p.name) for p in EDGE_CASE_PROMPTS],
    )
    def test_prompt_has_name(self, prompt: EdgeCasePrompt) -> None:
        """Each edge case prompt should have a name."""
        assert prompt.name

    @pytest.mark.parametrize(
        "prompt",
        [pytest.param(p, id=p.name) for p in EDGE_CASE_PROMPTS],
    )
    def test_prompt_has_prompt_text(self, prompt: EdgeCasePrompt) -> None:
        """Each edge case prompt should have prompt text."""
        assert prompt.prompt

    @pytest.mark.parametrize(
        "prompt",
        [pytest.param(p, id=p.name) for p in EDGE_CASE_PROMPTS],
    )
    def test_prompt_has_valid_expected_behavior(self, prompt: EdgeCasePrompt) -> None:
        """Each edge case prompt should have valid expected_behavior."""
        assert prompt.expected_behavior in ("boxed_format", "numeric", "pattern_match")

    def test_basic_boxed_prompt_exists(self) -> None:
        """Test that basic boxed prompt exists."""
        names = [p.name for p in EDGE_CASE_PROMPTS]
        assert "basic_boxed" in names
