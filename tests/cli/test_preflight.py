"""Tests for pre-flight testing functionality."""

import pytest

from open_telco.cli.preflight import (
    EDGE_CASE_PROMPTS,
    EdgeCasePrompt,
    PreflightConfig,
    PreflightStatus,
    run_edge_case_validation,
    validate_boxed_response,
)


class TestBoxedValidation:
    """Test \\boxed{} format validation."""

    def test_basic_boxed(self) -> None:
        """Test basic boxed format extraction."""
        response = "The answer is \\boxed{42}"
        valid, content = validate_boxed_response(response)
        assert valid
        assert content == "42"

    def test_boxed_with_text_around(self) -> None:
        """Test boxed format with surrounding text."""
        response = "After calculation, I get \\boxed{100} as the result."
        valid, content = validate_boxed_response(response)
        assert valid
        assert content == "100"

    def test_nested_braces(self) -> None:
        """Test boxed format with nested braces."""
        response = "Result: \\boxed{{nested}}"
        valid, content = validate_boxed_response(response)
        assert valid
        assert content == "{nested}"

    def test_multiple_boxed_takes_last(self) -> None:
        """Test that multiple boxed extracts the last one."""
        response = "Step 1: \\boxed{5}, Final: \\boxed{10}"
        valid, content = validate_boxed_response(response)
        assert valid
        assert content == "10"

    def test_no_boxed_fails(self) -> None:
        """Test that response without boxed fails."""
        response = "The answer is 42"
        valid, message = validate_boxed_response(response)
        assert not valid
        assert "No \\boxed{}" in message

    def test_empty_response(self) -> None:
        """Test that empty response fails."""
        response = ""
        valid, message = validate_boxed_response(response)
        assert not valid
        assert "Empty response" in message

    def test_boxed_with_whitespace(self) -> None:
        """Test boxed with whitespace content."""
        response = "\\boxed{  answer  }"
        valid, content = validate_boxed_response(response)
        assert valid
        assert content == "answer"


class TestEdgeCaseValidation:
    """Test edge case prompt validation."""

    def test_valid_boxed_format(self) -> None:
        """Test validation of valid boxed format response."""
        edge_case = EdgeCasePrompt(
            name="test",
            prompt="Test prompt",
            expected_behavior="boxed_format",
        )
        response = "The answer is \\boxed{42}"
        valid, explanation = run_edge_case_validation(response, edge_case)
        assert valid
        assert "Valid boxed format" in explanation

    def test_invalid_boxed_format(self) -> None:
        """Test validation of invalid boxed format response."""
        edge_case = EdgeCasePrompt(
            name="test",
            prompt="Test prompt",
            expected_behavior="boxed_format",
        )
        response = "The answer is 42"
        valid, explanation = run_edge_case_validation(response, edge_case)
        assert not valid

    def test_numeric_validation_success(self) -> None:
        """Test numeric validation with valid response."""
        edge_case = EdgeCasePrompt(
            name="test",
            prompt="Test prompt",
            expected_behavior="numeric",
        )
        response = "\\boxed{42}"
        valid, explanation = run_edge_case_validation(response, edge_case)
        assert valid
        assert "Valid numeric format" in explanation

    def test_numeric_validation_failure(self) -> None:
        """Test numeric validation with non-numeric response."""
        edge_case = EdgeCasePrompt(
            name="test",
            prompt="Test prompt",
            expected_behavior="numeric",
        )
        response = "\\boxed{abc}"
        valid, explanation = run_edge_case_validation(response, edge_case)
        assert not valid
        assert "Expected numeric" in explanation

    def test_empty_content_when_expected(self) -> None:
        """Test empty boxed content when should_have_content is True."""
        edge_case = EdgeCasePrompt(
            name="test",
            prompt="Test prompt",
            expected_behavior="boxed_format",
            should_have_content=True,
        )
        response = "\\boxed{}"
        valid, explanation = run_edge_case_validation(response, edge_case)
        assert not valid
        assert "non-empty content" in explanation

    def test_empty_content_allowed(self) -> None:
        """Test empty boxed content when should_have_content is False."""
        edge_case = EdgeCasePrompt(
            name="test",
            prompt="Test prompt",
            expected_behavior="boxed_format",
            should_have_content=False,
        )
        response = "\\boxed{}"
        valid, explanation = run_edge_case_validation(response, edge_case)
        assert valid


class TestPreflightConfig:
    """Test pre-flight configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PreflightConfig()
        assert config.sample_limit == 5
        assert config.timeout_seconds == 120
        assert len(config.tasks) == 2
        assert "telelogs/telelogs.py" in config.tasks

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = PreflightConfig(
            sample_limit=3,
            timeout_seconds=60,
            tasks=["task1.py", "task2.py", "task3.py"],
        )
        assert config.sample_limit == 3
        assert config.timeout_seconds == 60
        assert len(config.tasks) == 3


class TestPreflightStatus:
    """Test pre-flight status enum."""

    def test_all_statuses_exist(self) -> None:
        """Test that all expected statuses exist."""
        assert PreflightStatus.PENDING.value == "pending"
        assert PreflightStatus.RUNNING.value == "running"
        assert PreflightStatus.PASSED.value == "passed"
        assert PreflightStatus.FAILED.value == "failed"
        assert PreflightStatus.TIMEOUT.value == "timeout"
        assert PreflightStatus.SKIPPED.value == "skipped"


class TestEdgeCasePrompts:
    """Test the predefined edge case prompts."""

    def test_edge_case_prompts_exist(self) -> None:
        """Test that edge case prompts are defined."""
        assert len(EDGE_CASE_PROMPTS) > 0

    def test_all_prompts_have_required_fields(self) -> None:
        """Test that all prompts have required fields."""
        for prompt in EDGE_CASE_PROMPTS:
            assert prompt.name
            assert prompt.prompt
            assert prompt.expected_behavior in ("boxed_format", "numeric", "pattern_match")

    def test_basic_boxed_prompt_exists(self) -> None:
        """Test that basic boxed prompt exists."""
        names = [p.name for p in EDGE_CASE_PROMPTS]
        assert "basic_boxed" in names


class TestStressTestEnvPath:
    """Test stress test .env path configuration."""

    def test_env_path_calculation_is_correct(self) -> None:
        """Verify the .env path calculation in stress_test.py finds project root."""
        from pathlib import Path

        # Simulate the path calculation from stress_test.py
        stress_test_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "open_telco"
            / "cli"
            / "preflight"
            / "stress_test.py"
        )

        # The calculation in stress_test.py: Path(__file__).parent.parent.parent.parent.parent
        # From stress_test.py: cli/preflight/stress_test.py
        # .parent = preflight/
        # .parent.parent = cli/
        # .parent.parent.parent = open_telco/
        # .parent.parent.parent.parent = src/
        # .parent.parent.parent.parent.parent = project_root
        project_root = stress_test_path.parent.parent.parent.parent.parent

        assert project_root.name == "open_telco", (
            f"Expected directory name 'open_telco' (project root), got '{project_root.name}'"
        )

        # Note: .env may not exist in CI, so we just verify the path calculation is correct
        # by checking that pyproject.toml exists (which is always present)
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), (
            f"pyproject.toml not found at {pyproject_path}. "
            f"The project root calculation may be wrong."
        )

    def test_stress_test_prompts_are_defined(self) -> None:
        """Verify stress test prompts are properly defined."""
        from open_telco.cli.preflight import STRESS_TEST_PROMPTS

        assert len(STRESS_TEST_PROMPTS) >= 4, "Should have at least 4 stress test prompts"

        for prompt in STRESS_TEST_PROMPTS:
            assert "name" in prompt, "Each prompt should have a name"
            assert "prompt" in prompt, "Each prompt should have a prompt"
            assert "timeout" in prompt, "Each prompt should have a timeout"
            assert isinstance(prompt["timeout"], int), "Timeout should be an integer"
            assert prompt["timeout"] > 0, "Timeout should be positive"
