"""Tests for GitHub service functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from open_telco.cli.screens.submit.github_service import GitHubService, PRResult


class TestGitHubServiceDirectAccess:
    """Test PR creation with direct write access."""

    def test_direct_access_pr_returns_success(
        self,
        direct_access_pr_result: tuple["PRResult", "GitHubService"],
    ) -> None:
        """PR creation with direct access should return success=True."""
        result, _ = direct_access_pr_result
        assert result.success is True

    def test_direct_access_pr_contains_url(
        self,
        direct_access_pr_result: tuple["PRResult", "GitHubService"],
    ) -> None:
        """PR creation with direct access should return PR URL."""
        result, _ = direct_access_pr_result
        assert "pull/123" in result.pr_url

    def test_direct_access_does_not_use_fork(
        self,
        direct_access_pr_result: tuple["PRResult", "GitHubService"],
    ) -> None:
        """PR creation with direct access should not use fork."""
        _, service = direct_access_pr_result
        assert service._use_fork is False


class TestGitHubServiceForkFallback:
    """Test PR creation via fork fallback."""

    def test_fork_fallback_pr_returns_success(
        self,
        fork_fallback_pr_result: tuple["PRResult", "GitHubService"],
    ) -> None:
        """PR creation via fork should return success=True."""
        result, _ = fork_fallback_pr_result
        assert result.success is True

    def test_fork_fallback_uses_fork(
        self,
        fork_fallback_pr_result: tuple["PRResult", "GitHubService"],
    ) -> None:
        """PR creation via fork should set _use_fork=True."""
        _, service = fork_fallback_pr_result
        assert service._use_fork is True


class TestGitHubServiceExistingPR:
    """Test PR creation when PR already exists."""

    def test_existing_pr_returns_success(
        self,
        existing_pr_result: "PRResult",
    ) -> None:
        """Existing PR scenario should return success=True."""
        assert existing_pr_result.success is True

    def test_existing_pr_returns_existing_url(
        self,
        existing_pr_result: "PRResult",
    ) -> None:
        """Existing PR scenario should return the existing PR URL."""
        assert "pull/99" in existing_pr_result.pr_url


class TestGitHubServiceHttpError:
    """Test PR creation with HTTP errors."""

    def test_http_error_returns_failure(
        self,
        http_error_pr_result: "PRResult",
    ) -> None:
        """HTTP error should return success=False."""
        assert http_error_pr_result.success is False

    def test_http_error_has_error_message(
        self,
        http_error_pr_result: "PRResult",
    ) -> None:
        """HTTP error should include error message."""
        assert http_error_pr_result.error is not None


class TestGitHubServiceBranchSanitization:
    """Test branch name sanitization."""

    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            pytest.param("gpt-4o", "gpt-4o", id="valid_hyphen_preserved"),
            pytest.param("gpt 4o", "gpt_4o", id="space_replaced"),
            pytest.param("gpt/4o", "gpt_4o", id="slash_replaced"),
            pytest.param("gpt@4o!", "gpt_4o", id="special_chars_replaced"),
            pytest.param("__gpt__", "gpt", id="leading_trailing_underscores_stripped"),
            pytest.param(
                "gpt-4o-2024-05-13", "gpt-4o-2024-05-13", id="date_suffix_preserved"
            ),
        ],
    )
    def test_branch_name_sanitization(
        self,
        github_service: "GitHubService",
        input_name: str,
        expected: str,
    ) -> None:
        """Branch name should be sanitized correctly."""
        assert github_service._sanitize_branch_name(input_name) == expected


class TestGitHubServicePRBody:
    """Test PR body generation."""

    @pytest.mark.parametrize(
        ("expected_content",),
        [
            pytest.param("gpt-4o", id="contains_model_name"),
            pytest.param("Openai", id="contains_provider"),
            pytest.param("Leaderboard Submission", id="contains_title"),
            pytest.param("model_cards", id="contains_model_cards_path"),
            pytest.param("trajectories", id="contains_trajectories_path"),
        ],
    )
    def test_generate_pr_body_contains_expected_content(
        self,
        github_service: "GitHubService",
        expected_content: str,
    ) -> None:
        """PR body should contain expected content."""
        body = github_service._generate_pr_body("gpt-4o", "Openai")
        assert expected_content in body


class TestGitHubErrorHandling401:
    """Test 401 unauthorized error handling."""

    def test_401_returns_failure(
        self,
        error_401_pr_result: "PRResult",
    ) -> None:
        """401 error should return success=False."""
        assert error_401_pr_result.success is False

    def test_401_has_error_message(
        self,
        error_401_pr_result: "PRResult",
    ) -> None:
        """401 error should include error message."""
        assert error_401_pr_result.error is not None


class TestGitHubErrorHandling403:
    """Test 403 forbidden error handling."""

    def test_403_returns_failure(
        self,
        error_403_pr_result: "PRResult",
    ) -> None:
        """403 error should return success=False."""
        assert error_403_pr_result.success is False

    def test_403_error_contains_rate_limit(
        self,
        error_403_pr_result: "PRResult",
    ) -> None:
        """403 error message should mention rate limit."""
        assert "rate limit" in error_403_pr_result.error.lower()


class TestGitHubErrorHandling404:
    """Test 404 not found error handling."""

    def test_404_returns_failure(
        self,
        error_404_pr_result: "PRResult",
    ) -> None:
        """404 error should return success=False."""
        assert error_404_pr_result.success is False

    def test_404_error_contains_not_found(
        self,
        error_404_pr_result: "PRResult",
    ) -> None:
        """404 error message should mention Not Found."""
        assert "Not Found" in error_404_pr_result.error


class TestGitHubErrorHandlingTimeout:
    """Test timeout error handling."""

    def test_timeout_returns_failure(
        self,
        error_timeout_pr_result: "PRResult",
    ) -> None:
        """Timeout error should return success=False."""
        assert error_timeout_pr_result.success is False

    def test_timeout_error_contains_timed_out(
        self,
        error_timeout_pr_result: "PRResult",
    ) -> None:
        """Timeout error message should mention timed out."""
        assert "timed out" in error_timeout_pr_result.error.lower()


class TestGitHubErrorHandlingRateLimit:
    """Test rate limit error handling."""

    def test_rate_limited_returns_failure(
        self,
        error_rate_limit_pr_result: "PRResult",
    ) -> None:
        """Rate limit error should return success=False."""
        assert error_rate_limit_pr_result.success is False

    def test_rate_limited_has_error_message(
        self,
        error_rate_limit_pr_result: "PRResult",
    ) -> None:
        """Rate limit error should include error message."""
        assert error_rate_limit_pr_result.error is not None


class TestPRResultSuccess:
    """Test successful PRResult dataclass."""

    def test_successful_pr_result_success_is_true(
        self, sample_pr_result: "PRResult"
    ) -> None:
        """Successful PRResult should have success=True."""
        assert sample_pr_result.success is True

    def test_successful_pr_result_has_url(self, sample_pr_result: "PRResult") -> None:
        """Successful PRResult should have a PR URL."""
        assert sample_pr_result.pr_url is not None

    def test_successful_pr_result_has_no_error(
        self, sample_pr_result: "PRResult"
    ) -> None:
        """Successful PRResult should have no error."""
        assert sample_pr_result.error is None


class TestPRResultFailure:
    """Test failed PRResult dataclass."""

    def test_failed_pr_result_success_is_false(
        self, sample_failed_pr_result: "PRResult"
    ) -> None:
        """Failed PRResult should have success=False."""
        assert sample_failed_pr_result.success is False

    def test_failed_pr_result_has_no_url(
        self, sample_failed_pr_result: "PRResult"
    ) -> None:
        """Failed PRResult should have no PR URL."""
        assert sample_failed_pr_result.pr_url is None

    def test_failed_pr_result_has_error(
        self, sample_failed_pr_result: "PRResult"
    ) -> None:
        """Failed PRResult should have an error message."""
        assert sample_failed_pr_result.error is not None
