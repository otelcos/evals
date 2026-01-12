"""Tests for GitHub service functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from open_telco.cli.screens.submit.github_service import GitHubService, PRResult


class TestGitHubService:
    """Test GitHub service core functionality."""

    def test_create_pr_with_direct_access(self) -> None:
        """PR should be created on main repo when user has write access."""
        with patch(
            "open_telco.cli.screens.submit.github_service.requests"
        ) as mock_requests:
            # Setup mocks
            service = GitHubService("test_token")

            # Mock user fetch
            user_resp = MagicMock()
            user_resp.status_code = 200
            user_resp.json.return_value = {"login": "testuser"}

            # Mock permission check - user has write access
            perm_resp = MagicMock()
            perm_resp.status_code = 200
            perm_resp.json.return_value = {"permission": "write"}

            # Mock branch ref
            ref_resp = MagicMock()
            ref_resp.status_code = 200
            ref_resp.json.return_value = {"object": {"sha": "abc123"}}

            # Mock branch check (doesn't exist)
            branch_check = MagicMock()
            branch_check.status_code = 404

            # Mock branch create
            branch_create = MagicMock()
            branch_create.status_code = 201

            # Mock file check (doesn't exist)
            file_check = MagicMock()
            file_check.status_code = 404

            # Mock file create
            file_create = MagicMock()
            file_create.status_code = 201

            # Mock PR check (no existing)
            pr_check = MagicMock()
            pr_check.status_code = 200
            pr_check.json.return_value = []

            # Mock PR create
            pr_create = MagicMock()
            pr_create.status_code = 201
            pr_create.json.return_value = {
                "html_url": "https://github.com/gsma-research/ot_leaderboard/pull/123",
            }

            def get_side_effect(url, **kwargs):
                if "/user" in url and "/permission" not in url:
                    return user_resp
                elif "/permission" in url:
                    return perm_resp
                elif "/git/refs/heads/main" in url:
                    return ref_resp
                elif "/git/refs/heads/submit" in url:
                    return branch_check
                elif "/contents/" in url:
                    return file_check
                elif "/pulls" in url:
                    return pr_check
                return ref_resp

            mock_requests.get.side_effect = get_side_effect
            mock_requests.post.return_value = pr_create
            mock_requests.put.return_value = file_create

            result = service.create_submission_pr(
                model_name="gpt-4o",
                provider="Openai",
                parquet_content=b"parquet_data",
                trajectory_files={"test.json": b"{}"},
            )

            assert result.success is True
            assert "pull/123" in result.pr_url
            # Should use gsma-research, not fork
            assert service._use_fork is False

    def test_create_pr_with_fork_fallback(self) -> None:
        """PR should be created via fork when user lacks write access."""
        with patch(
            "open_telco.cli.screens.submit.github_service.requests"
        ) as mock_requests:
            service = GitHubService("test_token")

            # Mock user fetch
            user_resp = MagicMock()
            user_resp.status_code = 200
            user_resp.json.return_value = {"login": "testuser"}

            # Mock permission check - no access
            perm_resp = MagicMock()
            perm_resp.status_code = 404

            # Mock fork check (doesn't exist)
            fork_check = MagicMock()
            fork_check.status_code = 404

            # Mock fork create
            fork_create = MagicMock()
            fork_create.status_code = 202
            fork_create.json.return_value = {
                "owner": {"login": "testuser"},
                "name": "ot_leaderboard",
            }

            # Mock branch ref
            ref_resp = MagicMock()
            ref_resp.status_code = 200
            ref_resp.json.return_value = {"object": {"sha": "abc123"}}

            # Mock branch check (doesn't exist)
            branch_check = MagicMock()
            branch_check.status_code = 404

            # Mock file check (doesn't exist)
            file_check = MagicMock()
            file_check.status_code = 404

            # Mock file create
            file_create = MagicMock()
            file_create.status_code = 201

            # Mock PR check (no existing)
            pr_check = MagicMock()
            pr_check.status_code = 200
            pr_check.json.return_value = []

            # Mock PR create
            pr_create = MagicMock()
            pr_create.status_code = 201
            pr_create.json.return_value = {
                "html_url": "https://github.com/gsma-research/ot_leaderboard/pull/123",
            }

            def get_side_effect(url, **kwargs):
                if "/user" in url and "/permission" not in url:
                    return user_resp
                elif "/permission" in url:
                    return perm_resp
                elif (
                    "/repos/testuser/ot_leaderboard" in url
                    and "/git/" not in url
                    and "/contents/" not in url
                ):
                    return fork_check
                elif "/git/refs/heads/main" in url:
                    return ref_resp
                elif "/git/refs/heads/submit" in url:
                    return branch_check
                elif "/contents/" in url:
                    return file_check
                elif "/pulls" in url:
                    return pr_check
                return ref_resp

            def post_side_effect(url, **kwargs):
                if "/forks" in url:
                    return fork_create
                elif "/pulls" in url:
                    return pr_create
                return MagicMock(status_code=201)

            mock_requests.get.side_effect = get_side_effect
            mock_requests.post.side_effect = post_side_effect
            mock_requests.put.return_value = file_create

            result = service.create_submission_pr(
                model_name="gpt-4o",
                provider="Openai",
                parquet_content=b"parquet_data",
                trajectory_files={"test.json": b"{}"},
            )

            assert result.success is True
            assert service._use_fork is True

    def test_existing_pr_returns_existing(self) -> None:
        """Should return existing PR URL if PR already exists (idempotent)."""
        with patch(
            "open_telco.cli.screens.submit.github_service.requests"
        ) as mock_requests:
            service = GitHubService("test_token")

            # Mock user fetch
            user_resp = MagicMock()
            user_resp.status_code = 200
            user_resp.json.return_value = {"login": "testuser"}

            # Mock permission check - has access
            perm_resp = MagicMock()
            perm_resp.status_code = 200
            perm_resp.json.return_value = {"permission": "write"}

            # Mock branch ref
            ref_resp = MagicMock()
            ref_resp.status_code = 200
            ref_resp.json.return_value = {"object": {"sha": "abc123"}}

            # Mock branch already exists
            branch_check = MagicMock()
            branch_check.status_code = 200

            # Mock file check
            file_check = MagicMock()
            file_check.status_code = 404

            file_create = MagicMock()
            file_create.status_code = 201

            # Mock PR check - PR already exists
            pr_check = MagicMock()
            pr_check.status_code = 200
            pr_check.json.return_value = [
                {
                    "html_url": "https://github.com/gsma-research/ot_leaderboard/pull/99",
                    "number": 99,
                }
            ]

            def get_side_effect(url, **kwargs):
                if "/user" in url and "/permission" not in url:
                    return user_resp
                elif "/permission" in url:
                    return perm_resp
                elif "/git/refs/heads/main" in url:
                    return ref_resp
                elif "/git/refs/heads/submit" in url:
                    return branch_check
                elif "/contents/" in url:
                    return file_check
                elif "/pulls" in url:
                    return pr_check
                return ref_resp

            mock_requests.get.side_effect = get_side_effect
            mock_requests.patch.return_value = MagicMock(status_code=200)
            mock_requests.put.return_value = file_create

            result = service.create_submission_pr(
                model_name="gpt-4o",
                provider="Openai",
                parquet_content=b"parquet_data",
                trajectory_files={},
            )

            assert result.success is True
            # Should return existing PR
            assert "pull/99" in result.pr_url

    def test_http_error_returns_pr_result_error(self) -> None:
        """HTTP errors should return PRResult with error message."""
        with patch(
            "open_telco.cli.screens.submit.github_service.requests.get"
        ) as mock_get:
            service = GitHubService("test_token")

            # Mock user fetch fails
            error_resp = MagicMock()
            error_resp.status_code = 401
            error_resp.json.return_value = {"message": "Bad credentials"}
            error_resp.raise_for_status.side_effect = requests.HTTPError(
                response=error_resp
            )

            mock_get.return_value = error_resp

            result = service.create_submission_pr(
                model_name="gpt-4o",
                provider="Openai",
                parquet_content=b"parquet_data",
                trajectory_files={},
            )

            assert result.success is False
            assert result.error is not None

    def test_branch_name_sanitization(self) -> None:
        """Invalid characters should be replaced with underscores."""
        service = GitHubService("test_token")

        # Test various inputs
        assert service._sanitize_branch_name("gpt-4o") == "gpt-4o"
        assert service._sanitize_branch_name("gpt 4o") == "gpt_4o"
        assert service._sanitize_branch_name("gpt/4o") == "gpt_4o"
        assert service._sanitize_branch_name("gpt@4o!") == "gpt_4o"
        assert service._sanitize_branch_name("__gpt__") == "gpt"
        assert service._sanitize_branch_name("gpt-4o-2024-05-13") == "gpt-4o-2024-05-13"

    def test_generate_pr_body(self) -> None:
        """PR body should contain model and provider info."""
        service = GitHubService("test_token")

        body = service._generate_pr_body("gpt-4o", "Openai")

        assert "gpt-4o" in body
        assert "Openai" in body
        assert "Leaderboard Submission" in body
        assert "model_cards" in body
        assert "trajectories" in body


class TestGitHubErrorHandling:
    """Test GitHub API error handling."""

    def test_github_api_401_unauthorized(self) -> None:
        """Should handle 401 unauthorized error."""
        with patch(
            "open_telco.cli.screens.submit.github_service.requests.get"
        ) as mock_get:
            service = GitHubService("invalid_token")

            error_resp = MagicMock()
            error_resp.status_code = 401
            error_resp.json.return_value = {"message": "Bad credentials"}
            error_resp.raise_for_status.side_effect = requests.HTTPError(
                response=error_resp
            )

            mock_get.return_value = error_resp

            result = service.create_submission_pr(
                model_name="gpt-4o",
                provider="Openai",
                parquet_content=b"data",
                trajectory_files={},
            )

            assert result.success is False
            assert result.error is not None

    def test_github_api_403_forbidden(self) -> None:
        """Should handle 403 forbidden error."""
        with patch(
            "open_telco.cli.screens.submit.github_service.requests.get"
        ) as mock_get:
            service = GitHubService("test_token")

            # First call succeeds (user fetch)
            user_resp = MagicMock()
            user_resp.status_code = 200
            user_resp.json.return_value = {"login": "testuser"}

            # Permission check succeeds
            perm_resp = MagicMock()
            perm_resp.status_code = 200
            perm_resp.json.return_value = {"permission": "write"}

            # Branch ref fails with 403
            error_resp = MagicMock()
            error_resp.status_code = 403
            error_resp.json.return_value = {"message": "API rate limit exceeded"}
            error_resp.raise_for_status.side_effect = requests.HTTPError(
                response=error_resp
            )

            def get_side_effect(url, **kwargs):
                if "/user" in url and "/permission" not in url:
                    return user_resp
                elif "/permission" in url:
                    return perm_resp
                else:
                    return error_resp

            mock_get.side_effect = get_side_effect

            result = service.create_submission_pr(
                model_name="gpt-4o",
                provider="Openai",
                parquet_content=b"data",
                trajectory_files={},
            )

            assert result.success is False
            assert "rate limit" in result.error.lower()

    def test_github_api_404_repo_not_found(self) -> None:
        """Should handle 404 not found error."""
        with (
            patch(
                "open_telco.cli.screens.submit.github_service.requests.get"
            ) as mock_get,
            patch(
                "open_telco.cli.screens.submit.github_service.requests.post"
            ) as mock_post,
        ):
            service = GitHubService("test_token")

            # Mock user fetch
            user_resp = MagicMock()
            user_resp.status_code = 200
            user_resp.json.return_value = {"login": "testuser"}

            # Permission check returns 404
            perm_resp = MagicMock()
            perm_resp.status_code = 404
            # This is expected - means no direct access

            # Fork check also 404
            fork_check = MagicMock()
            fork_check.status_code = 404

            # Fork creation fails with 404
            fork_fail = MagicMock()
            fork_fail.status_code = 404
            fork_fail.json.return_value = {"message": "Not Found"}
            fork_fail.raise_for_status.side_effect = requests.HTTPError(
                response=fork_fail
            )

            def get_side_effect(url, **kwargs):
                if "/user" in url and "/permission" not in url:
                    return user_resp
                elif "/permission" in url:
                    return perm_resp
                elif "/repos/testuser/ot_leaderboard" in url:
                    return fork_check
                return perm_resp

            mock_get.side_effect = get_side_effect
            mock_post.return_value = fork_fail

            result = service.create_submission_pr(
                model_name="gpt-4o",
                provider="Openai",
                parquet_content=b"data",
                trajectory_files={},
            )

            assert result.success is False
            assert "Not Found" in result.error

    def test_github_api_timeout(self) -> None:
        """Should handle request timeout."""
        with patch(
            "open_telco.cli.screens.submit.github_service.requests.get"
        ) as mock_get:
            service = GitHubService("test_token")

            mock_get.side_effect = requests.Timeout("Connection timed out")

            result = service.create_submission_pr(
                model_name="gpt-4o",
                provider="Openai",
                parquet_content=b"data",
                trajectory_files={},
            )

            assert result.success is False
            assert "timed out" in result.error.lower()

    def test_github_api_rate_limited(self) -> None:
        """Should handle rate limit error with message."""
        with patch(
            "open_telco.cli.screens.submit.github_service.requests.get"
        ) as mock_get:
            service = GitHubService("test_token")

            error_resp = MagicMock()
            error_resp.status_code = 403
            error_resp.json.return_value = {
                "message": "API rate limit exceeded for user"
            }
            error_resp.raise_for_status.side_effect = requests.HTTPError(
                response=error_resp
            )

            mock_get.return_value = error_resp

            result = service.create_submission_pr(
                model_name="gpt-4o",
                provider="Openai",
                parquet_content=b"data",
                trajectory_files={},
            )

            assert result.success is False
            assert result.error is not None


class TestPRResult:
    """Test PRResult dataclass."""

    def test_successful_pr_result(self, sample_pr_result: PRResult) -> None:
        """Successful PRResult should have URL and no error."""
        assert sample_pr_result.success is True
        assert sample_pr_result.pr_url is not None
        assert sample_pr_result.error is None

    def test_failed_pr_result(self, sample_failed_pr_result: PRResult) -> None:
        """Failed PRResult should have error and no URL."""
        assert sample_failed_pr_result.success is False
        assert sample_failed_pr_result.pr_url is None
        assert sample_failed_pr_result.error is not None
