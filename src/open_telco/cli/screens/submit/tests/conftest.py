"""Submit-specific test fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import requests

if TYPE_CHECKING:
    from open_telco.cli.screens.submit.github_service import GitHubService, PRResult


@pytest.fixture
def github_service(mock_github_token: str) -> "GitHubService":
    """Create a GitHubService instance with mocked token."""
    from open_telco.cli.screens.submit.github_service import GitHubService

    return GitHubService(token=mock_github_token)


@pytest.fixture
def default_submission_params() -> dict:
    """Default parameters for PR creation."""
    import io

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "model": "gpt-4o (Openai)",
                "teleqna": [83.6, 1.17, 1000.0],
                "date": "2026-01-09",
            }
        ]
    )
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)

    return {
        "model_name": "gpt-4o",
        "provider": "Openai",
        "parquet_content": buffer.getvalue(),
        "trajectory_files": {"eval.json": b'{"eval": {}}'},
    }


@pytest.fixture
def mock_direct_access_responses() -> MagicMock:
    """Mock responses for direct access PR creation (no fork needed)."""
    mock = MagicMock()

    # Mock successful branch creation (direct access)
    branch_response = MagicMock()
    branch_response.status_code = 201
    branch_response.json.return_value = {"ref": "refs/heads/submission/gpt-4o"}

    # Mock successful file creation
    file_response = MagicMock()
    file_response.status_code = 201

    # Mock successful PR creation
    pr_response = MagicMock()
    pr_response.status_code = 201
    pr_response.json.return_value = {
        "html_url": "https://github.com/gsma-research/ot_leaderboard/pull/123"
    }

    # Get default branch
    repo_response = MagicMock()
    repo_response.status_code = 200
    repo_response.json.return_value = {
        "default_branch": "main",
        "permissions": {"push": True},
    }

    # Get ref (for base SHA)
    ref_response = MagicMock()
    ref_response.status_code = 200
    ref_response.json.return_value = {"object": {"sha": "abc123"}}

    mock.get.side_effect = [repo_response, ref_response]
    mock.post.side_effect = [branch_response, file_response, file_response, pr_response]

    return mock


@pytest.fixture
def direct_access_pr_result(
    github_service: "GitHubService",
    default_submission_params: dict,
    mock_direct_access_responses: MagicMock,
) -> tuple["PRResult", "GitHubService"]:
    """PRResult from successful direct access PR creation."""
    from unittest.mock import patch

    with (
        patch(
            "open_telco.cli.screens.submit.github_service.requests.get",
            side_effect=mock_direct_access_responses.get.side_effect,
        ),
        patch(
            "open_telco.cli.screens.submit.github_service.requests.post",
            side_effect=mock_direct_access_responses.post.side_effect,
        ),
    ):
        result = github_service.create_submission_pr(**default_submission_params)
        return result, github_service


@pytest.fixture
def mock_fork_fallback_responses() -> MagicMock:
    """Mock responses for fork fallback PR creation (403 on direct, success via fork)."""
    mock = MagicMock()

    # First try direct access - fails with 403
    branch_fail = MagicMock()
    branch_fail.status_code = 403
    branch_fail.raise_for_status.side_effect = requests.HTTPError()

    # Fork creation succeeds
    fork_response = MagicMock()
    fork_response.status_code = 202
    fork_response.json.return_value = {
        "full_name": "user/ot_leaderboard",
        "default_branch": "main",
    }

    # Branch creation on fork succeeds
    branch_success = MagicMock()
    branch_success.status_code = 201

    # File uploads succeed
    file_response = MagicMock()
    file_response.status_code = 201

    # PR creation succeeds
    pr_response = MagicMock()
    pr_response.status_code = 201
    pr_response.json.return_value = {
        "html_url": "https://github.com/gsma-research/ot_leaderboard/pull/456"
    }

    # Get repo info
    repo_response = MagicMock()
    repo_response.status_code = 200
    repo_response.json.return_value = {
        "default_branch": "main",
        "permissions": {"push": False},
    }

    # Get ref
    ref_response = MagicMock()
    ref_response.status_code = 200
    ref_response.json.return_value = {"object": {"sha": "abc123"}}

    mock.get.side_effect = [repo_response, ref_response, ref_response]
    mock.post.side_effect = [
        fork_response,
        branch_success,
        file_response,
        file_response,
        pr_response,
    ]

    return mock


@pytest.fixture
def fork_fallback_pr_result(
    github_service: "GitHubService",
    default_submission_params: dict,
    mock_fork_fallback_responses: MagicMock,
) -> tuple["PRResult", "GitHubService"]:
    """PRResult from fork fallback PR creation."""
    from unittest.mock import patch

    with (
        patch(
            "open_telco.cli.screens.submit.github_service.requests.get",
            side_effect=mock_fork_fallback_responses.get.side_effect,
        ),
        patch(
            "open_telco.cli.screens.submit.github_service.requests.post",
            side_effect=mock_fork_fallback_responses.post.side_effect,
        ),
    ):
        result = github_service.create_submission_pr(**default_submission_params)
        return result, github_service


@pytest.fixture
def mock_existing_pr_responses() -> MagicMock:
    """Mock responses when PR already exists."""
    mock = MagicMock()

    # Get repo info
    repo_response = MagicMock()
    repo_response.status_code = 200
    repo_response.json.return_value = {
        "default_branch": "main",
        "permissions": {"push": True},
    }

    # Branch already exists (422)
    branch_exists = MagicMock()
    branch_exists.status_code = 422
    branch_exists.json.return_value = {"message": "Reference already exists"}

    # PR search returns existing PR
    search_response = MagicMock()
    search_response.status_code = 200
    search_response.json.return_value = [
        {"html_url": "https://github.com/gsma-research/ot_leaderboard/pull/99"}
    ]

    mock.get.side_effect = [repo_response, search_response]
    mock.post.side_effect = [branch_exists]

    return mock


@pytest.fixture
def existing_pr_result(
    github_service: "GitHubService",
    default_submission_params: dict,
    mock_existing_pr_responses: MagicMock,
) -> "PRResult":
    """PRResult when PR already exists."""
    from unittest.mock import patch

    with (
        patch(
            "open_telco.cli.screens.submit.github_service.requests.get",
            side_effect=mock_existing_pr_responses.get.side_effect,
        ),
        patch(
            "open_telco.cli.screens.submit.github_service.requests.post",
            side_effect=mock_existing_pr_responses.post.side_effect,
        ),
    ):
        return github_service.create_submission_pr(**default_submission_params)


@pytest.fixture
def mock_http_error_responses() -> MagicMock:
    """Mock responses for HTTP error scenario."""
    mock = MagicMock()

    # Get repo info fails
    error_response = MagicMock()
    error_response.status_code = 500
    error_response.raise_for_status.side_effect = requests.HTTPError("Server Error")

    mock.get.return_value = error_response

    return mock


@pytest.fixture
def http_error_pr_result(
    github_service: "GitHubService",
    default_submission_params: dict,
    mock_http_error_responses: MagicMock,
) -> "PRResult":
    """PRResult from HTTP error."""
    from unittest.mock import patch

    with patch(
        "open_telco.cli.screens.submit.github_service.requests.get",
        return_value=mock_http_error_responses.get.return_value,
    ):
        return github_service.create_submission_pr(**default_submission_params)


# Error-specific fixtures


@pytest.fixture
def mock_401_error_responses() -> MagicMock:
    """Mock 401 unauthorized error."""
    mock = MagicMock()
    response = MagicMock()
    response.status_code = 401
    response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
    mock.get.return_value = response
    return mock


@pytest.fixture
def error_401_pr_result(
    github_service: "GitHubService",
    default_submission_params: dict,
    mock_401_error_responses: MagicMock,
) -> "PRResult":
    """PRResult from 401 error."""
    from unittest.mock import patch

    with patch(
        "open_telco.cli.screens.submit.github_service.requests.get",
        return_value=mock_401_error_responses.get.return_value,
    ):
        return github_service.create_submission_pr(**default_submission_params)


@pytest.fixture
def mock_403_error_responses() -> MagicMock:
    """Mock 403 rate limit error."""
    mock = MagicMock()
    response = MagicMock()
    response.status_code = 403
    response.json.return_value = {"message": "API rate limit exceeded"}
    response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
    mock.get.return_value = response
    return mock


@pytest.fixture
def error_403_pr_result(
    github_service: "GitHubService",
    default_submission_params: dict,
    mock_403_error_responses: MagicMock,
) -> "PRResult":
    """PRResult from 403 rate limit error."""
    from unittest.mock import patch

    with patch(
        "open_telco.cli.screens.submit.github_service.requests.get",
        return_value=mock_403_error_responses.get.return_value,
    ):
        return github_service.create_submission_pr(**default_submission_params)


@pytest.fixture
def mock_404_error_responses() -> MagicMock:
    """Mock 404 not found error."""
    mock = MagicMock()
    response = MagicMock()
    response.status_code = 404
    response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock.get.return_value = response
    return mock


@pytest.fixture
def error_404_pr_result(
    github_service: "GitHubService",
    default_submission_params: dict,
    mock_404_error_responses: MagicMock,
) -> "PRResult":
    """PRResult from 404 error."""
    from unittest.mock import patch

    with patch(
        "open_telco.cli.screens.submit.github_service.requests.get",
        return_value=mock_404_error_responses.get.return_value,
    ):
        return github_service.create_submission_pr(**default_submission_params)


@pytest.fixture
def mock_timeout_error_responses() -> MagicMock:
    """Mock timeout error."""
    mock = MagicMock()
    mock.get.side_effect = requests.Timeout("Connection timed out")
    return mock


@pytest.fixture
def error_timeout_pr_result(
    github_service: "GitHubService",
    default_submission_params: dict,
    mock_timeout_error_responses: MagicMock,
) -> "PRResult":
    """PRResult from timeout error."""
    from unittest.mock import patch

    with patch(
        "open_telco.cli.screens.submit.github_service.requests.get",
        side_effect=mock_timeout_error_responses.get.side_effect,
    ):
        return github_service.create_submission_pr(**default_submission_params)


@pytest.fixture
def mock_rate_limit_error_responses() -> MagicMock:
    """Mock rate limit exceeded error."""
    mock = MagicMock()
    response = MagicMock()
    response.status_code = 429
    response.json.return_value = {"message": "Rate limit exceeded"}
    response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
    mock.get.return_value = response
    return mock


@pytest.fixture
def error_rate_limit_pr_result(
    github_service: "GitHubService",
    default_submission_params: dict,
    mock_rate_limit_error_responses: MagicMock,
) -> "PRResult":
    """PRResult from rate limit error."""
    from unittest.mock import patch

    with patch(
        "open_telco.cli.screens.submit.github_service.requests.get",
        return_value=mock_rate_limit_error_responses.get.return_value,
    ):
        return github_service.create_submission_pr(**default_submission_params)
