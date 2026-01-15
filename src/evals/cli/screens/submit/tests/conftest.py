"""Submit-specific test fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import requests

if TYPE_CHECKING:
    from evals.cli.screens.submit.github_service import GitHubService, PRResult


@pytest.fixture
def github_service(mock_github_token: str) -> "GitHubService":
    """Create a GitHubService instance with mocked token."""
    from evals.cli.screens.submit.github_service import GitHubService

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

    # 1. GET /user - authenticated user
    user_response = MagicMock()
    user_response.status_code = 200
    user_response.json.return_value = {"login": "testuser"}

    # 2. GET /repos/.../collaborators/{user}/permission - check write access
    permission_response = MagicMock()
    permission_response.status_code = 200
    permission_response.json.return_value = {"permission": "write"}

    # 3. GET /repos/.../git/refs/heads/main - get base SHA
    ref_response = MagicMock()
    ref_response.status_code = 200
    ref_response.json.return_value = {"object": {"sha": "abc123"}}

    # 4. GET /repos/.../git/refs/heads/{branch} - branch doesn't exist
    branch_check_response = MagicMock()
    branch_check_response.status_code = 404

    # 5. GET /repos/.../contents/{parquet} - file doesn't exist
    file_check_404 = MagicMock()
    file_check_404.status_code = 404

    # 6. GET /repos/.../contents/{trajectory} - file doesn't exist
    # (reuse file_check_404)

    # 7. GET /repos/.../pulls - no existing PRs
    no_prs_response = MagicMock()
    no_prs_response.status_code = 200
    no_prs_response.json.return_value = []

    # POST /repos/.../git/refs - create branch
    branch_response = MagicMock()
    branch_response.status_code = 201
    branch_response.json.return_value = {"ref": "refs/heads/submission/gpt-4o"}

    # POST /repos/.../pulls - create PR
    pr_response = MagicMock()
    pr_response.status_code = 201
    pr_response.json.return_value = {
        "html_url": "https://github.com/otelcos/leaderboard/pull/123"
    }

    # PUT /repos/.../contents/{file} - create files
    file_response = MagicMock()
    file_response.status_code = 201

    mock.get.side_effect = [
        user_response,  # 1. /user
        permission_response,  # 2. /collaborators/.../permission
        ref_response,  # 3. /git/refs/heads/main
        branch_check_response,  # 4. check branch exists
        file_check_404,  # 5. check parquet exists
        file_check_404,  # 6. check trajectory exists
        no_prs_response,  # 7. check existing PRs
    ]
    mock.post.side_effect = [branch_response, pr_response]
    mock.put.side_effect = [file_response, file_response]

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
            "evals.cli.screens.submit.github_service.requests.get",
            side_effect=mock_direct_access_responses.get.side_effect,
        ),
        patch(
            "evals.cli.screens.submit.github_service.requests.post",
            side_effect=mock_direct_access_responses.post.side_effect,
        ),
        patch(
            "evals.cli.screens.submit.github_service.requests.put",
            side_effect=mock_direct_access_responses.put.side_effect,
        ),
    ):
        result = github_service.create_submission_pr(**default_submission_params)
        return result, github_service


@pytest.fixture
def mock_fork_fallback_responses() -> MagicMock:
    """Mock responses for fork fallback PR creation (no direct access, use fork)."""
    mock = MagicMock()

    # 1. GET /user - authenticated user
    user_response = MagicMock()
    user_response.status_code = 200
    user_response.json.return_value = {"login": "testuser"}

    # 2. GET /repos/.../collaborators/{user}/permission - no write access (read only)
    permission_response = MagicMock()
    permission_response.status_code = 200
    permission_response.json.return_value = {"permission": "read"}

    # 3. GET /repos/{user}/{repo} - check if fork exists (404 = doesn't exist)
    fork_check_404 = MagicMock()
    fork_check_404.status_code = 404

    # 4. GET /repos/{user}/{repo}/git/refs/heads/main - get base SHA from fork
    ref_response = MagicMock()
    ref_response.status_code = 200
    ref_response.json.return_value = {"object": {"sha": "abc123"}}

    # 5. GET /repos/{user}/{repo}/git/refs/heads/{branch} - branch doesn't exist
    branch_check_404 = MagicMock()
    branch_check_404.status_code = 404

    # 6-7. GET /repos/.../contents/{file} - files don't exist
    file_check_404 = MagicMock()
    file_check_404.status_code = 404

    # 8. GET /repos/.../pulls - no existing PRs
    no_prs_response = MagicMock()
    no_prs_response.status_code = 200
    no_prs_response.json.return_value = []

    # POST /repos/.../forks - create fork
    fork_response = MagicMock()
    fork_response.status_code = 202
    fork_response.json.return_value = {
        "full_name": "testuser/leaderboard",
        "default_branch": "main",
        "owner": {"login": "testuser"},
    }

    # POST /repos/{user}/{repo}/git/refs - create branch
    branch_response = MagicMock()
    branch_response.status_code = 201

    # POST /repos/.../pulls - create PR
    pr_response = MagicMock()
    pr_response.status_code = 201
    pr_response.json.return_value = {
        "html_url": "https://github.com/otelcos/leaderboard/pull/456"
    }

    # PUT /repos/{user}/{repo}/contents/{file} - create files
    file_response = MagicMock()
    file_response.status_code = 201

    mock.get.side_effect = [
        user_response,  # 1. /user
        permission_response,  # 2. /collaborators/.../permission (read only)
        fork_check_404,  # 3. check if fork exists
        ref_response,  # 4. get base SHA from fork
        branch_check_404,  # 5. check branch exists
        file_check_404,  # 6. check parquet exists
        file_check_404,  # 7. check trajectory exists
        no_prs_response,  # 8. check existing PRs
    ]
    mock.post.side_effect = [fork_response, branch_response, pr_response]
    mock.put.side_effect = [file_response, file_response]

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
            "evals.cli.screens.submit.github_service.requests.get",
            side_effect=mock_fork_fallback_responses.get.side_effect,
        ),
        patch(
            "evals.cli.screens.submit.github_service.requests.post",
            side_effect=mock_fork_fallback_responses.post.side_effect,
        ),
        patch(
            "evals.cli.screens.submit.github_service.requests.put",
            side_effect=mock_fork_fallback_responses.put.side_effect,
        ),
    ):
        result = github_service.create_submission_pr(**default_submission_params)
        return result, github_service


@pytest.fixture
def mock_existing_pr_responses() -> MagicMock:
    """Mock responses when PR already exists."""
    mock = MagicMock()

    # 1. GET /user - authenticated user
    user_response = MagicMock()
    user_response.status_code = 200
    user_response.json.return_value = {"login": "testuser"}

    # 2. GET /repos/.../collaborators/{user}/permission - write access
    permission_response = MagicMock()
    permission_response.status_code = 200
    permission_response.json.return_value = {"permission": "write"}

    # 3. GET /repos/.../git/refs/heads/main - get base SHA
    ref_response = MagicMock()
    ref_response.status_code = 200
    ref_response.json.return_value = {"object": {"sha": "abc123"}}

    # 4. GET /repos/.../git/refs/heads/{branch} - branch already exists
    branch_exists_response = MagicMock()
    branch_exists_response.status_code = 200
    branch_exists_response.json.return_value = {"object": {"sha": "def456"}}

    # 5-6. GET /repos/.../contents/{file} - files don't exist
    file_check_404 = MagicMock()
    file_check_404.status_code = 404

    # 7. GET /repos/.../pulls - existing PR found!
    existing_pr_response = MagicMock()
    existing_pr_response.status_code = 200
    existing_pr_response.json.return_value = [
        {"html_url": "https://github.com/otelcos/leaderboard/pull/99"}
    ]

    # PATCH /repos/.../git/refs/heads/{branch} - update existing branch
    patch_response = MagicMock()
    patch_response.status_code = 200

    # PUT /repos/.../contents/{file} - create files
    file_response = MagicMock()
    file_response.status_code = 201

    mock.get.side_effect = [
        user_response,  # 1. /user
        permission_response,  # 2. /collaborators/.../permission
        ref_response,  # 3. /git/refs/heads/main
        branch_exists_response,  # 4. branch already exists
        file_check_404,  # 5. check parquet exists
        file_check_404,  # 6. check trajectory exists
        existing_pr_response,  # 7. existing PR found
    ]
    mock.post.side_effect = []  # No POST needed (branch exists, PR exists)
    mock.patch.side_effect = [patch_response]
    mock.put.side_effect = [file_response, file_response]

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
            "evals.cli.screens.submit.github_service.requests.get",
            side_effect=mock_existing_pr_responses.get.side_effect,
        ),
        patch(
            "evals.cli.screens.submit.github_service.requests.post",
            side_effect=mock_existing_pr_responses.post.side_effect,
        ),
        patch(
            "evals.cli.screens.submit.github_service.requests.patch",
            side_effect=mock_existing_pr_responses.patch.side_effect,
        ),
        patch(
            "evals.cli.screens.submit.github_service.requests.put",
            side_effect=mock_existing_pr_responses.put.side_effect,
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
        "evals.cli.screens.submit.github_service.requests.get",
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
        "evals.cli.screens.submit.github_service.requests.get",
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
        "evals.cli.screens.submit.github_service.requests.get",
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
        "evals.cli.screens.submit.github_service.requests.get",
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
        "evals.cli.screens.submit.github_service.requests.get",
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
        "evals.cli.screens.submit.github_service.requests.get",
        return_value=mock_rate_limit_error_responses.get.return_value,
    ):
        return github_service.create_submission_pr(**default_submission_params)
