"""GitHub API service for PR creation."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class PRResult:
    """Result of PR creation."""

    success: bool
    pr_url: str | None = None
    error: str | None = None


class GitHubService:
    """Service for interacting with GitHub API."""

    REPO_OWNER = "gsma-research"
    REPO_NAME = "ot_leaderboard"
    API_BASE = "https://api.github.com"

    def __init__(self, token: str) -> None:
        self.token = token
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._user: str | None = None
        self._use_fork: bool = False

    def _get_authenticated_user(self) -> str:
        """Get the authenticated user's login."""
        if self._user:
            return self._user

        response = requests.get(
            f"{self.API_BASE}/user",
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        self._user = response.json()["login"]
        return self._user

    def _check_direct_access(self) -> bool:
        """Check if user has direct write access to the repo.

        Returns:
            True if user can push directly, False if fork is needed
        """
        user = self._get_authenticated_user()

        # Check user's permission level on the repo
        response = requests.get(
            f"{self.API_BASE}/repos/{self.REPO_OWNER}/{self.REPO_NAME}/collaborators/{user}/permission",
            headers=self.headers,
            timeout=30,
        )

        if response.status_code == 200:
            permission = response.json().get("permission", "")
            # write, admin, or maintain permissions allow direct push
            return permission in ("write", "admin", "maintain")

        return False

    def _ensure_fork(self) -> dict[str, Any]:
        """Ensure user has a fork of the repository.

        Returns:
            Fork repository info
        """
        user = self._get_authenticated_user()

        # Check if fork already exists
        response = requests.get(
            f"{self.API_BASE}/repos/{user}/{self.REPO_NAME}",
            headers=self.headers,
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()

        # Create fork
        response = requests.post(
            f"{self.API_BASE}/repos/{self.REPO_OWNER}/{self.REPO_NAME}/forks",
            headers=self.headers,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def _get_default_branch_sha(self, owner: str) -> str:
        """Get SHA of the default branch head.

        Args:
            owner: Repository owner

        Returns:
            SHA of the default branch head
        """
        response = requests.get(
            f"{self.API_BASE}/repos/{owner}/{self.REPO_NAME}/git/refs/heads/main",
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["object"]["sha"]

    def _create_branch(self, owner: str, branch_name: str, base_sha: str) -> None:
        """Create a new branch.

        Args:
            owner: Repository owner
            branch_name: Name for the new branch
            base_sha: SHA to base the branch on
        """
        # Check if branch exists
        response = requests.get(
            f"{self.API_BASE}/repos/{owner}/{self.REPO_NAME}/git/refs/heads/{branch_name}",
            headers=self.headers,
            timeout=30,
        )

        if response.status_code == 200:
            # Branch exists, update it
            requests.patch(
                f"{self.API_BASE}/repos/{owner}/{self.REPO_NAME}/git/refs/heads/{branch_name}",
                headers=self.headers,
                json={"sha": base_sha, "force": True},
                timeout=30,
            )
            return

        # Create new branch
        response = requests.post(
            f"{self.API_BASE}/repos/{owner}/{self.REPO_NAME}/git/refs",
            headers=self.headers,
            json={
                "ref": f"refs/heads/{branch_name}",
                "sha": base_sha,
            },
            timeout=30,
        )
        response.raise_for_status()

    def _create_or_update_file(
        self,
        owner: str,
        file_path: str,
        content: bytes,
        branch: str,
        message: str,
    ) -> None:
        """Create or update a file in the repository.

        Args:
            owner: Repository owner
            file_path: Path to the file in the repo
            content: File content as bytes
            branch: Branch to commit to
            message: Commit message
        """
        # Check if file exists to get SHA
        response = requests.get(
            f"{self.API_BASE}/repos/{owner}/{self.REPO_NAME}/contents/{file_path}",
            headers=self.headers,
            params={"ref": branch},
            timeout=30,
        )

        data = {
            "message": message,
            "content": base64.b64encode(content).decode("utf-8"),
            "branch": branch,
        }

        if response.status_code == 200:
            # File exists, include SHA to update
            data["sha"] = response.json()["sha"]

        response = requests.put(
            f"{self.API_BASE}/repos/{owner}/{self.REPO_NAME}/contents/{file_path}",
            headers=self.headers,
            json=data,
            timeout=30,
        )
        response.raise_for_status()

    def _create_pull_request(
        self,
        head_owner: str,
        branch_name: str,
        model_name: str,
        provider: str,
    ) -> dict[str, Any]:
        """Create a pull request.

        Args:
            head_owner: Owner of the branch (user or upstream)
            branch_name: Branch with changes
            model_name: Name of the model being submitted
            provider: Model provider

        Returns:
            PR info dictionary
        """
        # Determine head reference format
        if self._use_fork:
            head_ref = f"{head_owner}:{branch_name}"
        else:
            head_ref = branch_name

        # Check for existing PR
        response = requests.get(
            f"{self.API_BASE}/repos/{self.REPO_OWNER}/{self.REPO_NAME}/pulls",
            headers=self.headers,
            params={
                "head": f"{head_owner}:{branch_name}",
                "state": "open",
            },
            timeout=30,
        )
        response.raise_for_status()
        existing = response.json()

        if existing:
            # Return existing PR
            return existing[0]

        # Create new PR
        response = requests.post(
            f"{self.API_BASE}/repos/{self.REPO_OWNER}/{self.REPO_NAME}/pulls",
            headers=self.headers,
            json={
                "title": f"[Submission] {model_name} ({provider})",
                "head": head_ref,
                "base": "main",
                "body": self._generate_pr_body(model_name, provider),
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _generate_pr_body(self, model_name: str, provider: str) -> str:
        """Generate PR body with submission details."""
        return f"""## Leaderboard Submission

**Model:** {model_name}
**Provider:** {provider}

### Submission Contents
- Model card (`model_cards/`) - parquet with scores
- Evaluation trajectories (`trajectories/`) - JSON logs

### Validation
Automated validation will check:
- [ ] Parquet schema (required columns)
- [ ] JSON trajectory format
- [ ] Model/provider format
- [ ] Inspect evaluation signature
- [ ] No errors in trajectory logs

---
*Submitted via Open Telco CLI*
"""

    def _sanitize_branch_name(self, name: str) -> str:
        """Sanitize string for use in branch name."""
        # Replace invalid characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Trim underscores from ends
        return sanitized.strip("_")

    def create_submission_pr(
        self,
        model_name: str,
        provider: str,
        parquet_content: bytes,
        trajectory_files: dict[str, bytes],
    ) -> PRResult:
        """Create a PR with model submission.

        Tries direct push first, falls back to fork if no write access.

        Args:
            model_name: Display name of the model
            provider: Provider name
            parquet_content: Parquet file bytes
            trajectory_files: Dict mapping filename to content bytes

        Returns:
            PRResult with success status and PR URL or error
        """
        try:
            # Determine if we can push directly or need to fork
            self._use_fork = not self._check_direct_access()

            if self._use_fork:
                # Fork workflow
                fork = self._ensure_fork()
                repo_owner = fork["owner"]["login"]
            else:
                # Direct access workflow
                repo_owner = self.REPO_OWNER

            # Get the default branch SHA
            base_sha = self._get_default_branch_sha(repo_owner)

            # Create a new branch
            safe_provider = self._sanitize_branch_name(provider.lower())
            safe_model = self._sanitize_branch_name(model_name)
            branch_name = f"submit/{safe_provider}_{safe_model}"

            self._create_branch(repo_owner, branch_name, base_sha)

            # Create file paths following flat structure:
            # model_cards/provider_model.parquet
            # trajectories/provider_model/*.json
            file_base = f"{safe_provider}_{safe_model}"

            # Add parquet file to model_cards/
            self._create_or_update_file(
                repo_owner,
                f"model_cards/{file_base}.parquet",
                parquet_content,
                branch_name,
                f"Add model card for {model_name}",
            )

            # Add trajectory JSON files to trajectories/provider_model/
            for filename, content in trajectory_files.items():
                self._create_or_update_file(
                    repo_owner,
                    f"trajectories/{file_base}/{filename}",
                    content,
                    branch_name,
                    f"Add trajectory: {filename}",
                )

            # Create pull request
            pr = self._create_pull_request(
                repo_owner,
                branch_name,
                model_name,
                provider,
            )

            return PRResult(success=True, pr_url=pr["html_url"])

        except requests.HTTPError as e:
            error_msg = str(e)
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("message", str(e))
                except ValueError:
                    error_msg = e.response.text or str(e)
            return PRResult(success=False, error=error_msg)
        except Exception as e:
            return PRResult(success=False, error=str(e))
