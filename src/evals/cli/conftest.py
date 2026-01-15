"""Shared fixtures for CLI tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest

if TYPE_CHECKING:
    from evals.cli.screens.submit.github_service import PRResult
    from evals.cli.screens.submit.trajectory_bundler import SubmissionBundle


@pytest.fixture
def mock_github_token(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set GITHUB_TOKEN for EnvManager and environment."""
    token = "ghp_test_token_12345"
    # Patch EnvManager.get to return the token for GITHUB_TOKEN
    from evals.cli.config import EnvManager

    original_get = EnvManager.get

    def patched_get(self: EnvManager, key: str) -> str | None:
        if key == "GITHUB_TOKEN":
            return token
        return original_get(self, key)

    monkeypatch.setattr(EnvManager, "get", patched_get)
    # Also set env var for backwards compatibility
    monkeypatch.setenv("GITHUB_TOKEN", token)
    return token


@pytest.fixture
def temp_results_parquet(tmp_path: Path) -> Path:
    """Temporary results.parquet with test data."""
    parquet_path = tmp_path / "results.parquet"
    df = pd.DataFrame(
        [
            {
                "model": "gpt-4o (Openai)",
                "teleqna": [83.6, 1.17, 1000.0],
                "telelogs": [75.0, 4.35, 100.0],
                "telemath": [39.0, 4.9, 100.0],
                "3gpp_tsg": [54.0, 5.01, 100.0],
                "date": "2026-01-09",
            },
            {
                "model": "claude-3-opus (Anthropic)",
                "teleqna": [85.2, 1.05, 1000.0],
                "telelogs": [78.0, 4.1, 100.0],
                "telemath": [42.0, 4.5, 100.0],
                "3gpp_tsg": [56.0, 4.8, 100.0],
                "date": "2026-01-09",
            },
        ]
    )
    df.to_parquet(parquet_path, index=False)
    return parquet_path


@pytest.fixture
def temp_trajectory_files(tmp_path: Path) -> list[Path]:
    """Temporary trajectory JSON files matching Inspect AI format."""
    trajectory_files = []

    # Create trajectory for gpt-4o
    traj1 = tmp_path / "eval_2026-01-09_teleqna_gpt4o.json"
    traj1.write_text(
        json.dumps(
            {
                "eval": {
                    "model": "openai/gpt-4o",
                    "task": "teleqna",
                    "dataset": {
                        "sample_ids": list(range(1000)),
                    },
                },
                "results": {
                    "accuracy": 0.836,
                },
            }
        )
    )
    trajectory_files.append(traj1)

    # Create trajectory for telelogs
    traj2 = tmp_path / "eval_2026-01-09_telelogs_gpt4o.json"
    traj2.write_text(
        json.dumps(
            {
                "eval": {
                    "model": "openai/gpt-4o",
                    "task": "telelogs",
                    "dataset": {
                        "sample_ids": list(range(100)),
                    },
                },
                "results": {
                    "accuracy": 0.75,
                },
            }
        )
    )
    trajectory_files.append(traj2)

    # Create trajectory for claude (different model)
    traj3 = tmp_path / "eval_2026-01-09_teleqna_claude.json"
    traj3.write_text(
        json.dumps(
            {
                "eval": {
                    "model": "anthropic/claude-3-opus",
                    "task": "teleqna",
                    "dataset": {
                        "sample_ids": list(range(1000)),
                    },
                },
                "results": {
                    "accuracy": 0.852,
                },
            }
        )
    )
    trajectory_files.append(traj3)

    return trajectory_files


@pytest.fixture
def temp_trajectory_with_limit(tmp_path: Path) -> Path:
    """Trajectory JSON with limited samples (--limit flag was used)."""
    traj = tmp_path / "eval_limited_teleqna.json"
    traj.write_text(
        json.dumps(
            {
                "eval": {
                    "model": "openai/gpt-4o",
                    "task": "teleqna",
                    "dataset": {
                        "sample_ids": list(
                            range(10)
                        ),  # Only 10 samples instead of full set
                    },
                },
                "results": {
                    "accuracy": 0.80,
                },
            }
        )
    )
    return traj


@pytest.fixture
def sample_pr_result() -> "PRResult":
    """Sample successful PRResult."""
    from evals.cli.screens.submit.github_service import PRResult

    return PRResult(
        success=True,
        pr_url="https://github.com/otelcos/leaderboard/pull/123",
    )


@pytest.fixture
def sample_failed_pr_result() -> "PRResult":
    """Sample failed PRResult."""
    from evals.cli.screens.submit.github_service import PRResult

    return PRResult(
        success=False,
        error="Bad credentials",
    )


@pytest.fixture
def sample_submission_bundle() -> "SubmissionBundle":
    """Sample SubmissionBundle with test data."""
    from evals.cli.screens.submit.trajectory_bundler import SubmissionBundle

    # Create minimal parquet bytes
    df = pd.DataFrame(
        [
            {
                "model": "gpt-4o (Openai)",
                "teleqna": [83.6, 1.17, 1000.0],
                "telelogs": [75.0, 4.35, 100.0],
                "telemath": [39.0, 4.9, 100.0],
                "3gpp_tsg": [54.0, 5.01, 100.0],
                "date": "2026-01-09",
            }
        ]
    )
    import io

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    parquet_bytes = buffer.getvalue()

    # Create sample trajectory
    trajectory_content = json.dumps(
        {
            "eval": {
                "model": "openai/gpt-4o",
                "task": "teleqna",
            },
            "results": {
                "accuracy": 0.836,
            },
        }
    ).encode()

    return SubmissionBundle(
        model_name="gpt-4o",
        provider="Openai",
        parquet_content=parquet_bytes,
        trajectory_files={
            "eval_teleqna.json": trajectory_content,
        },
    )


@pytest.fixture
def mock_github_service() -> MagicMock:
    """Mocked GitHubService with configurable responses."""
    mock = MagicMock()

    # Default successful PR creation
    from evals.cli.screens.submit.github_service import PRResult

    mock.create_submission_pr.return_value = PRResult(
        success=True,
        pr_url="https://github.com/otelcos/leaderboard/pull/123",
    )

    return mock
