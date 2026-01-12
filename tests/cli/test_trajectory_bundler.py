"""Tests for trajectory bundler functionality."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from open_telco.cli.screens.submit.trajectory_bundler import (
    SubmissionBundle,
    _find_trajectory_files,
    _trajectory_matches_model,
    create_submission_bundle,
)


class TestTrajectoryBundler:
    """Test trajectory bundler core functionality."""

    def test_create_bundle_filters_to_model(
        self, temp_results_parquet: Path, temp_trajectory_files: list[Path]
    ) -> None:
        """Bundle parquet should contain only the specified model's row."""
        bundle = create_submission_bundle(
            model_name="gpt-4o",
            provider="Openai",
            results_parquet_path=temp_results_parquet,
            log_dir=temp_trajectory_files[0].parent,
        )

        # Parse the parquet content
        df = pd.read_parquet(io.BytesIO(bundle.parquet_content))

        assert len(df) == 1
        assert df.iloc[0]["model"] == "gpt-4o (Openai)"

    def test_find_trajectories_by_eval_model(
        self, temp_trajectory_files: list[Path]
    ) -> None:
        """Should find trajectories matching eval.model field."""
        log_dir = temp_trajectory_files[0].parent

        files = _find_trajectory_files(log_dir, "gpt-4o", "Openai")

        # Should find the gpt-4o trajectories but not the claude one
        assert len(files) == 2
        filenames = list(files.keys())
        assert any("gpt4o" in name for name in filenames)
        assert not any("claude" in name for name in filenames)

    def test_find_trajectories_provider_model_format(self, tmp_path: Path) -> None:
        """Should match provider/model format like 'openai/gpt-4o'."""
        traj = tmp_path / "test_traj.json"
        traj.write_text(
            json.dumps(
                {
                    "eval": {
                        "model": "openai/gpt-4o",
                    },
                }
            )
        )

        result = _trajectory_matches_model(
            {"eval": {"model": "openai/gpt-4o"}},
            "gpt-4o",
            "Openai",
        )

        assert result is True

    def test_find_trajectories_router_provider_model(self, tmp_path: Path) -> None:
        """Should match router/provider/model format like 'openrouter/openai/gpt-4o'."""
        result = _trajectory_matches_model(
            {"eval": {"model": "openrouter/openai/gpt-4o"}},
            "gpt-4o",
            "Openai",
        )

        assert result is True

    def test_no_trajectories_found_returns_empty(self, tmp_path: Path) -> None:
        """Should return empty dict when no matching trajectories found."""
        # Create a trajectory for a different model
        traj = tmp_path / "other_model.json"
        traj.write_text(
            json.dumps(
                {
                    "eval": {
                        "model": "anthropic/claude-3",
                    },
                }
            )
        )

        files = _find_trajectory_files(tmp_path, "gpt-4o", "Openai")

        assert files == {}

    def test_invalid_json_skipped(self, tmp_path: Path) -> None:
        """Malformed JSON files should be skipped without error."""
        # Create invalid JSON file
        invalid = tmp_path / "invalid.json"
        invalid.write_text("{ this is not valid json }")

        # Create valid JSON file
        valid = tmp_path / "valid.json"
        valid.write_text(
            json.dumps(
                {
                    "eval": {
                        "model": "openai/gpt-4o",
                    },
                }
            )
        )

        files = _find_trajectory_files(tmp_path, "gpt-4o", "Openai")

        # Should find the valid file and skip the invalid one
        assert len(files) == 1
        assert "valid.json" in files

    def test_model_not_in_parquet_raises(
        self, temp_results_parquet: Path, tmp_path: Path
    ) -> None:
        """Should raise ValueError when model not found in parquet."""
        with pytest.raises(ValueError, match="not found"):
            create_submission_bundle(
                model_name="nonexistent-model",
                provider="Unknown",
                results_parquet_path=temp_results_parquet,
                log_dir=tmp_path,
            )


class TestTrajectoryMatching:
    """Test _trajectory_matches_model function."""

    def test_matches_model_name_substring(self) -> None:
        """Should match when model name is substring of trajectory model."""
        result = _trajectory_matches_model(
            {"eval": {"model": "gpt-4o"}},
            "gpt-4o",
            "Openai",
        )
        assert result is True

    def test_matches_case_insensitive(self) -> None:
        """Matching should be case insensitive."""
        result = _trajectory_matches_model(
            {"eval": {"model": "OPENAI/GPT-4O"}},
            "gpt-4o",
            "Openai",
        )
        assert result is True

    def test_matches_top_level_model_field(self) -> None:
        """Should match top-level model field as fallback."""
        result = _trajectory_matches_model(
            {"model": "openai/gpt-4o"},
            "gpt-4o",
            "Openai",
        )
        assert result is True

    def test_no_match_different_model(self) -> None:
        """Should not match a different model."""
        result = _trajectory_matches_model(
            {"eval": {"model": "anthropic/claude-3"}},
            "gpt-4o",
            "Openai",
        )
        assert result is False

    def test_no_match_empty_model_field(self) -> None:
        """Should not match when model field is empty."""
        result = _trajectory_matches_model(
            {"eval": {"model": ""}},
            "gpt-4o",
            "Openai",
        )
        assert result is False

    def test_no_match_missing_model_field(self) -> None:
        """Should not match when model field is missing."""
        result = _trajectory_matches_model(
            {"eval": {}},
            "gpt-4o",
            "Openai",
        )
        assert result is False


class TestSampleCountValidation:
    """Test sample count validation for --limit flag scenarios."""

    def test_trajectory_with_full_sample_ids(
        self, temp_trajectory_files: list[Path]
    ) -> None:
        """Full sample set should be detected correctly."""
        # Read the teleqna trajectory (should have 1000 samples)
        traj_path = [f for f in temp_trajectory_files if "teleqna_gpt4o" in f.name][0]
        with open(traj_path) as f:
            data = json.load(f)

        sample_ids = data["eval"]["dataset"]["sample_ids"]
        assert len(sample_ids) == 1000

    def test_trajectory_with_limited_samples(
        self, temp_trajectory_with_limit: Path
    ) -> None:
        """Limited samples should be detected (--limit flag was used)."""
        with open(temp_trajectory_with_limit) as f:
            data = json.load(f)

        sample_ids = data["eval"]["dataset"]["sample_ids"]
        # Only 10 samples instead of full set
        assert len(sample_ids) == 10

    def test_extract_sample_ids_from_trajectory(
        self, temp_trajectory_files: list[Path]
    ) -> None:
        """Should parse eval.dataset.sample_ids field."""
        traj_path = temp_trajectory_files[0]
        with open(traj_path) as f:
            data = json.load(f)

        # Verify structure
        assert "eval" in data
        assert "dataset" in data["eval"]
        assert "sample_ids" in data["eval"]["dataset"]

        sample_ids = data["eval"]["dataset"]["sample_ids"]
        assert isinstance(sample_ids, list)
        assert len(sample_ids) > 0

    def test_bundle_trajectory_files_content(
        self, temp_results_parquet: Path, temp_trajectory_files: list[Path]
    ) -> None:
        """Bundle should include trajectory file contents."""
        bundle = create_submission_bundle(
            model_name="gpt-4o",
            provider="Openai",
            results_parquet_path=temp_results_parquet,
            log_dir=temp_trajectory_files[0].parent,
        )

        # Check trajectory files are included
        assert len(bundle.trajectory_files) > 0

        # Verify content can be parsed as JSON
        for filename, content in bundle.trajectory_files.items():
            data = json.loads(content.decode())
            assert "eval" in data


class TestSubmissionBundle:
    """Test SubmissionBundle dataclass."""

    def test_bundle_has_all_required_fields(
        self, sample_submission_bundle: SubmissionBundle
    ) -> None:
        """Bundle should have all required fields."""
        assert sample_submission_bundle.model_name == "gpt-4o"
        assert sample_submission_bundle.provider == "Openai"
        assert isinstance(sample_submission_bundle.parquet_content, bytes)
        assert isinstance(sample_submission_bundle.trajectory_files, dict)

    def test_bundle_parquet_is_valid(
        self, sample_submission_bundle: SubmissionBundle
    ) -> None:
        """Bundle parquet content should be valid parquet."""
        df = pd.read_parquet(io.BytesIO(sample_submission_bundle.parquet_content))
        assert not df.empty
        assert "model" in df.columns

    def test_bundle_trajectories_are_valid_json(
        self, sample_submission_bundle: SubmissionBundle
    ) -> None:
        """Bundle trajectory files should be valid JSON."""
        for filename, content in sample_submission_bundle.trajectory_files.items():
            data = json.loads(content.decode())
            assert isinstance(data, dict)
