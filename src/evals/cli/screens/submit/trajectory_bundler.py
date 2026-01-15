"""Bundle trajectories for submission."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import evals_df

from evals.cli.types import Result


@dataclass
class SubmissionBundle:
    """Bundle of files for submission."""

    model_name: str
    provider: str
    parquet_content: bytes
    trajectory_files: dict[str, bytes]  # filename -> content


def _try_load_model_parquet(
    parquet_path: Path | None, model_str: str
) -> Result[bytes, str]:
    """Try to load parquet data for a specific model."""
    if parquet_path is None:
        return Result.err("No parquet path provided")
    if not parquet_path.exists():
        return Result.err(f"Parquet file not found: {parquet_path}")

    df_result = _try_read_parquet(parquet_path)
    if not df_result.success:
        return Result.err(df_result.error or "Failed to read parquet")

    model_df = df_result.value[df_result.value["model"] == model_str]
    if model_df.empty:
        return Result.err(f"Model '{model_str}' not found in parquet")

    buffer = io.BytesIO()
    model_df.to_parquet(buffer, index=False)
    return Result.ok(buffer.getvalue())


def _try_read_parquet(path: Path) -> Result[pd.DataFrame, str]:
    """Try to read a parquet file."""
    try:
        return Result.ok(pd.read_parquet(path))
    except Exception as e:
        return Result.err(f"Cannot read parquet {path}: {e}")


def _try_load_json(path: Path) -> Result[dict, str]:
    """Try to load JSON from a file."""
    try:
        return Result.ok(json.loads(path.read_text()))
    except json.JSONDecodeError as e:
        return Result.err(f"Invalid JSON in {path}: {e}")
    except OSError as e:
        return Result.err(f"Cannot read {path}: {e}")


def create_submission_bundle(
    model_name: str,
    provider: str,
    results_parquet_path: Path | None,
    log_dir: Path,
    raw_model: str | None = None,
) -> SubmissionBundle:
    """Create a submission bundle for a specific model.

    Can work with either parquet file or raw JSON logs.

    Args:
        model_name: Display name of the model (e.g., "gpt-4o")
        provider: Provider name (e.g., "Openai")
        results_parquet_path: Path to results.parquet (optional)
        log_dir: Path to log directory containing JSON trajectories
        raw_model: Raw model string for trajectory matching (e.g., "openai/gpt-4o")

    Returns:
        SubmissionBundle with all required files
    """
    # Expected model string format in parquet
    model_str = f"{model_name} ({provider})"

    # Try parquet first if available
    parquet_result = _try_load_model_parquet(results_parquet_path, model_str)

    # Generate parquet from JSON logs if needed
    if parquet_result.success:
        parquet_bytes = parquet_result.value
    else:
        parquet_bytes = _generate_parquet_from_logs(
            log_dir, model_name, provider, raw_model
        )

    # Find trajectory JSON files for this model
    trajectory_files = _find_trajectory_files(log_dir, model_name, provider, raw_model)

    return SubmissionBundle(
        model_name=model_name,
        provider=provider,
        parquet_content=parquet_bytes,
        trajectory_files=trajectory_files,
    )


def _generate_parquet_from_logs(
    log_dir: Path,
    model_name: str,
    provider: str,
    raw_model: str | None = None,
) -> bytes:
    """Generate parquet content from JSON trajectory logs.

    Args:
        log_dir: Directory with JSON logs
        model_name: Display model name
        provider: Provider name
        raw_model: Raw model string for matching

    Returns:
        Parquet file content as bytes
    """
    df = evals_df(str(log_dir), quiet=True)

    if df.empty:
        raise ValueError(f"Model not found: no eval logs in {log_dir}")

    # Filter to this model
    if raw_model:
        model_df = df[df["model"] == raw_model]
    else:
        model_df = df[df["model"].str.contains(model_name, case=False, na=False)]

    if model_df.empty:
        raise ValueError(f"Model '{model_name}' not found in logs")

    # Build leaderboard format row
    row: dict = {
        "model": f"{model_name} ({provider})",
        "teleqna": None,
        "telelogs": None,
        "telemath": None,
        "3gpp_tsg": None,
        "date": date.today().isoformat(),
    }

    task_column_map = {
        "teleqna": "teleqna",
        "telelogs": "telelogs",
        "telemath": "telemath",
        "three_gpp": "3gpp_tsg",
    }

    for _, eval_row in model_df.iterrows():
        task_name = str(eval_row.get("task_name", "")).lower()
        score = eval_row.get("score_headline_value")
        stderr = eval_row.get("score_headline_stderr")

        # Get n_samples from dataset_sample_ids (the actual samples evaluated)
        # dataset_sample_ids is stored as a JSON string like "[1, 2, 3, ...]"
        dataset_sample_ids = eval_row.get("dataset_sample_ids", "[]")
        if isinstance(dataset_sample_ids, str):
            try:
                sample_ids = json.loads(dataset_sample_ids)
                n_samples = len(sample_ids) if isinstance(sample_ids, list) else 0
            except (json.JSONDecodeError, TypeError):
                n_samples = 0
        elif hasattr(dataset_sample_ids, "__len__"):
            n_samples = len(dataset_sample_ids)
        else:
            n_samples = eval_row.get(
                "completed_samples", eval_row.get("total_samples", 0)
            )

        for task_key, column_name in task_column_map.items():
            if task_key in task_name:
                if pd.notna(score):
                    score_val = (
                        float(score) * 100 if float(score) <= 1.0 else float(score)
                    )
                    stderr_val = (
                        float(stderr) * 100
                        if pd.notna(stderr) and float(stderr) <= 1.0
                        else 0.0
                    )
                    row[column_name] = [
                        score_val,
                        stderr_val,
                        float(n_samples) if pd.notna(n_samples) else 0.0,
                    ]
                break

    result_df = pd.DataFrame([row])

    buffer = io.BytesIO()
    result_df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _find_trajectory_files(
    log_dir: Path,
    model_name: str,
    provider: str,
    raw_model: str | None = None,
) -> dict[str, bytes]:
    """Find trajectory JSON files matching the model.

    Args:
        log_dir: Directory containing JSON log files
        model_name: Model name to match
        provider: Provider name to match
        raw_model: Raw model string for exact matching

    Returns:
        Dict mapping filename to file content bytes
    """
    trajectory_files: dict[str, bytes] = {}

    # Look for JSON files in the log directory
    for json_file in log_dir.glob("*.json"):
        data_result = _try_load_json(json_file)
        if not data_result.success:
            continue

        if not _trajectory_matches_model(
            data_result.value, model_name, provider, raw_model
        ):
            continue

        content_result = _try_read_bytes(json_file)
        if content_result.success:
            trajectory_files[json_file.name] = content_result.value

    return trajectory_files


def _try_read_bytes(path: Path) -> Result[bytes, str]:
    """Try to read file as bytes."""
    try:
        return Result.ok(path.read_bytes())
    except OSError as e:
        return Result.err(f"Cannot read {path}: {e}")


def _trajectory_matches_model(
    data: dict,
    model_name: str,
    provider: str,
    raw_model: str | None = None,
) -> bool:
    """Check if trajectory belongs to the specified model.

    Inspect AI JSON log files have a structure like:
    {
        "eval": {
            "model": "openai/gpt-4o",
            ...
        },
        ...
    }

    Args:
        data: Parsed JSON data from trajectory file
        model_name: Model name to match (e.g., "gpt-4o")
        provider: Provider name to match (e.g., "Openai")
        raw_model: Raw model string for exact matching

    Returns:
        True if trajectory matches the model
    """
    # Try to extract model from various possible locations
    traj_model = ""

    # Inspect AI format: eval.model
    if "eval" in data and isinstance(data["eval"], dict):
        traj_model = data["eval"].get("model", "")

    # Alternative: top-level model field
    if not traj_model:
        traj_model = data.get("model", "")

    if not traj_model:
        return False

    # Exact match with raw_model if provided
    if raw_model and traj_model == raw_model:
        return True

    # Normalize for comparison
    traj_model_lower = traj_model.lower()
    model_name_lower = model_name.lower()
    provider_lower = provider.lower()

    # Check if model name appears in trajectory model string
    # Handles formats like: "openai/gpt-4o", "gpt-4o", etc.
    if model_name_lower in traj_model_lower:
        return True

    # Also check provider prefix format: "provider/model"
    if f"{provider_lower}/{model_name_lower}" in traj_model_lower:
        return True

    # Check if the provider appears in the model string
    # and the model name is a suffix
    parts = traj_model_lower.split("/")
    if len(parts) >= 2:
        # Format: provider/model or router/provider/model
        traj_provider = parts[0] if len(parts) == 2 else parts[1]
        traj_name = parts[-1]

        if provider_lower in traj_provider and model_name_lower in traj_name:
            return True

    return False
