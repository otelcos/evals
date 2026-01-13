"""HuggingFace dataset API client for leaderboard data."""

from __future__ import annotations

import re
from typing import Any

import requests

from open_telco.cli.services.tci_calculator import LeaderboardEntry

HUGGINGFACE_API_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=GSMA/leaderboard&config=default&split=train&offset=0&length=100"
)


class HuggingFaceError(Exception):
    """Error fetching from HuggingFace API."""

    pass


def _fetch_with_timeout(url: str, timeout: int) -> requests.Response | None:
    """Fetch URL with timeout. Returns None on failure, raises HuggingFaceError."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.Timeout:
        raise HuggingFaceError("request-timed-out. try-again.")
    except requests.RequestException as e:
        raise HuggingFaceError(f"failed-to-fetch-leaderboard: {e}")


def _try_parse_json(response: requests.Response) -> dict | None:
    """Try to parse JSON from response. Returns None on failure."""
    try:
        return response.json()
    except (ValueError, requests.JSONDecodeError):
        return None


def fetch_leaderboard(timeout: int = 30) -> list[dict[str, Any]]:
    """Fetch leaderboard data from HuggingFace.

    Args:
        timeout: Request timeout in seconds

    Returns:
        List of raw row dictionaries from HuggingFace

    Raises:
        HuggingFaceError: On network errors, timeout, or invalid response
    """
    response = _fetch_with_timeout(HUGGINGFACE_API_URL, timeout)
    if response is None:
        raise HuggingFaceError("failed-to-fetch-leaderboard")

    data = _try_parse_json(response)
    if data is None:
        raise HuggingFaceError("invalid-response-format")

    if "rows" not in data:
        raise HuggingFaceError("invalid-huggingface-response-format")

    return [item["row"] for item in data["rows"]]


def parse_model_provider(combined: str) -> tuple[str, str]:
    """Parse 'model (Provider)' format into (model, provider) tuple.

    Args:
        combined: Combined model string like "gpt-4o (Openai)"

    Returns:
        Tuple of (model_name, provider)
    """
    match = re.match(r"^(.+?)\s*\(([^)]+)\)$", combined)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return combined, "Unknown"


def _extract_score(value: list[float] | None) -> tuple[float | None, float | None]:
    """Extract score and stderr from HuggingFace format [score, stderr, n_samples].

    Args:
        value: List of [score, stderr, n_samples] or None

    Returns:
        Tuple of (score, stderr)
    """
    if value is None or not isinstance(value, list) or len(value) < 2:
        return None, None
    return value[0], value[1]


def transform_hf_row(row: dict[str, Any]) -> LeaderboardEntry:
    """Transform a HuggingFace row to LeaderboardEntry.

    Args:
        row: Raw row from HuggingFace API

    Returns:
        LeaderboardEntry with scores populated
    """
    model_str = row.get("model", "Unknown")
    model, provider = parse_model_provider(model_str)

    teleqna, teleqna_stderr = _extract_score(row.get("teleqna"))
    telelogs, telelogs_stderr = _extract_score(row.get("telelogs"))
    telemath, telemath_stderr = _extract_score(row.get("telemath"))
    tsg, tsg_stderr = _extract_score(row.get("3gpp_tsg"))
    tci, _tci_stderr = _extract_score(row.get("tci"))

    return LeaderboardEntry(
        model=model,
        provider=provider,
        teleqna=teleqna,
        teleqna_stderr=teleqna_stderr,
        telelogs=telelogs,
        telelogs_stderr=telelogs_stderr,
        telemath=telemath,
        telemath_stderr=telemath_stderr,
        tsg=tsg,
        tsg_stderr=tsg_stderr,
        tci=tci,
    )


def fetch_and_transform_leaderboard(timeout: int = 30) -> list[LeaderboardEntry]:
    """Fetch leaderboard and transform to LeaderboardEntry list.

    Args:
        timeout: Request timeout in seconds

    Returns:
        List of LeaderboardEntry objects

    Raises:
        HuggingFaceError: On fetch or parse errors
    """
    rows = fetch_leaderboard(timeout)
    return [transform_hf_row(row) for row in rows]
