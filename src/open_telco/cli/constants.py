"""Centralized constants for Open Telco CLI.

All magic numbers, repeated dictionaries, and configuration values
are defined here. Import from this module instead of defining locally.
"""
from __future__ import annotations


class Colors:
    """GSMA brand colors used throughout the CLI."""

    RED = "#a61d2d"
    BACKGROUND = "#0d1117"
    BG_PRIMARY = "#0d1117"
    TEXT_MUTED = "#8b949e"
    TEXT_PRIMARY = "#f0f6fc"
    TEXT_DISABLED = "#484f58"
    BORDER = "#30363d"
    HOVER = "#21262d"
    SUCCESS = "#3fb950"
    ERROR = "#f85149"
    WARNING = "#f0883e"
    LINK = "#58a6ff"


PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "openai": "Openai",
    "anthropic": "Anthropic",
    "google": "Google",
    "mistralai": "Mistral",
    "deepseek": "Deepseek",
    "meta-llama": "Meta",
    "cohere": "Cohere",
    "together": "Together",
    "openrouter": "Openrouter",
    "groq": "Groq",
    "fireworks": "Fireworks",
    "allenai": "Allenai",
    "xai": "xAI",
    "perplexity": "Perplexity",
}


class Timeouts:
    """Timeout values in seconds."""

    MINI_TEST = 300
    FIND_K = 600
    FULL_EVAL = 3600
    NETWORK_REQUEST = 30


class Ports:
    """Network port configuration."""

    INSPECT_VIEWER = 7575


class Animation:
    """Animation timing configuration."""

    INTERVAL_SECONDS = 0.6
    DOT_CYCLE_LENGTH = 3
    PROGRESS_CYCLE_LENGTH = 5


ALL_TASKS: tuple[str, ...] = (
    "telelogs/telelogs.py",
    "telemath/telemath.py",
    "teleqna/teleqna.py",
    "three_gpp/three_gpp.py",
)

TASK_TO_COLUMN: dict[str, str] = {
    "teleqna": "teleqna",
    "telelogs": "telelogs",
    "telemath": "telemath",
    "three_gpp": "3gpp_tsg",
}

TASK_DISPLAY_NAMES: dict[str, str] = {
    "telelogs/telelogs.py": "telelogs",
    "telemath/telemath.py": "telemath",
    "teleqna/teleqna.py": "teleqna",
    "three_gpp/three_gpp.py": "3gpp_tsg",
}

MIN_SCORE_ARRAY_LEN = 2

TOP_N_DISPLAY = 10
