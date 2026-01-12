"""Model string parsing utilities.

Centralizes the logic for parsing model identifiers into
provider/model components and display formats.
"""
from __future__ import annotations

from open_telco.cli.constants import PROVIDER_DISPLAY_NAMES
from open_telco.cli.types import ModelInfo


def parse_model_string(model_str: str) -> ModelInfo:
    """Parse model string into structured ModelInfo.

    Handles formats:
    - openrouter/openai/gpt-4o -> (openai, gpt-4o, "gpt-4o (Openai)")
    - openai/gpt-4o -> (openai, gpt-4o, "gpt-4o (Openai)")
    - gpt-4 -> (unknown, gpt-4, "gpt-4 (Unknown)")

    Args:
        model_str: Raw model identifier string

    Returns:
        ModelInfo with parsed provider, model_name, and display_name
    """
    parts = model_str.split("/")

    if len(parts) >= 3:
        provider = parts[1]
        model_name = "/".join(parts[2:])
        provider_display = PROVIDER_DISPLAY_NAMES.get(provider.lower(), provider.title())
        display_name = f"{model_name} ({provider_display})"
        return ModelInfo(provider=provider, model_name=model_name, display_name=display_name)

    if len(parts) == 2:
        provider = parts[0]
        model_name = parts[1]
        provider_display = PROVIDER_DISPLAY_NAMES.get(provider.lower(), provider.title())
        display_name = f"{model_name} ({provider_display})"
        return ModelInfo(provider=provider, model_name=model_name, display_name=display_name)

    provider = "unknown"
    model_name = model_str
    display_name = f"{model_name} (Unknown)"
    return ModelInfo(provider=provider, model_name=model_name, display_name=display_name)


def get_provider_display(provider_key: str) -> str:
    """Get display name for a provider key.

    Args:
        provider_key: Lowercase provider identifier

    Returns:
        Human-readable provider name
    """
    return PROVIDER_DISPLAY_NAMES.get(provider_key.lower(), provider_key.title())
