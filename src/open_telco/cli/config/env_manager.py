"""Environment configuration manager for Open Telco CLI."""

from pathlib import Path

from dotenv import dotenv_values, set_key

from open_telco.cli.types import Result

# Provider configuration mapping
PROVIDERS: dict[str, dict[str, str]] = {
    "OpenAI": {
        "env_key": "OPENAI_API_KEY",
        "prefix": "openai",
        "example_model": "openai/gpt-4o",
    },
    "Anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "prefix": "anthropic",
        "example_model": "anthropic/claude-3-5-sonnet-20240620",
    },
    "Google": {
        "env_key": "GOOGLE_API_KEY",
        "prefix": "google",
        "example_model": "google/gemini-1.5-pro",
    },
    "Mistral": {
        "env_key": "MISTRAL_API_KEY",
        "prefix": "mistral",
        "example_model": "mistral/mistral-large-latest",
    },
    "OpenRouter": {
        "env_key": "OPENROUTER_API_KEY",
        "prefix": "openrouter",
        "example_model": "openrouter/openai/gpt-4o",
    },
    "Groq": {
        "env_key": "GROQ_API_KEY",
        "prefix": "groq",
        "example_model": "groq/llama-3.1-70b-versatile",
    },
    "Grok (xAI)": {
        "env_key": "XAI_API_KEY",
        "prefix": "xai",
        "example_model": "xai/grok-2",
    },
    "DeepSeek": {
        "env_key": "DEEPSEEK_API_KEY",
        "prefix": "deepseek",
        "example_model": "deepseek/deepseek-chat",
    },
    "Perplexity": {
        "env_key": "PERPLEXITY_API_KEY",
        "prefix": "perplexity",
        "example_model": "perplexity/llama-3.1-sonar-large-128k-online",
    },
}


class EnvManager:
    """Manages environment variables in .env file."""

    def __init__(self, env_path: Path | None = None) -> None:
        """Initialize with path to .env file."""
        if env_path is None:
            # Default to project root .env
            self.env_path = Path.cwd() / ".env"
        else:
            self.env_path = env_path

    def get(self, key: str) -> str | None:
        """Get a value from .env file."""
        values = dotenv_values(self.env_path)
        return values.get(key)

    def has_key(self, key: str) -> bool:
        """Check if a key exists and has a non-empty value in .env file."""
        values = dotenv_values(self.env_path)
        value = values.get(key)
        return value is not None and len(value) > 0

    def set(self, key: str, value: str) -> Result[bool, str]:
        """Set a value in .env file."""
        try:
            if not self.env_path.exists():
                self.env_path.touch()

            success, _, _ = set_key(str(self.env_path), key, value)
            if success:
                return Result.ok(True)
            return Result.err("Failed to write to .env file")
        except OSError as e:
            return Result.err(f"File error: {e}")

    def get_all(self) -> dict[str, str | None]:
        """Get all values from .env file."""
        return dotenv_values(self.env_path)
