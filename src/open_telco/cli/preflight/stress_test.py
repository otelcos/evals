"""Stress testing module for model validation."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.model import get_model

# Load .env from project root (6 levels up from this file)
# Path: cli/preflight/stress_test.py -> open_telco -> src -> project_root
_project_root = Path(__file__).parent.parent.parent.parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


@dataclass
class StressTestResult:
    """Result of stress testing."""

    passed: bool
    error: str | None = None
    tests_completed: int = 0


# Stress test prompts with individual timeouts
STRESS_TEST_PROMPTS = [
    {
        "name": "nested_braces",
        "prompt": "Calculate {x: x+1} for x=5. Put your final answer in \\boxed{}.",
        "timeout": 30,
    },
    {
        "name": "multiple_boxed",
        "prompt": "Solve step by step: First \\boxed{step1}, then \\boxed{step2}, final answer \\boxed{result}. What is 2+3?",
        "timeout": 30,
    },
    {
        "name": "long_context",
        "prompt": (
            "Context: " + ("The system processes data. " * 50) + "\n\n"
            "Based on the above context, what is 7+8? Answer in \\boxed{} format."
        ),
        "timeout": 30,
    },
    {
        "name": "simple_math",
        "prompt": "What is 10 divided by 2? Put your answer in \\boxed{}.",
        "timeout": 30,
    },
]


async def _send_prompt(model_name: str, prompt: str) -> str:
    """Send a single prompt to the model and get response."""
    model = get_model(model_name)
    response = await model.generate(prompt)
    return response.completion


async def run_stress_tests(model_name: str) -> StressTestResult:
    """Run all stress test prompts with individual timeouts.

    Args:
        model_name: The model identifier (e.g., "openrouter/openai/gpt-4o").

    Returns:
        StressTestResult with pass/fail status and any error message.
    """
    tests_completed = 0

    for test in STRESS_TEST_PROMPTS:
        try:
            await asyncio.wait_for(
                _send_prompt(model_name, test["prompt"]),
                timeout=test["timeout"],
            )
            tests_completed += 1

        except asyncio.TimeoutError:
            return StressTestResult(
                passed=False,
                error=f"'{test['name']}' timed out after {test['timeout']}s",
                tests_completed=tests_completed,
            )

        except Exception as e:
            return StressTestResult(
                passed=False,
                error=f"'{test['name']}' failed: {e}",
                tests_completed=tests_completed,
            )

    return StressTestResult(
        passed=True,
        tests_completed=tests_completed,
    )


def run_stress_tests_sync(model_name: str) -> StressTestResult:
    """Synchronous wrapper for run_stress_tests.

    For use in threaded workers where we create our own event loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_stress_tests(model_name))
    finally:
        loop.close()
