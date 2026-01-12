"""Stress testing module for model validation."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.model import get_model

_project_root = Path(__file__).parent.parent.parent.parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

DEFAULT_TEST_TIMEOUT = 30


@dataclass
class StressTestResult:
    """Result of stress testing."""

    passed: bool
    error: str | None = None
    tests_completed: int = 0


STRESS_TEST_PROMPTS = (
    {"name": "nested_braces", "prompt": "Calculate {x: x+1} for x=5. Put your final answer in \\boxed{}.", "timeout": DEFAULT_TEST_TIMEOUT},
    {"name": "multiple_boxed", "prompt": "Solve step by step: First \\boxed{step1}, then \\boxed{step2}, final answer \\boxed{result}. What is 2+3?", "timeout": DEFAULT_TEST_TIMEOUT},
    {"name": "long_context", "prompt": f"Context: {'The system processes data. ' * 50}\n\nBased on the above context, what is 7+8? Answer in \\boxed{{}} format.", "timeout": DEFAULT_TEST_TIMEOUT},
    {"name": "simple_math", "prompt": "What is 10 divided by 2? Put your answer in \\boxed{}.", "timeout": DEFAULT_TEST_TIMEOUT},
)


async def _send_prompt(model_name: str, prompt: str) -> str:
    model = get_model(model_name)
    response = await model.generate(prompt)
    return response.completion


async def _run_single_test(model_name: str, test: dict) -> str | None:
    try:
        await asyncio.wait_for(
            _send_prompt(model_name, test["prompt"]),
            timeout=test["timeout"],
        )
        return None
    except asyncio.TimeoutError:
        return f"'{test['name']}' timed out after {test['timeout']}s"
    except Exception as e:
        return f"'{test['name']}' failed: {e}"


async def run_stress_tests(model_name: str) -> StressTestResult:
    """Run all stress test prompts with individual timeouts."""
    for i, test in enumerate(STRESS_TEST_PROMPTS):
        error = await _run_single_test(model_name, test)
        if error:
            return StressTestResult(passed=False, error=error, tests_completed=i)

    return StressTestResult(passed=True, tests_completed=len(STRESS_TEST_PROMPTS))


def run_stress_tests_sync(model_name: str) -> StressTestResult:
    """Synchronous wrapper for run_stress_tests."""
    return asyncio.run(run_stress_tests(model_name))
