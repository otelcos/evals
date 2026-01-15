"""Edge case prompts for stress testing model responses."""

import re
from dataclasses import dataclass

BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


@dataclass
class EdgeCasePrompt:
    """A test prompt with expected behavior."""

    name: str
    prompt: str
    expected_behavior: str  # "boxed_format", "numeric", "pattern_match"
    should_have_content: bool = True


EDGE_CASE_PROMPTS = [
    EdgeCasePrompt(
        name="basic_boxed",
        prompt="What is 2+2? Answer in \\boxed{} format.",
        expected_behavior="boxed_format",
    ),
    EdgeCasePrompt(
        name="nested_braces",
        prompt="Calculate {x: x+1} for x=5. Put answer in \\boxed{}.",
        expected_behavior="boxed_format",
    ),
    EdgeCasePrompt(
        name="multiple_boxed",
        prompt="Step 1: \\boxed{5}, Final: \\boxed{10}. What's the final?",
        expected_behavior="boxed_format",
    ),
    EdgeCasePrompt(
        name="empty_response_trigger",
        prompt="Reply with only: \\boxed{}",
        expected_behavior="boxed_format",
        should_have_content=False,
    ),
    EdgeCasePrompt(
        name="numeric_extraction",
        prompt="The answer is forty-two (42). Extract the number in \\boxed{}.",
        expected_behavior="numeric",
    ),
]


def validate_boxed_response(response: str) -> tuple[bool, str]:
    r"""Validate response contains valid \boxed{} format.

    Returns:
        (True, extracted_content) if valid, (False, error_message) if invalid.
    """
    if not response:
        return False, "Empty response"

    matches = BOXED_PATTERN.findall(response)
    if not matches:
        return False, "No \\boxed{} found in response"

    return True, matches[-1].strip()


def _validate_boxed(response: str, edge_case: EdgeCasePrompt) -> tuple[bool, str]:
    valid, content = validate_boxed_response(response)
    if not valid:
        return False, content

    if edge_case.should_have_content and not content:
        return False, "Expected non-empty content in \\boxed{}"

    return True, f"Valid boxed format: {content}"


def _validate_numeric(response: str) -> tuple[bool, str]:
    valid, content = validate_boxed_response(response)
    if not valid:
        return False, content

    if not re.search(r"\d+", content):
        return False, f"Expected numeric content, got: {content}"

    return True, f"Valid numeric format: {content}"


def _validate_pattern(response: str) -> tuple[bool, str]:
    if not response.strip():
        return False, "Empty response"

    return True, "Response contains content"


def run_edge_case_validation(
    response: str, edge_case: EdgeCasePrompt
) -> tuple[bool, str]:
    """Validate a response against an edge case expectation."""
    match edge_case.expected_behavior:
        case "boxed_format":
            return _validate_boxed(response, edge_case)
        case "numeric":
            return _validate_numeric(response)
        case "pattern_match":
            return _validate_pattern(response)
        case _:
            return False, f"Unknown expected behavior: {edge_case.expected_behavior}"
