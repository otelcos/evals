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


# Edge cases to test format parsing
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
        expected_behavior="boxed_format",  # Should extract last
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
    """Validate response contains valid \\boxed{} format.

    Args:
        response: The model's response text.

    Returns:
        Tuple of (is_valid, content_or_error_message).
        If valid, returns (True, extracted_content).
        If invalid, returns (False, error_message).
    """
    if not response:
        return False, "Empty response"

    matches = BOXED_PATTERN.findall(response)
    if not matches:
        return False, "No \\boxed{} found in response"

    # Return the last boxed content (following telelogs pattern)
    return True, matches[-1].strip()


def run_edge_case_validation(
    response: str, edge_case: EdgeCasePrompt
) -> tuple[bool, str]:
    """Validate a response against an edge case expectation.

    Args:
        response: The model's response text.
        edge_case: The edge case definition with expected behavior.

    Returns:
        Tuple of (is_valid, explanation).
    """
    if edge_case.expected_behavior == "boxed_format":
        valid, content = validate_boxed_response(response)
        if not valid:
            return False, content  # content is error message

        if edge_case.should_have_content and not content:
            return False, "Expected non-empty content in \\boxed{}"

        return True, f"Valid boxed format: {content}"

    elif edge_case.expected_behavior == "numeric":
        valid, content = validate_boxed_response(response)
        if not valid:
            return False, content

        # Check if content contains a number
        if not re.search(r"\d+", content):
            return False, f"Expected numeric content, got: {content}"

        return True, f"Valid numeric format: {content}"

    elif edge_case.expected_behavior == "pattern_match":
        # For pattern matching tasks like 3GPP
        if not response.strip():
            return False, "Empty response"
        return True, "Response contains content"

    return False, f"Unknown expected behavior: {edge_case.expected_behavior}"
