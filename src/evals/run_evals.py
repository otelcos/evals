"""Example script for running Open Telco evaluations.

Usage:
    python -m evals.run_evals              # small sample (default)
    python -m evals.run_evals --full       # full benchmarks
"""

import argparse

from inspect_ai import eval_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full benchmarks (GSMA/ot-full-benchmarks) instead of small samples.",
    )
    args = parser.parse_args()

    success, logs = eval_set(
        tasks=[
            "telelogs/telelogs.py",
            "three_gpp/three_gpp.py",
            "teleqna/teleqna.py",
            "telemath/telemath.py",
            "oranbench/oranbench.py",
            "srsranbench/srsranbench.py",
        ],  # set how many tasks you want to run
        model=[
            "openrouter/openai/gpt-5.2",
            "openrouter/anthropic/claude-opus-4.5",
            "openrouter/google/gemini-3-flash-preview",
            "openrouter/mistralai/mistral-large-3-2512",
            "openrouter/deepseek/deepseek-v3.2",
        ],  # set models you want to run in parallel
        task_args={"full": args.full},
        log_dir="logs/leaderboard",  # set directory
        epochs=1,  # set resampling iterations
        temperature=0.0,
    )

    # for more information: https://inspect.aisi.org.uk/eval-sets.html
