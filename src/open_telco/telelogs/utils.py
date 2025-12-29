from collections import Counter
from itertools import groupby
from operator import attrgetter

from inspect_ai.scorer import CORRECT, metric, SampleScore, Value


@metric
def maj_at_k():
    """Majority voting metric across epochs.

    For each sample, takes the most common predicted answer across all epochs
    and checks if that majority answer is correct. This implements ensemble
    voting to reduce variance in model predictions.

    Returns:
        Metric function that computes majority-vote accuracy.
    """

    def metric_fn(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0

        # Group scores by sample_id
        sorted_scores = sorted(scores, key=attrgetter("sample_id"))
        grouped = {
            sample_id: list(group)
            for sample_id, group in groupby(sorted_scores, key=attrgetter("sample_id"))
        }

        correct = 0
        for sample_scores in grouped.values():
            # Collect non-empty answers
            answers = [s.score.answer for s in sample_scores if s.score.answer]
            if not answers:
                continue

            # Find majority answer
            majority = Counter(answers).most_common(1)[0][0]

            # Check if any score with the majority answer is correct
            correct += any(
                s.score.value == CORRECT and s.score.answer == majority
                for s in sample_scores
            )

        return correct / len(grouped) if grouped else 0.0

    return metric_fn
