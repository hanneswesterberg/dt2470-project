"""Metrics for comparing predicted chord timelines and beats with references.

The main entry point is :func:`evaluate_timeline`; lower-level lookup and beat
matching helpers are exported for experiments and tests.
"""

from src.evaluation.metrics import (
    EvaluationResult,
    evaluate_timeline,
    labels_at_times,
    match_beats,
)

__all__ = [
    "EvaluationResult",
    "evaluate_timeline",
    "labels_at_times",
    "match_beats",
]
