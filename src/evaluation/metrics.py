"""Measure chord-timeline accuracy, coverage, and beat-detection quality.

Use :func:`evaluate_timeline` when a processed input song has ground-truth beat
annotations. It compares chords on a regular time grid and beats with a
one-to-one tolerance match. Use :func:`labels_at_times` or :func:`match_beats`
directly when an experiment needs only one of those operations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from src.music.chords import CHORD_TO_INDEX


@dataclass(frozen=True)
class EvaluationResult:
    chord_accuracy: float
    prediction_coverage: float
    scored_samples: int
    beat_precision: float
    beat_recall: float
    beat_f1: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def labels_at_times(
    query_times: NDArray[np.floating],
    starts: NDArray[np.floating],
    ends: NDArray[np.floating],
    labels: NDArray[np.str_],
) -> NDArray[np.str_]:
    """Look up interval labels; uncovered query times receive ``N``."""

    query = np.asarray(query_times, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    labels = np.asarray(labels, dtype="<U8")
    output = np.full(len(query), "N", dtype="<U8")
    if not len(starts):
        return output
    indices = np.searchsorted(starts, query, side="right") - 1
    valid = (indices >= 0) & (indices < len(labels))
    clipped = np.clip(indices, 0, len(labels) - 1)
    valid &= query < ends[clipped]
    output[valid] = labels[clipped[valid]]
    return output


def match_beats(
    reference: NDArray[np.floating],
    estimated: NDArray[np.floating],
    tolerance: float = 0.07,
) -> tuple[float, float, float]:
    """Greedily match sorted beat times one-to-one within a tolerance."""

    reference = np.sort(np.asarray(reference, dtype=np.float64))
    estimated = np.sort(np.asarray(estimated, dtype=np.float64))
    if not len(reference) or not len(estimated):
        return 0.0, 0.0, 0.0
    ref_index = est_index = matches = 0
    while ref_index < len(reference) and est_index < len(estimated):
        delta = reference[ref_index] - estimated[est_index]
        if abs(delta) <= tolerance:
            matches += 1
            ref_index += 1
            est_index += 1
        elif delta < 0:
            ref_index += 1
        else:
            est_index += 1
    precision = matches / len(estimated)
    recall = matches / len(reference)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_timeline(
    reference_starts: NDArray[np.floating],
    reference_ends: NDArray[np.floating],
    reference_labels: NDArray[np.str_],
    predicted_starts: NDArray[np.floating],
    predicted_ends: NDArray[np.floating],
    predicted_labels: NDArray[np.str_],
    sample_period: float = 0.05,
) -> EvaluationResult:
    """Evaluate chords over time and beat timestamps independently."""

    if not len(reference_starts):
        raise ValueError("Cannot evaluate an empty reference timeline.")
    query = (
        np.arange(reference_starts[0], reference_ends[-1], sample_period)
        + sample_period / 2
    )
    truth = labels_at_times(query, reference_starts, reference_ends, reference_labels)
    predicted = labels_at_times(
        query, predicted_starts, predicted_ends, predicted_labels
    )
    scored = np.asarray([label in CHORD_TO_INDEX for label in truth])
    if not np.any(scored):
        raise ValueError("The reference contains no supported major/minor chords.")
    estimated_beats = np.asarray(predicted_starts)
    estimated_beats = estimated_beats[
        (estimated_beats >= reference_starts[0])
        & (estimated_beats <= reference_starts[-1])
    ]
    precision, recall, f1 = match_beats(reference_starts, estimated_beats)
    return EvaluationResult(
        chord_accuracy=float(np.mean(truth[scored] == predicted[scored])),
        prediction_coverage=float(np.mean(predicted[scored] != "N")),
        scored_samples=int(scored.sum()),
        beat_precision=precision,
        beat_recall=recall,
        beat_f1=f1,
    )
