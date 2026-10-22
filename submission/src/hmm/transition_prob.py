"""Learn chord-to-chord probabilities for the current HMM.

Use :class:`RelativeTransitionModel` to pool transitions by previous quality,
root movement, and next quality. For the standard training entry point, call
``src.training.train_transition_model(song_state_sequences)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from src.music import CHORDS


@dataclass
class RelativeTransitionModel:
    """Transposition-invariant transitions pooled by root motion and quality."""

    smoothing: float = 0.5
    matrix: NDArray[np.float64] | None = None
    relative_probabilities: NDArray[np.float64] | None = None

    def fit(self, sequences: Iterable[NDArray[np.integer]]) -> RelativeTransitionModel:
        """Learn transitions from sequences; negative states are skipped."""

        if self.smoothing <= 0:
            raise ValueError(
                "smoothing must be positive so all transitions remain possible."
            )
        counts = np.full((2, 12, 2), self.smoothing, dtype=np.float64)
        for sequence in sequences:
            states = np.asarray(sequence, dtype=np.int64)
            for previous, current in pairwise(states):
                if previous < 0 or current < 0:
                    continue
                previous_root, previous_quality = divmod(int(previous), 2)
                current_root, current_quality = divmod(int(current), 2)
                interval = (current_root - previous_root) % 12
                counts[previous_quality, interval, current_quality] += 1

        relative = counts / counts.sum(axis=(1, 2), keepdims=True)
        matrix = np.empty((len(CHORDS), len(CHORDS)), dtype=np.float64)
        for previous in range(len(CHORDS)):
            for current in range(len(CHORDS)):
                interval = (current // 2 - previous // 2) % 12
                matrix[previous, current] = relative[
                    previous % 2, interval, current % 2
                ]
        self.matrix = matrix
        self.relative_probabilities = relative
        return self


def simple_transition_matrix(stay_probability: float = 0.85) -> NDArray[np.float64]:
    """Return a baseline matrix with one stay probability and uniform changes."""

    if not 0 < stay_probability < 1:
        raise ValueError("stay_probability must be strictly between zero and one.")
    n_states = len(CHORDS)
    matrix = np.full(
        (n_states, n_states), (1 - stay_probability) / (n_states - 1), dtype=np.float64
    )
    np.fill_diagonal(matrix, stay_probability)
    return matrix
