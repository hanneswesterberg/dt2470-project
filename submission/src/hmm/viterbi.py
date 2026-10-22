"""Decode the most likely complete chord sequence from HMM scores.

Use :func:`viterbi_decode` when comparing decoders directly. Normal inference
calls it through :class:`src.models.ChordRecognitionModel`, which supplies the
emission scores, transition matrix, and tuned transition weight.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def viterbi_decode(
    log_emissions: NDArray[np.floating],
    transition_matrix: NDArray[np.floating],
    transition_weight: float = 1.0,
    initial_probabilities: NDArray[np.floating] | None = None,
) -> NDArray[np.int64]:
    """Return the most likely state sequence for an entire observation series."""

    emissions = np.asarray(log_emissions, dtype=np.float64)
    transitions = np.asarray(transition_matrix, dtype=np.float64)
    if emissions.ndim != 2:
        raise ValueError("log_emissions must be a two-dimensional matrix.")
    n_steps, n_states = emissions.shape
    if transitions.shape != (n_states, n_states):
        raise ValueError("transition_matrix shape must match the number of states.")
    if transition_weight < 0:
        raise ValueError("transition_weight cannot be negative.")
    if n_steps == 0:
        return np.empty(0, dtype=np.int64)

    if initial_probabilities is None:
        initial = np.full(n_states, 1 / n_states, dtype=np.float64)
    else:
        initial = np.asarray(initial_probabilities, dtype=np.float64)
        if (
            initial.shape != (n_states,)
            or np.any(initial < 0)
            or not np.isclose(initial.sum(), 1)
        ):
            raise ValueError("initial_probabilities must be a normalized state vector.")

    log_transitions = transition_weight * np.log(np.maximum(transitions, 1e-300))
    dynamic = np.empty((n_steps, n_states), dtype=np.float64)
    backpointers = np.zeros((n_steps, n_states), dtype=np.int32)
    dynamic[0] = emissions[0] + np.log(np.maximum(initial, 1e-300))

    for step in range(1, n_steps):
        candidates = dynamic[step - 1, :, None] + log_transitions
        backpointers[step] = np.argmax(candidates, axis=0)
        dynamic[step] = (
            emissions[step] + candidates[backpointers[step], np.arange(n_states)]
        )

    path = np.empty(n_steps, dtype=np.int64)
    path[-1] = int(np.argmax(dynamic[-1]))
    for step in range(n_steps - 2, -1, -1):
        path[step] = backpointers[step + 1, path[step + 1]]
    return path


# Short alias for code based directly on the notebook.
viterbi = viterbi_decode
