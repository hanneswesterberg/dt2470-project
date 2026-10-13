"""Probability models and decoder used by chord recognition.

Import from this package for low-level HMM experiments. For normal dataset
training, the public functions in :mod:`src.training` provide simpler entry
points and preserve song boundaries correctly.
"""

from src.hmm.emission import GaussianEmissionModel
from src.hmm.transition_prob import RelativeTransitionModel, simple_transition_matrix
from src.hmm.viterbi import viterbi_decode

__all__ = [
    "GaussianEmissionModel",
    "RelativeTransitionModel",
    "simple_transition_matrix",
    "viterbi_decode",
]
