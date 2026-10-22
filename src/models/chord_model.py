"""Assemble HMM components into a trainable, serializable chord model.

Use :class:`ChordRecognitionModel` after fitting an emission model and a
transition model, or use ``fit`` to train both from prepared arrays. The class
also owns sequence decoding and the trusted local ``.pkl`` checkpoint format::

    model = ChordRecognitionModel.load("models/experiment.pkl")
    labels = model.predict(features)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.hmm.emission import GaussianEmissionModel
from src.hmm.transition_prob import RelativeTransitionModel
from src.hmm.viterbi import viterbi_decode
from src.music.chords import CHORDS, chord_labels

MODEL_FORMAT_VERSION = 1


@dataclass
class ChordRecognitionModel:
    """Emission, transition, and decoding choices assembled in one object."""

    emission: GaussianEmissionModel = field(default_factory=GaussianEmissionModel)
    transition: RelativeTransitionModel = field(default_factory=RelativeTransitionModel)
    transition_weight: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_fitted(self) -> bool:
        return self.emission.is_fitted and self.transition.matrix is not None

    def fit(
        self,
        features: NDArray[np.floating],
        states: NDArray[np.integer],
        state_sequences: list[NDArray[np.integer]],
    ) -> ChordRecognitionModel:
        """Fit both probabilistic components from training observations."""

        self.emission.fit(features, states)
        self.transition.fit(state_sequences)
        return self

    def emission_scores(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        return self.emission.score(features)

    def decode_indices(self, features: NDArray[np.floating]) -> NDArray[np.int64]:
        """Decode a complete song feature sequence."""

        if self.transition.matrix is None:
            raise RuntimeError("Fit or load the transition model before decoding.")
        return viterbi_decode(
            self.emission_scores(features),
            self.transition.matrix,
            self.transition_weight,
        )

    def predict(self, features: NDArray[np.floating]) -> NDArray[np.str_]:
        return chord_labels(self.decode_indices(features))

    def save(self, path: Path | str) -> None:
        """Atomically save a trusted local checkpoint."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": MODEL_FORMAT_VERSION,
            "chords": CHORDS,
            "model": self,
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream)
        temporary.replace(target)

    @classmethod
    def load(cls, path: Path | str) -> ChordRecognitionModel:
        """Load a checkpoint created by this project.

        Pickle files can execute code while loading; only load checkpoints made
        locally or received from a trusted collaborator.
        """

        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("format_version") != MODEL_FORMAT_VERSION:
            raise ValueError("Unsupported model checkpoint version.")
        if tuple(payload.get("chords", ())) != CHORDS:
            raise ValueError(
                "Checkpoint chord vocabulary differs from this code version."
            )
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError("Checkpoint does not contain a ChordRecognitionModel.")
        return model
