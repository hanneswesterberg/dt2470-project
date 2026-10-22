"""Fit and score the Gaussian observation model used by the current HMM.

Use :class:`GaussianEmissionModel` when experimenting with emissions directly,
or call :func:`src.training.train_emission_model` for the convenient training
entry point. Inputs must be the 24D chroma vectors from :mod:`src.features`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf

from src.music import CHORDS


def shift_to_c(features: NDArray[np.floating], root_index: int) -> NDArray[np.float64]:
    """Rotate both chroma bands so a candidate root is represented as C."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 24:
        raise ValueError("Expected a feature matrix with shape (n_intervals, 24).")
    if not 0 <= root_index < 12:
        raise ValueError("root_index must be between 0 and 11.")
    return np.roll(values.reshape(-1, 2, 12), -root_index, axis=2).reshape(-1, 24)


@dataclass
class GaussianEmissionModel:
    """One shared full-covariance Gaussian per chord quality.

    Major and minor observations are transposed to C before fitting. This pools
    evidence across keys while preserving a 24-state decoder interface.
    """

    means: NDArray[np.float64] | None = None
    covariances: NDArray[np.float64] | None = None
    sample_counts: tuple[int, int] = (0, 0)
    covariance_floor: float = 1e-4

    @property
    def is_fitted(self) -> bool:
        return self.means is not None and self.covariances is not None

    def fit(
        self, features: NDArray[np.floating], state_indices: NDArray[np.integer]
    ) -> GaussianEmissionModel:
        """Fit from 24D observations and their 0--23 chord-state indices."""

        features = np.asarray(features, dtype=np.float64)
        states = np.asarray(state_indices, dtype=np.int64)
        if (
            features.ndim != 2
            or features.shape[1] != 24
            or len(features) != len(states)
        ):
            raise ValueError("features must have shape (n, 24) and one state per row.")

        grouped: list[list[NDArray[np.float64]]] = [[], []]
        for root in range(12):
            for quality in range(2):
                mask = states == 2 * root + quality
                if np.any(mask):
                    grouped[quality].append(shift_to_c(features[mask], root))

        means: list[NDArray[np.float64]] = []
        covariances: list[NDArray[np.float64]] = []
        counts: list[int] = []
        for quality, blocks in enumerate(grouped):
            if not blocks:
                raise ValueError(f"No training observations for quality {quality}.")
            samples = np.vstack(blocks)
            if len(samples) < 2:
                raise ValueError(
                    f"At least two observations are required for quality {quality}."
                )
            estimator = LedoitWolf().fit(samples)
            means.append(estimator.location_)
            covariances.append(
                estimator.covariance_
                + self.covariance_floor * np.eye(features.shape[1])
            )
            counts.append(len(samples))
        self.means = np.asarray(means)
        self.covariances = np.asarray(covariances)
        self.sample_counts = (counts[0], counts[1])
        return self

    def score(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return log-likelihoods with shape ``(n_intervals, 24)``."""

        if not self.is_fitted:
            raise RuntimeError("Fit the emission model before scoring observations.")
        features = np.asarray(features, dtype=np.float64)
        scores = np.empty((len(features), len(CHORDS)), dtype=np.float64)
        for root in range(12):
            aligned = shift_to_c(features, root)
            for quality in range(2):
                state = 2 * root + quality
                scores[:, state] = multivariate_normal.logpdf(
                    aligned,
                    mean=self.means[quality],
                    cov=self.covariances[quality],
                )
        return scores
