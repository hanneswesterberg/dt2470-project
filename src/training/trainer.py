"""Train HMM components or a complete model from annotated AAM songs.

Use :func:`train_model` for the standard leak-free workflow: split songs,
extract annotated-beat features, train both components, tune the transition
weight on validation songs, and return a checkpoint-ready model.

For focused experiments, train either component directly::

    emission = train_emission_model(features, states)
    transition = train_transition_model(state_sequences, smoothing=0.5)

Or assemble both from already-extracted arrays without running dataset I/O::

    model = train_hmm(features, states, state_sequences)

This module intentionally contains orchestration rather than probability math;
the actual estimators live in :mod:`src.hmm`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from data.annotations import labels_to_indices, load_annotations
from data.load_data import SongFiles
from src.features import CQTChromaExtractor
from src.hmm import GaussianEmissionModel, RelativeTransitionModel
from src.models import ChordRecognitionModel

DEFAULT_WEIGHT_GRID = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)


@dataclass(frozen=True)
class SongFeatures:
    """Prepared observations and reference states for one annotated song."""

    song_id: str
    features: NDArray[np.float32]
    states: NDArray[np.int64]


def split_songs(
    songs: Sequence[SongFiles], seed: int = 42, validation_fraction: float = 0.2
) -> tuple[list[SongFiles], list[SongFiles]]:
    """Create deterministic, song-disjoint training and validation splits."""

    if len(songs) < 2:
        raise ValueError(
            "At least two songs are needed for train/validation splitting."
        )
    shuffled = np.random.default_rng(seed).permutation(list(songs)).tolist()
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    return shuffled[:-validation_count], shuffled[-validation_count:]


def extract_song_features(
    song: SongFiles, extractor: CQTChromaExtractor
) -> SongFeatures:
    """Extract annotated-beat features for one training song."""

    audio, duration = extractor.load_audio(song.audio)
    annotations = load_annotations(song.annotations)
    starts, ends, labels = annotations.intervals(duration)
    if len(starts) < 2:
        raise ValueError(f"Song {song.song_id} has too few valid annotated intervals.")
    return SongFeatures(
        song_id=song.song_id,
        features=extractor.transform(audio, starts, ends),
        states=labels_to_indices(labels),
    )


def train_emission_from_songs(
    songs: Sequence[SongFiles],
    extractor: CQTChromaExtractor | None = None,
    progress: Callable[[str], None] | None = None,
) -> GaussianEmissionModel:
    """Extract annotated-beat features and train only the emission component."""

    if not songs:
        raise ValueError("At least one song is required for emission training.")
    extractor = extractor or CQTChromaExtractor()
    report = progress or (lambda _: None)
    prepared: list[SongFeatures] = []
    for index, song in enumerate(songs, start=1):
        report(f"Emission feature extraction {index}/{len(songs)}: {song.song_id}")
        prepared.append(extract_song_features(song, extractor))
    return train_emission_model(
        np.vstack([song.features for song in prepared]),
        np.concatenate([song.states for song in prepared]),
    )


def train_transition_from_songs(
    songs: Sequence[SongFiles],
    smoothing: float = 0.5,
    progress: Callable[[str], None] | None = None,
) -> RelativeTransitionModel:
    """Train only transitions from annotations, without loading any audio."""

    if not songs:
        raise ValueError("At least one song is required for transition training.")
    report = progress or (lambda _: None)
    sequences: list[NDArray[np.int64]] = []
    for index, song in enumerate(songs, start=1):
        report(f"Transition annotations {index}/{len(songs)}: {song.song_id}")
        sequences.append(labels_to_indices(load_annotations(song.annotations).labels))
    return train_transition_model(sequences, smoothing=smoothing)


def train_emission_model(
    features: NDArray[np.floating],
    states: NDArray[np.integer],
    model: GaussianEmissionModel | None = None,
) -> GaussianEmissionModel:
    """Fit one emission model from feature rows and chord-state IDs.

    Unsupported states (negative IDs) are removed here, so callers may pass
    arrays produced directly by :func:`extract_song_features`.

    Args:
        features: Matrix with one 24D chroma vector per annotated interval.
        states: State ID per row; values below zero represent unsupported labels.
        model: Optional custom/subclassed Gaussian estimator to fit in place.

    Returns:
        The fitted emission model, ready for ``score(features)``.
    """

    observations = np.asarray(features)
    state_ids = np.asarray(states, dtype=np.int64)
    if len(observations) != len(state_ids):
        raise ValueError("features and states must contain the same number of rows.")
    supported = state_ids >= 0
    if not np.any(supported):
        raise ValueError("No supported major/minor observations were supplied.")
    estimator = model or GaussianEmissionModel()
    return estimator.fit(observations[supported], state_ids[supported])


def train_transition_model(
    state_sequences: Sequence[NDArray[np.integer]],
    smoothing: float = 0.5,
    model: RelativeTransitionModel | None = None,
) -> RelativeTransitionModel:
    """Fit the HMM transition component from complete song-state sequences.

    Sequence boundaries remain separate, so the estimator never invents a
    transition from the end of one song to the beginning of another. Negative
    states are skipped by the estimator.

    Args:
        state_sequences: One ordered state array per training song.
        smoothing: Additive smoothing used when creating a default estimator.
        model: Optional custom/subclassed transition estimator to fit in place.

    Returns:
        A fitted transition model with a normalized 24x24 ``matrix``.
    """

    if not state_sequences:
        raise ValueError("At least one state sequence is required.")
    estimator = model or RelativeTransitionModel(smoothing=smoothing)
    return estimator.fit(state_sequences)


def train_hmm(
    features: NDArray[np.floating],
    states: NDArray[np.integer],
    state_sequences: Sequence[NDArray[np.integer]],
    *,
    smoothing: float = 0.5,
    transition_weight: float = 1.0,
    emission_model: GaussianEmissionModel | None = None,
    transition_model: RelativeTransitionModel | None = None,
) -> ChordRecognitionModel:
    """Train and assemble an HMM from already-extracted training arrays.

    This is the simplest entry point for notebook experiments that already have
    features and labels in memory. It deliberately does not split data or tune
    ``transition_weight``; use :func:`train_model` for the complete dataset
    workflow with held-out validation.

    Args:
        features: All 24D training observations.
        states: One chord-state ID per feature row.
        state_sequences: The same states kept separate by song for transitions.
        smoothing: Additive transition smoothing.
        transition_weight: Initial decoder weight, normally tuned separately.
        emission_model: Optional alternative emission estimator.
        transition_model: Optional alternative transition estimator.

    Returns:
        A fitted model ready to decode feature sequences.
    """

    emission = train_emission_model(features, states, model=emission_model)
    transition = train_transition_model(
        state_sequences,
        smoothing=smoothing,
        model=transition_model,
    )
    return ChordRecognitionModel(
        emission=emission,
        transition=transition,
        transition_weight=transition_weight,
    )


def choose_transition_weight(
    model: ChordRecognitionModel,
    validation: Sequence[SongFeatures],
    weights: Sequence[float] = DEFAULT_WEIGHT_GRID,
) -> tuple[float, dict[float, float]]:
    """Select transition influence by pooled validation-set beat accuracy."""

    if model.transition.matrix is None:
        raise RuntimeError(
            "The transition model must be fitted before tuning its weight."
        )
    correct = {float(weight): 0 for weight in weights}
    totals = {float(weight): 0 for weight in weights}
    original_weight = model.transition_weight
    try:
        for song in validation:
            supported = song.states >= 0
            if not np.any(supported):
                continue
            for weight in correct:
                model.transition_weight = weight
                prediction = model.decode_indices(song.features)
                correct[weight] += int(
                    np.sum(prediction[supported] == song.states[supported])
                )
                totals[weight] += int(supported.sum())
    finally:
        model.transition_weight = original_weight
    scores = {weight: correct[weight] / max(1, totals[weight]) for weight in correct}
    best = max(scores, key=scores.get)
    return best, scores


def train_model(
    songs: Sequence[SongFiles],
    extractor: CQTChromaExtractor | None = None,
    seed: int = 42,
    progress: Callable[[str], None] | None = None,
) -> ChordRecognitionModel:
    """Fit emissions/transitions and tune decoding without validation leakage."""

    extractor = extractor or CQTChromaExtractor()
    report = progress or (lambda _: None)
    training_songs, validation_songs = split_songs(songs, seed=seed)

    def extract_many(items: Sequence[SongFiles], purpose: str) -> list[SongFeatures]:
        results = []
        for index, song in enumerate(items, start=1):
            report(f"{purpose} {index}/{len(items)}: {song.song_id}")
            results.append(extract_song_features(song, extractor))
        return results

    training = extract_many(training_songs, "Training feature extraction")
    validation = extract_many(validation_songs, "Validation feature extraction")
    features = np.vstack([item.features for item in training])
    states = np.concatenate([item.states for item in training])
    model = train_hmm(
        features,
        states,
        [item.states for item in training],
    )
    best_weight, validation_scores = choose_transition_weight(model, validation)
    model.transition_weight = best_weight
    model.metadata = {
        "training_song_ids": [song.song_id for song in training_songs],
        "validation_song_ids": [song.song_id for song in validation_songs],
        "validation_weight_scores": validation_scores,
        "seed": seed,
    }
    return model
