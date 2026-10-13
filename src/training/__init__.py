"""Public training API for individual HMM components or complete models.

Typical usage::

    from src.training import train_model
    model = train_model(dataset_songs)

For component experiments, use ``train_emission_model`` or
``train_transition_model`` independently, or ``train_hmm`` when prepared
feature/state arrays are already available.
"""

from src.training.trainer import (
    SongFeatures,
    choose_transition_weight,
    extract_song_features,
    split_songs,
    train_emission_from_songs,
    train_emission_model,
    train_hmm,
    train_model,
    train_transition_from_songs,
    train_transition_model,
)

__all__ = [
    "SongFeatures",
    "choose_transition_weight",
    "extract_song_features",
    "split_songs",
    "train_emission_from_songs",
    "train_emission_model",
    "train_hmm",
    "train_model",
    "train_transition_from_songs",
    "train_transition_model",
]
