"""Shared music-domain definitions, currently the supported chord vocabulary.

Most callers should import the constants or conversion helpers directly from
this package rather than duplicating chord ordering assumptions.
"""

from src.music.chords import (
    CHORD_TO_INDEX,
    CHORDS,
    NOTES,
    chord_index,
    chord_labels,
    normalize_chord,
)

__all__ = [
    "CHORDS",
    "CHORD_TO_INDEX",
    "NOTES",
    "chord_index",
    "chord_labels",
    "normalize_chord",
]
