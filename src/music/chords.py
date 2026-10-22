"""Define and convert the chord labels used throughout the system.

Use this module whenever code needs the canonical 24-state ordering, needs to
normalize an annotation label, or convert between chord names and state IDs.
Keeping these operations here prevents training and inference from silently
using different state orders.

Example::

    state = chord_index(normalize_chord("Db:min"))
    label = chord_labels([state])[0]
"""

from __future__ import annotations

import re

import numpy as np

NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
CHORDS = tuple(f"{note}{quality}" for note in NOTES for quality in ("maj", "min"))
CHORD_TO_INDEX = {chord: index for index, chord in enumerate(CHORDS)}

_ENHARMONIC = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "Cb": "B",
    "Fb": "E",
    "E#": "F",
    "B#": "C",
}


def normalize_chord(label: object) -> str:
    """Map a chord label to 24 major/minor states, or ``N`` if unsupported."""

    chord = str(label).strip().strip("'\"")
    if chord.lower() in {"", "n", "none", "x", "?"}:
        return "N"
    chord = chord.replace(":", "").replace("_", "")
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(maj|min|m|M)?", chord)
    if not match:
        return "N"
    written_root = match.group(1).upper() + match.group(2)
    root = _ENHARMONIC.get(written_root, written_root)
    quality = "min" if match.group(3) in {"min", "m"} else "maj"
    normalized = root + quality
    return normalized if normalized in CHORD_TO_INDEX else "N"


def chord_index(label: str) -> int:
    """Return a state index, or ``-1`` for an unsupported chord."""

    return CHORD_TO_INDEX.get(label, -1)


def chord_labels(indices: np.ndarray) -> np.ndarray:
    """Convert model-state indices to chord labels."""

    return np.asarray(CHORDS, dtype="<U8")[np.asarray(indices, dtype=int)]
