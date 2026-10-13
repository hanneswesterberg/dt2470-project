"""Parsing and normalization for AAM beat-level chord annotations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.music import chord_index, normalize_chord


@dataclass(frozen=True)
class ChordAnnotations:
    """Chord labels placed at annotated beat times."""

    times: NDArray[np.float64]
    labels: NDArray[np.str_]

    def intervals(
        self, audio_duration: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.str_]]:
        """Convert beat timestamps to valid intervals within the audio."""

        if audio_duration <= 0:
            raise ValueError("Audio duration must be positive.")
        if len(self.times) == 0:
            return self.times.copy(), self.times.copy(), self.labels.copy()
        typical = float(np.median(np.diff(self.times))) if len(self.times) > 1 else 1.0
        ends = np.minimum(
            np.r_[self.times[1:], self.times[-1] + typical], audio_duration
        )
        valid = (self.times >= 0) & (self.times < audio_duration) & (ends > self.times)
        return self.times[valid], ends[valid], self.labels[valid]


def read_arff_beatinfo(path: Path | str) -> list[tuple[float, str]]:
    """Parse timestamp/chord pairs from an AAM ``*_beatinfo.arff`` file."""

    lines = Path(path).read_bytes().decode("utf-8-sig", errors="replace").splitlines()
    rows: list[tuple[float, str]] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(("@", "%")):
            continue
        fields = next(csv.reader([line], quotechar="'", skipinitialspace=True))
        if len(fields) < 4:
            continue
        try:
            timestamp = float(fields[0])
        except ValueError:
            continue
        if np.isfinite(timestamp):
            rows.append((timestamp, fields[3].strip().strip("'\"")))
    if not rows:
        raise ValueError(f"No valid annotations found in {path}")
    return sorted(rows)


def load_annotations(path: Path | str) -> ChordAnnotations:
    """Load annotations, normalize labels, and merge duplicate timestamps.

    Conflicting labels at the same timestamp are represented as ``N`` and are
    ignored by the 24-class training/evaluation code.
    """

    by_time: dict[float, list[str]] = {}
    for timestamp, label in read_arff_beatinfo(path):
        by_time.setdefault(round(timestamp, 6), []).append(normalize_chord(label))
    cleaned = [
        (timestamp, labels[0] if len(set(labels)) == 1 else "N")
        for timestamp, labels in sorted(by_time.items())
    ]
    return ChordAnnotations(
        times=np.asarray([row[0] for row in cleaned], dtype=np.float64),
        labels=np.asarray([row[1] for row in cleaned], dtype="<U8"),
    )


def labels_to_indices(labels: NDArray[np.str_]) -> NDArray[np.int64]:
    """Map labels to model-state indices; unsupported labels become ``-1``."""

    return np.asarray([chord_index(label) for label in labels], dtype=np.int64)
