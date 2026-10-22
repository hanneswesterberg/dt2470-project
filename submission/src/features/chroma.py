"""Convert audio into beat-synchronous CQT chroma feature vectors.

Use :class:`CQTChromaExtractor` in both training and inference so observations
have the same 24D representation: 12 bass pitch classes followed by 12
midrange pitch classes. A typical caller loads audio and supplies interval
boundaries::

    extractor = CQTChromaExtractor()
    audio, duration = extractor.load_audio("song.wav")
    features = extractor.transform(audio, starts, ends)

The functions at the bottom are compatibility helpers for early project code;
new code should normally use the extractor class.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class CQTConfig:
    """Parameters for the time-frequency representation."""

    sample_rate: int = 22_050
    hop_length: int = 512
    n_bins: int = 84
    bins_per_octave: int = 12
    minimum_note: str = "C1"


class CQTChromaExtractor:
    """Extract the notebook's 24D bass/midrange chroma representation.

    The public ``transform`` method intentionally matches the shape expected by
    emission models. A different extractor can be plugged into the pipeline as
    long as it implements that same method.
    """

    def __init__(self, config: CQTConfig | None = None) -> None:
        self.config = config or CQTConfig()

    def load_audio(self, path: Path | str) -> tuple[FloatArray, float]:
        """Load mono audio at the configured sample rate."""

        audio, _ = librosa.load(path, sr=self.config.sample_rate, mono=True)
        return audio, len(audio) / self.config.sample_rate

    def spectrogram(self, audio: FloatArray) -> FloatArray:
        """Compute a magnitude Constant-Q Transform."""

        return np.abs(
            librosa.cqt(
                y=audio,
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length,
                fmin=librosa.note_to_hz(self.config.minimum_note),
                n_bins=self.config.n_bins,
                bins_per_octave=self.config.bins_per_octave,
            )
        ).astype(np.float32)

    def chromagram(self, audio: FloatArray) -> FloatArray:
        """Collapse CQT octaves into bass and midrange pitch classes."""

        cqt = self.spectrogram(audio)
        if cqt.shape[0] < 60:
            raise ValueError("The 24D representation requires at least 60 CQT bins.")
        bass = cqt[:36].reshape(3, 12, -1).sum(axis=0)
        midrange = cqt[36:60].reshape(2, 12, -1).sum(axis=0)
        return np.concatenate((bass, midrange), axis=0)

    def aggregate(
        self,
        chromagram: FloatArray,
        starts: FloatArray,
        ends: FloatArray,
    ) -> FloatArray:
        """Average and normalize chroma over the central 90% of intervals."""

        starts = np.asarray(starts, dtype=np.float64)
        ends = np.asarray(ends, dtype=np.float64)
        if starts.shape != ends.shape or starts.ndim != 1:
            raise ValueError(
                "starts and ends must be equally sized one-dimensional arrays."
            )
        if np.any(ends <= starts):
            raise ValueError("Every interval end must be greater than its start.")
        if len(starts) == 0:
            return np.empty((0, chromagram.shape[0]), dtype=np.float32)

        frame_times = librosa.frames_to_time(
            np.arange(chromagram.shape[1]),
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
        )
        inner_starts = starts + 0.05 * (ends - starts)
        inner_ends = ends - 0.05 * (ends - starts)
        first = np.searchsorted(frame_times, inner_starts, side="left")
        last = np.searchsorted(frame_times, inner_ends, side="left")
        first = np.clip(first, 0, chromagram.shape[1] - 1)
        last = np.clip(np.maximum(first + 1, last), 1, chromagram.shape[1])

        cumulative = np.concatenate(
            (
                np.zeros((chromagram.shape[0], 1), dtype=np.float64),
                np.cumsum(chromagram, axis=1, dtype=np.float64),
            ),
            axis=1,
        )
        features = (cumulative[:, last] - cumulative[:, first]).T
        features /= np.maximum(last - first, 1)[:, None]
        for band in (slice(0, 12), slice(12, 24)):
            norms = np.linalg.norm(features[:, band], axis=1, keepdims=True)
            features[:, band] /= np.maximum(norms, 1e-10)
        return features.astype(np.float32)

    def transform(
        self, audio: FloatArray, starts: FloatArray, ends: FloatArray
    ) -> FloatArray:
        """Compute one 24D feature vector per interval."""

        return self.aggregate(self.chromagram(audio), starts, ends)


# Compatibility helpers retained for collaborators already using the first API.
DEFAULT_CONFIG = CQTConfig()
SR = DEFAULT_CONFIG.sample_rate
HOP_LENGTH = DEFAULT_CONFIG.hop_length
N_BINS = DEFAULT_CONFIG.n_bins


def compute_cqt(audio_path: Path | str) -> tuple[FloatArray, float]:
    extractor = CQTChromaExtractor()
    audio, duration = extractor.load_audio(audio_path)
    return extractor.spectrogram(audio), duration


def compute_chroma(
    cqt: FloatArray, start: float, end: float, low_bin: int = 0, high_bin: int = 84
) -> FloatArray:
    """Compute normalized 12D chroma for one CQT band and time interval."""

    if end <= start:
        raise ValueError("End time must be greater than start time.")
    if not 0 <= low_bin < high_bin <= cqt.shape[0]:
        raise ValueError("Invalid CQT frequency-bin range.")
    duration = end - start
    first = max(
        0, min(int((start + 0.05 * duration) * SR / HOP_LENGTH), cqt.shape[1] - 1)
    )
    last = max(
        first + 1, min(int((end - 0.05 * duration) * SR / HOP_LENGTH), cqt.shape[1])
    )
    average = cqt[low_bin:high_bin, first:last].mean(axis=1)
    chroma = np.zeros(12, dtype=np.float64)
    np.add.at(chroma, np.arange(low_bin, high_bin) % 12, average)
    return chroma / max(np.linalg.norm(chroma), 1e-10)


def compute_chroma_24d(cqt: FloatArray, start: float, end: float) -> FloatArray:
    return np.concatenate(
        (
            compute_chroma(cqt, start, end, 0, 36),
            compute_chroma(cqt, start, end, 36, 60),
        )
    )


def compute_chroma_intervals(
    cqt: FloatArray,
    intervals: FloatArray,
    feature_fn: Callable[[FloatArray, float, float], FloatArray] = compute_chroma_24d,
) -> FloatArray:
    return np.asarray([feature_fn(cqt, start, end) for start, end in intervals])
