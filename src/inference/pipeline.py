"""Run a fitted chord model on one audio file and write its timeline.

Use :class:`ChordRecognitionPipeline` for inference only; model fitting belongs
in :mod:`src.training`. The pipeline connects a beat tracker, feature extractor,
and fitted model without embedding their algorithms here::

    pipeline = ChordRecognitionPipeline(model, beat_tracker, feature_extractor)
    timeline = pipeline.process("input/song.wav")
    timeline.write_csv("output/song_chords.csv")
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.beat_net.beat_tracker import BeatTracker
from src.features import CQTChromaExtractor
from src.models import ChordRecognitionModel


@dataclass(frozen=True)
class ChordTimeline:
    """A model prediction represented by time intervals."""

    starts: NDArray[np.float64]
    ends: NDArray[np.float64]
    labels: NDArray[np.str_]

    def write_csv(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("start_seconds", "end_seconds", "chord"))
            writer.writerows(
                (f"{start:.6f}", f"{end:.6f}", label)
                for start, end, label in zip(self.starts, self.ends, self.labels)
            )

    def write_json(self, path: Path | str) -> None:
        """Write the same event timeline as a machine-readable JSON array."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        events = [
            {
                "start_seconds": float(start),
                "end_seconds": float(end),
                "chord": str(label),
            }
            for start, end, label in zip(self.starts, self.ends, self.labels)
        ]
        target.write_text(json.dumps(events, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class PreparedAudio:
    """Beat intervals and features computed once for one input song."""

    duration: float
    starts: NDArray[np.float64]
    ends: NDArray[np.float64]
    features: NDArray[np.floating]

    def predict(self, model: ChordRecognitionModel) -> ChordTimeline:
        """Decode this prepared song with one fitted model."""

        return ChordTimeline(self.starts, self.ends, model.predict(self.features))


def beat_intervals(
    beat_times: NDArray[np.floating], audio_duration: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Turn valid beat timestamps into consecutive decoding intervals."""

    starts = np.unique(np.asarray(beat_times, dtype=np.float64))
    starts = starts[np.isfinite(starts) & (starts >= 0) & (starts < audio_duration)]
    if len(starts) < 2:
        raise RuntimeError("Beat tracker returned fewer than two valid beats.")
    return starts, np.r_[starts[1:], audio_duration]


def prepare_audio(
    audio_path: Path | str,
    beat_tracker: BeatTracker,
    feature_extractor: CQTChromaExtractor,
) -> PreparedAudio:
    """Detect beats and extract features once for use with one or more models."""

    audio, duration = feature_extractor.load_audio(audio_path)
    starts, ends = beat_intervals(beat_tracker.track(audio_path), duration)
    features = feature_extractor.transform(audio, starts, ends)
    return PreparedAudio(duration, starts, ends, features)


@dataclass
class ChordRecognitionPipeline:
    """Compose interchangeable beat, feature, and sequence-model components."""

    model: ChordRecognitionModel
    beat_tracker: BeatTracker
    feature_extractor: CQTChromaExtractor

    def process(self, audio_path: Path | str) -> ChordTimeline:
        """Predict one chord for every automatically detected beat interval."""

        prepared = prepare_audio(audio_path, self.beat_tracker, self.feature_extractor)
        return prepared.predict(self.model)
