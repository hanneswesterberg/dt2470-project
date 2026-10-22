"""Select or implement the beat detector used before chord recognition.

Use :func:`create_beat_tracker` for command-line/configuration choices. New beat
tracking systems can be plugged into inference by implementing the small
:class:`BeatTracker` protocol::

    class MyTracker:
        def track(self, audio_path):
            return beat_times_in_seconds
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, runtime_checkable

import librosa
import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class BeatTracker(Protocol):
    """Minimal interface required by the chord-recognition pipeline."""

    def track(self, audio_path: Path | str) -> NDArray[np.float64]:
        """Return sorted beat timestamps in seconds."""


@dataclass(frozen=True)
class LibrosaBeatTracker:
    """Portable default beat tracker supplied by librosa."""

    sample_rate: int = 22_050
    hop_length: int = 512

    def track(self, audio_path: Path | str) -> NDArray[np.float64]:
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        _, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            units="time",
        )
        return _clean_times(beats)


@dataclass(frozen=True)
class BeatNetTracker:
    """BeatNet adapter matching the configuration used in the notebook.

    BeatNet has stricter legacy dependencies than the rest of this project, so
    it runs in ``.venv-beatnet`` through a small subprocess adapter. Selecting
    BeatNet explicitly creates that environment automatically when needed.
    """

    model_number: int = 1

    def track(self, audio_path: Path | str) -> NDArray[np.float64]:
        from scripts.environments import ensure_beatnet_environment

        python = ensure_beatnet_environment()
        runner = Path(__file__).with_name("runner.py")
        with TemporaryDirectory() as directory:
            output = Path(directory) / "events.npy"
            result = subprocess.run(
                [
                    str(python),
                    str(runner),
                    str(audio_path),
                    str(output),
                    str(self.model_number),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"BeatNet failed: {detail}")
            events = np.load(output)
        if events.ndim != 2 or events.shape[1] < 1:
            raise RuntimeError(f"Unexpected BeatNet output shape: {events.shape}")
        return _clean_times(events[:, 0])


def _clean_times(times: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(times, dtype=np.float64).reshape(-1)
    return np.unique(values[np.isfinite(values) & (values >= 0)])


def create_beat_tracker(name: str) -> BeatTracker:
    """Construct a beat tracker from its CLI name."""

    if name.lower() == "auto":
        from scripts.environments import beatnet_environment_available

        name = "beatnet" if beatnet_environment_available() else "librosa"
    trackers = {"librosa": LibrosaBeatTracker, "beatnet": BeatNetTracker}
    try:
        return trackers[name.lower()]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown beat tracker {name!r}; choose from {sorted(trackers)}."
        ) from exc
