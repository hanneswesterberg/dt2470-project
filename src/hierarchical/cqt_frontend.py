"""CQT72 continuous-audio frontend used by the hierarchical model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CQT72Config:
    sample_rate: int = 22_050
    hop_length: int = 512
    window_frames: int = 32
    step: float = 0.2
    chunk_seconds: float = 30.0
    overlap_seconds: float = 1.0
    bins_per_octave: int = 36
    n_bins: int = 216
    minimum_note: str = "C1"

    def cache_values(self) -> dict[str, int | float | str]:
        return asdict(self)


class ContinuousCQT72Extractor:
    """Create overlapping 72x32 windows on a 200 ms observation grid."""

    def __init__(self, config: CQT72Config | None = None) -> None:
        self.config = config or CQT72Config()

    def load_audio(self, path: Path | str) -> tuple[NDArray[np.float32], float]:
        audio, _ = librosa.load(
            str(path), sr=self.config.sample_rate, mono=True
        )
        audio = np.asarray(audio, dtype=np.float32)
        if not len(audio):
            raise ValueError(f"Audio file is empty: {path}")
        return audio, len(audio) / self.config.sample_rate

    def continuous_cqt(
        self, audio: NDArray[np.floating]
    ) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
        cfg = self.config
        duration = len(audio) / cfg.sample_rate
        chunks: list[NDArray[np.float32]] = []
        times: list[NDArray[np.float64]] = []
        for begin in np.arange(0, duration, cfg.chunk_seconds):
            finish = min(duration, begin + cfg.chunk_seconds)
            lo = max(0.0, begin - cfg.overlap_seconds)
            hi = min(duration, finish + cfg.overlap_seconds)
            clip = audio[int(lo * cfg.sample_rate) : int(hi * cfg.sample_rate)]
            harmonic, _ = librosa.effects.hpss(clip, margin=(2.0, 4.0))
            cqt = np.abs(
                librosa.cqt(
                    harmonic,
                    sr=cfg.sample_rate,
                    hop_length=cfg.hop_length,
                    fmin=librosa.note_to_hz(cfg.minimum_note),
                    bins_per_octave=cfg.bins_per_octave,
                    n_bins=cfg.n_bins,
                )
            ).astype(np.float32)
            cqt = cqt.reshape(72, 3, -1).sum(axis=1)
            frame_times = (
                librosa.frames_to_time(
                    np.arange(cqt.shape[1]),
                    sr=cfg.sample_rate,
                    hop_length=cfg.hop_length,
                )
                + lo
            )
            keep = (frame_times >= begin) & (frame_times < finish)
            chunks.append(cqt[:, keep])
            times.append(frame_times[keep])
        return np.concatenate(chunks, axis=1), np.concatenate(times)

    def rolling_windows(
        self,
        cqt: NDArray[np.floating],
        frame_times: NDArray[np.floating],
        duration: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        cfg = self.config
        frame_times = np.asarray(frame_times, dtype=np.float64)
        if cqt.shape[0] != 72 or cqt.shape[1] != len(frame_times):
            raise ValueError("Expected CQT shape (72, frames) matching frame_times.")
        if len(frame_times) < 2:
            raise ValueError("At least two CQT frames are required for interpolation.")
        centers = np.arange(cfg.step / 2, duration, cfg.step, dtype=np.float64)
        output = np.empty((len(centers), 72, cfg.window_frames), np.float32)
        offsets = np.linspace(
            -(cfg.window_frames - 1) * cfg.hop_length / (2 * cfg.sample_rate),
            (cfg.window_frames - 1) * cfg.hop_length / (2 * cfg.sample_rate),
            cfg.window_frames,
        )
        for index, center in enumerate(centers):
            targets = np.clip(center + offsets, frame_times[0], frame_times[-1])
            right = np.clip(np.searchsorted(frame_times, targets), 1, len(frame_times) - 1)
            left = right - 1
            weights = (targets - frame_times[left]) / (
                frame_times[right] - frame_times[left] + 1e-10
            )
            values = cqt[:, left] * (1 - weights)[None, :] + cqt[:, right] * weights[None, :]
            values = np.sqrt(np.maximum(values, 0))
            values /= values.sum(axis=0, keepdims=True) + 1e-8
            output[index] = values
        return centers, output

    def transform_file(
        self, path: Path | str
    ) -> tuple[NDArray[np.float64], NDArray[np.float32], float]:
        audio, duration = self.load_audio(path)
        cqt, frame_times = self.continuous_cqt(audio)
        centers, windows = self.rolling_windows(cqt, frame_times, duration)
        return centers, windows, duration
