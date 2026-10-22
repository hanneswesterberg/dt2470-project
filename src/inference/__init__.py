"""End-to-end inference orchestration and prediction timelines.

Use :class:`ChordRecognitionPipeline` when converting an audio file into a
beat-synchronous chord timeline.
"""

from src.inference.pipeline import (
    ChordRecognitionPipeline,
    ChordTimeline,
    PreparedAudio,
    beat_intervals,
    prepare_audio,
)

__all__ = [
    "ChordRecognitionPipeline",
    "ChordTimeline",
    "PreparedAudio",
    "beat_intervals",
    "prepare_audio",
]
