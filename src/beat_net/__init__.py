"""Public API for pluggable beat-tracking implementations.

Despite the historical package name, this contains both BeatNet and the
portable librosa fallback. Use ``create_beat_tracker("auto")`` for the normal
selection behavior.
"""

from src.beat_net.beat_tracker import (
    BeatNetTracker,
    BeatTracker,
    LibrosaBeatTracker,
    create_beat_tracker,
)

__all__ = ["BeatNetTracker", "BeatTracker", "LibrosaBeatTracker", "create_beat_tracker"]
