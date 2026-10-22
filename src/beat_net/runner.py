"""Execute one BeatNet prediction inside the isolated Python 3.9 process.

This module is an internal subprocess entry point. Normal application code
should use :class:`src.beat_net.BeatNetTracker`, which manages the temporary
result file and converts events into beat timestamps.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: runner.py AUDIO_PATH OUTPUT_PATH MODEL_NUMBER")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from BeatNet.BeatNet import BeatNet

    audio_path, output_path, model_number = sys.argv[1:]
    model = BeatNet(
        int(model_number),
        mode="offline",
        inference_model="DBN",
        plot=[],
        thread=False,
    )
    events = np.asarray(model.process(audio_path))
    if events.ndim != 2 or events.shape[1] < 1:
        raise RuntimeError(f"Unexpected BeatNet output shape: {events.shape}")
    np.save(Path(output_path), events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
