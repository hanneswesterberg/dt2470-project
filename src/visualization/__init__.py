"""Visual outputs for inspecting chord-recognition results.

Use :func:`write_chord_chart` after inference to create a readable PNG chart
with four beats per displayed bar. The chart can optionally include and score
against a ground-truth timeline.
"""

from src.visualization.chord_chart import write_chord_chart

__all__ = ["write_chord_chart"]
