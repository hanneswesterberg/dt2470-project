"""Create bar-based PNG charts from one or more chord timelines.

The public :func:`write_chord_chart` function is intended for inference output.
It displays four consecutive beat intervals per bar and splits long songs into
multiple sections. When a reference timeline is supplied, it becomes the
chart's beat grid and prediction cells are colored by correctness.

The displayed bars are groups of four beats starting at the first available
beat. They are a visual grouping, not an estimate of musical downbeats.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch, Rectangle

from src.evaluation import labels_at_times
from src.inference import ChordTimeline
from src.music import CHORD_TO_INDEX

_CORRECT_COLOR = "#b7e4c7"
_INCORRECT_COLOR = "#f4acb7"
_REFERENCE_COLOR = "#d9e2ec"
_UNSCORED_COLOR = "#e9ecef"
_NO_CHORD_COLOR = "#f1f3f5"


def _short_label(label: str) -> str:
    """Return compact text suitable for a beat-sized chart cell."""

    if label.endswith("maj"):
        return label[:-3]
    if label.endswith("min"):
        return f"{label[:-3]}m"
    return label


def _chord_color(label: str) -> tuple[float, float, float, float] | str:
    """Assign stable colors to chord classes in prediction-only charts."""

    index = CHORD_TO_INDEX.get(label)
    if index is None:
        return _NO_CHORD_COLOR
    return plt.get_cmap("tab20")(index % 20)


def _validate_timeline(name: str, timeline: ChordTimeline) -> None:
    lengths = (len(timeline.starts), len(timeline.ends), len(timeline.labels))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Timeline {name!r} has inconsistent array lengths: {lengths}."
        )
    if not lengths[0]:
        raise ValueError(f"Timeline {name!r} is empty and cannot be charted.")


def _draw_section(
    axis: Axes,
    section_start: int,
    section_end: int,
    row_names: list[str],
    row_labels: list[np.ndarray],
    reference_labels: np.ndarray | None,
    beats_per_bar: int,
) -> None:
    beat_count = section_end - section_start
    for row_index, labels in enumerate(row_labels):
        for local_index, label in enumerate(labels[section_start:section_end]):
            absolute_index = section_start + local_index
            if reference_labels is None:
                color = _chord_color(str(label))
            elif row_index == 0:
                color = _REFERENCE_COLOR
            elif reference_labels[absolute_index] not in CHORD_TO_INDEX:
                color = _UNSCORED_COLOR
            elif label == reference_labels[absolute_index]:
                color = _CORRECT_COLOR
            else:
                color = _INCORRECT_COLOR
            axis.add_patch(
                Rectangle(
                    (local_index, row_index),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
            axis.text(
                local_index + 0.5,
                row_index + 0.5,
                _short_label(str(label)),
                ha="center",
                va="center",
                fontsize=7,
            )

    for boundary in range(0, beat_count + 1, beats_per_bar):
        axis.axvline(boundary, color="#343a40", linewidth=1.6, zorder=3)

    bar_starts = list(range(0, beat_count, beats_per_bar))
    axis.set_xticks(
        [start + min(beats_per_bar, beat_count - start) / 2 for start in bar_starts],
        [f"Bar {(section_start + start) // beats_per_bar + 1}" for start in bar_starts],
    )
    axis.tick_params(axis="x", labeltop=True, labelbottom=False, length=0, pad=4)
    axis.set_yticks(np.arange(len(row_names)) + 0.5, row_names)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.set_xlim(0, beat_count)
    axis.set_ylim(len(row_names), 0)
    axis.set_frame_on(False)


def write_chord_chart(
    song_name: str,
    predictions: Mapping[str, ChordTimeline],
    output_path: Path | str,
    reference: ChordTimeline | None = None,
    *,
    beats_per_bar: int = 4,
    bars_per_section: int = 8,
) -> Path:
    """Write a PNG chart and return its path.

    Args:
        song_name: Display name used in the chart title.
        predictions: Model names mapped to their predicted timelines.
        output_path: Destination PNG path.
        reference: Optional annotated chord timeline. If present, predictions
            are sampled at reference-beat midpoints and colored green/red.
        beats_per_bar: Number of displayed beat cells per bar.
        bars_per_section: Number of bars shown on each horizontal section.

    The first prediction supplies the beat grid when no reference is provided.
    Other predictions are sampled at the midpoint of that shared grid, making
    charts comparable even if models use different interval boundaries.
    """

    if not predictions:
        raise ValueError("At least one prediction is required to create a chart.")
    if beats_per_bar <= 0 or bars_per_section <= 0:
        raise ValueError("beats_per_bar and bars_per_section must be positive.")
    for model_name, timeline in predictions.items():
        _validate_timeline(model_name, timeline)

    grid = reference or next(iter(predictions.values()))
    _validate_timeline("Ground truth" if reference is not None else "chart grid", grid)
    midpoints = (np.asarray(grid.starts) + np.asarray(grid.ends)) / 2

    row_names: list[str] = []
    row_labels: list[np.ndarray] = []
    reference_labels: np.ndarray | None = None
    if reference is not None:
        reference_labels = np.asarray(reference.labels)
        row_names.append("Ground truth")
        row_labels.append(reference_labels)

    for model_name, timeline in predictions.items():
        row_names.append(model_name)
        row_labels.append(
            labels_at_times(midpoints, timeline.starts, timeline.ends, timeline.labels)
        )

    beats_per_section = beats_per_bar * bars_per_section
    section_count = math.ceil(len(midpoints) / beats_per_section)
    figure_height = max(2.8, section_count * (1.2 + 0.55 * len(row_names)))
    figure, axes = plt.subplots(
        section_count,
        1,
        figsize=(16, figure_height),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(f"Chord chart: {song_name}", fontsize=15, fontweight="bold")

    for section_index, axis in enumerate(axes[:, 0]):
        start = section_index * beats_per_section
        end = min(start + beats_per_section, len(midpoints))
        _draw_section(
            axis,
            start,
            end,
            row_names,
            row_labels,
            reference_labels,
            beats_per_bar,
        )

    if reference is not None:
        figure.legend(
            handles=(
                Patch(facecolor=_CORRECT_COLOR, label="Correct"),
                Patch(facecolor=_INCORRECT_COLOR, label="Incorrect"),
                Patch(facecolor=_UNSCORED_COLOR, label="Unsupported ground truth"),
            ),
            loc="outside lower center",
            ncols=3,
            frameon=False,
        )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return target
