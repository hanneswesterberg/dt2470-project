"""Tests for static and interactive inference visualizations."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.inference import ChordTimeline
from src.visualization import write_chord_chart, write_interactive_viewer


def _timeline(labels: list[str], offset: float = 0.0) -> ChordTimeline:
    starts = np.arange(len(labels), dtype=float) + offset
    return ChordTimeline(starts, starts + 1.0, np.asarray(labels))


class ChordChartTests(unittest.TestCase):
    def test_writes_prediction_only_png(self) -> None:
        prediction = _timeline(["Cmaj", "Cmaj", "Gmaj", "Gmaj", "Amin"])
        with TemporaryDirectory() as directory:
            target = Path(directory) / "charts" / "song.png"
            returned = write_chord_chart("song", {"model": prediction}, target)

            self.assertEqual(returned, target)
            self.assertTrue(target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_writes_ground_truth_comparison_with_multiple_models(self) -> None:
        reference = _timeline(["Cmaj", "Gmaj", "Amin", "Fmaj"] * 3)
        shifted = _timeline(["Cmaj", "Gmaj", "Cmaj", "Fmaj"] * 3, offset=0.1)
        with TemporaryDirectory() as directory:
            target = Path(directory) / "comparison.png"
            write_chord_chart(
                "comparison",
                {"first": reference, "second": shifted},
                target,
                reference,
                bars_per_section=2,
            )

            self.assertGreater(target.stat().st_size, 1_000)

    def test_rejects_empty_prediction_collection(self) -> None:
        with (
            TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "At least one prediction"),
        ):
            write_chord_chart("song", {}, Path(directory) / "song.png")


class InteractiveViewerTests(unittest.TestCase):
    def test_writes_audio_player_with_synchronized_events(self) -> None:
        prediction = _timeline(["Cmaj7", "Am7", "Fmaj7", "G7"])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "input" / "demo song.wav"
            audio.parent.mkdir()
            audio.touch()
            target = root / "output" / "demo_viewer.html"

            returned = write_interactive_viewer(
                "Demo song",
                audio,
                prediction,
                target,
                model_name="Hierarchical CNN + HSMM",
            )
            document = target.read_text(encoding="utf-8")

            self.assertEqual(returned, target)
            self.assertIn("<audio", document)
            self.assertIn("../input/demo%20song.wav", document)
            self.assertIn('"chord": "Cmaj7"', document)
            self.assertIn("Live chord transcription", document)

    def test_rejects_an_empty_timeline(self) -> None:
        empty = ChordTimeline(np.asarray([]), np.asarray([]), np.asarray([]))
        with (
            TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "empty timeline"),
        ):
            root = Path(directory)
            write_interactive_viewer(
                "Empty", root / "audio.wav", empty, root / "viewer.html", model_name="test"
            )


if __name__ == "__main__":
    unittest.main()
