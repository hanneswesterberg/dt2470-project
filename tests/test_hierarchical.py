"""Structural tests for the default hierarchical recognition pipeline."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.hierarchical.artifacts import HierarchicalArtifacts
from src.hierarchical.cqt_frontend import CQT72Config, ContinuousCQT72Extractor
from src.hierarchical.vocabulary import ChordVocabulary
from src.inference import ChordTimeline


class HierarchicalVocabularyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = ChordVocabulary(
            ("maj", "min", "dim", "aug", "sus2", "sus4"),
            ("none", "b7", "maj7", "dim7"),
        )

    def test_288_state_round_trip(self) -> None:
        self.assertEqual(self.vocabulary.state_count, 288)
        for state in range(self.vocabulary.state_count):
            self.assertEqual(self.vocabulary.encode(*self.vocabulary.decode(state)), state)

    def test_complex_chord_names(self) -> None:
        self.assertEqual(self.vocabulary.chord_name(self.vocabulary.encode(0, 0, 2)), "Cmaj7")
        self.assertEqual(self.vocabulary.chord_name(self.vocabulary.encode(10, 1, 1)), "Bbm7")
        self.assertEqual(self.vocabulary.chord_name(self.vocabulary.encode(7, 5, 0)), "Gsus4")


class FrontendTests(unittest.TestCase):
    def test_rolling_window_shape_and_normalization(self) -> None:
        config = CQT72Config(window_frames=32, step=0.2)
        extractor = ContinuousCQT72Extractor(config)
        times = np.linspace(0.0, 1.0, 50)
        cqt = np.ones((72, len(times)), dtype=np.float32)
        centers, windows = extractor.rolling_windows(cqt, times, 1.0)
        self.assertEqual(centers.shape, (5,))
        self.assertEqual(windows.shape, (5, 72, 32))
        np.testing.assert_allclose(windows.sum(axis=1), 1.0, atol=1e-6)


class ArtifactAndExportTests(unittest.TestCase):
    def test_explicit_artifact_set_must_be_complete(self) -> None:
        with self.assertRaises(ValueError):
            HierarchicalArtifacts.resolve(cnn_checkpoint=Path("model.pt"))

    def test_timeline_json_output(self) -> None:
        timeline = ChordTimeline(
            np.asarray([0.0]), np.asarray([0.8]), np.asarray(["Cmaj7"])
        )
        with TemporaryDirectory() as directory:
            target = Path(directory) / "events.json"
            timeline.write_json(target)
            self.assertIn('"chord": "Cmaj7"', target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
