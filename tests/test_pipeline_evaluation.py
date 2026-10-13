import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.evaluation import evaluate_timeline, labels_at_times, match_beats
from src.inference import ChordTimeline, beat_intervals
from src.models import list_models, resolve_model_path


class TimelineTests(unittest.TestCase):
    def test_named_models_resolve_inside_models_folder(self) -> None:
        self.assertEqual(
            resolve_model_path("experiment_2", Path("models")),
            Path("models/experiment_2.pkl"),
        )
        self.assertEqual(
            resolve_model_path("checkpoints/custom.pkl", Path("models")),
            Path("checkpoints/custom.pkl"),
        )

    def test_models_are_listed_by_name(self) -> None:
        with TemporaryDirectory() as directory:
            models = Path(directory)
            (models / "second.pkl").touch()
            (models / "first.pkl").touch()
            (models / "notes.txt").touch()
            self.assertEqual(
                [path.stem for path in list_models(models)], ["first", "second"]
            )

    def test_beat_intervals_end_at_audio_duration(self) -> None:
        starts, ends = beat_intervals(np.asarray([0.2, 0.7, 1.2]), 1.5)
        np.testing.assert_allclose(starts, [0.2, 0.7, 1.2])
        np.testing.assert_allclose(ends, [0.7, 1.2, 1.5])

    def test_csv_output(self) -> None:
        timeline = ChordTimeline(
            np.asarray([0.0]), np.asarray([1.0]), np.asarray(["Cmaj"])
        )
        with TemporaryDirectory() as directory:
            target = Path(directory) / "result.csv"
            timeline.write_csv(target)
            self.assertIn("0.000000,1.000000,Cmaj", target.read_text())


class EvaluationTests(unittest.TestCase):
    def test_interval_lookup_respects_gaps(self) -> None:
        labels = labels_at_times(
            np.asarray([0.25, 0.75, 1.25]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.5, 1.5]),
            np.asarray(["Cmaj", "Gmaj"]),
        )
        np.testing.assert_array_equal(labels, ["Cmaj", "N", "Gmaj"])

    def test_perfect_evaluation(self) -> None:
        starts = np.asarray([0.0, 0.5, 1.0])
        ends = np.asarray([0.5, 1.0, 1.5])
        labels = np.asarray(["Cmaj", "Gmaj", "Amin"])
        result = evaluate_timeline(starts, ends, labels, starts, ends, labels)
        self.assertEqual(result.chord_accuracy, 1.0)
        self.assertEqual(result.beat_f1, 1.0)

    def test_beat_matching_is_one_to_one(self) -> None:
        precision, recall, f1 = match_beats(
            np.asarray([0.0, 1.0]), np.asarray([0.01, 0.02, 1.02])
        )
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 0.8)


if __name__ == "__main__":
    unittest.main()
