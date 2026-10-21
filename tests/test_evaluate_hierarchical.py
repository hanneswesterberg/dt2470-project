import unittest

from evaluate_hierarchical import triad_label


class HierarchicalEvaluationTests(unittest.TestCase):
    def test_collapses_extended_predictions_to_annotation_quality(self) -> None:
        self.assertEqual(triad_label("Cmaj7"), "Cmaj")
        self.assertEqual(triad_label("Am7"), "Amin")
        self.assertEqual(triad_label("Am"), "Amin")

    def test_unsupported_quality_is_not_scored_as_major(self) -> None:
        self.assertEqual(triad_label("Fdim7"), "N")
        self.assertEqual(triad_label("Gaug"), "N")


if __name__ == "__main__":
    unittest.main()
