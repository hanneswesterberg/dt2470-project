"""Tests for the public component-training API."""

import unittest

import numpy as np

from src.training import train_emission_model, train_hmm, train_transition_model


class ComponentTrainingTests(unittest.TestCase):
    def test_emission_can_be_trained_independently(self) -> None:
        rng = np.random.default_rng(12)
        features = rng.normal(size=(17, 24))
        states = np.asarray([0] * 8 + [1] * 8 + [-1])

        emission = train_emission_model(features, states)

        self.assertTrue(emission.is_fitted)
        self.assertEqual(emission.sample_counts, (8, 8))
        self.assertEqual(emission.score(features[:2]).shape, (2, 24))

    def test_transition_model_preserves_song_boundaries(self) -> None:
        first_song = np.asarray([0, 0, 10])
        second_song = np.asarray([2, 2, 12])

        transition = train_transition_model([first_song, second_song], smoothing=0.25)

        self.assertIsNotNone(transition.matrix)
        np.testing.assert_allclose(transition.matrix.sum(axis=1), 1.0)

    def test_complete_hmm_can_be_trained_from_prepared_arrays(self) -> None:
        rng = np.random.default_rng(3)
        features = rng.normal(size=(16, 24))
        states = np.asarray([0, 1] * 8)

        model = train_hmm(features, states, [states], transition_weight=2.0)

        self.assertTrue(model.is_fitted)
        self.assertEqual(model.transition_weight, 2.0)
        self.assertEqual(len(model.predict(features)), len(features))


if __name__ == "__main__":
    unittest.main()
