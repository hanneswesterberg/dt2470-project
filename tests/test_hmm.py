import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.hmm.emission import GaussianEmissionModel, shift_to_c
from src.hmm.transition_prob import RelativeTransitionModel
from src.hmm.viterbi import viterbi_decode
from src.models import ChordRecognitionModel


class EmissionTests(unittest.TestCase):
    def test_shift_to_c_rotates_both_bands(self) -> None:
        features = np.zeros((1, 24))
        features[0, [2, 14]] = 1
        shifted = shift_to_c(features, 2)
        self.assertEqual(shifted[0, 0], 1)
        self.assertEqual(shifted[0, 12], 1)

    def test_shared_gaussian_scores_all_chords(self) -> None:
        rng = np.random.default_rng(4)
        features = rng.normal(size=(16, 24))
        states = np.asarray([0] * 8 + [1] * 8)
        model = GaussianEmissionModel().fit(features, states)
        self.assertEqual(model.score(features[:3]).shape, (3, 24))

    def test_fitted_model_checkpoint_round_trip(self) -> None:
        rng = np.random.default_rng(8)
        features = rng.normal(size=(16, 24))
        states = np.asarray([0, 1] * 8)
        model = ChordRecognitionModel(
            GaussianEmissionModel().fit(features, states),
            RelativeTransitionModel().fit([states]),
            transition_weight=2.0,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.pkl"
            model.save(path)
            restored = ChordRecognitionModel.load(path)
            np.testing.assert_array_equal(
                restored.predict(features), model.predict(features)
            )
            self.assertEqual(restored.transition_weight, 2.0)


class TransitionAndDecoderTests(unittest.TestCase):
    def test_relative_transition_matrix_is_stochastic(self) -> None:
        sequences = [np.asarray([0, 0, 10, 10, 1]), np.asarray([2, 2, 12])]
        model = RelativeTransitionModel().fit(sequences)
        np.testing.assert_allclose(model.matrix.sum(axis=1), 1.0)
        self.assertTrue(np.all(model.matrix > 0))

    def test_viterbi_uses_later_observations_to_select_path(self) -> None:
        emissions = np.asarray([[0.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
        transitions = np.asarray([[0.95, 0.05], [0.05, 0.95]])
        path = viterbi_decode(emissions, transitions)
        np.testing.assert_array_equal(path, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
