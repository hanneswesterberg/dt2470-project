import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train_hierarchical_cnn import load_song_arrays, transpose_cqt


class CNNTrainingTests(unittest.TestCase):
    def test_requires_extension_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_dir = Path(directory)
            np.savez(
                feature_dir / "RWC_P001.npz",
                X=np.zeros((2, 72, 32), dtype=np.float32),
                root=np.array([0, 11]),
                quality=np.array([0, 1]),
                seventh=np.array([0, 2]),
            )

            with self.assertRaisesRegex(KeyError, "Missing extension targets"):
                load_song_arrays(feature_dir, 1, 7, None)

    def test_loads_explicit_extension_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_dir = Path(directory)
            extensions = np.zeros((1, 7), dtype=np.float32)
            extensions[0, 2] = 1
            np.savez(
                feature_dir / "RWC_P002.npz",
                X=np.zeros((1, 72, 32), dtype=np.float32),
                root=np.array([4]),
                quality=np.array([0]),
                seventh=np.array([0]),
                extensions=extensions,
            )

            arrays = load_song_arrays(feature_dir, 2, 7, "extensions")

            np.testing.assert_array_equal(arrays[4], extensions)

    def test_transposition_rotates_each_octave(self) -> None:
        features = torch.arange(72 * 32).reshape(72, 32)

        shifted = transpose_cqt(features, 1).reshape(6, 12, 32)
        original = features.reshape(6, 12, 32)

        torch.testing.assert_close(shifted[:, 1], original[:, 0])
        torch.testing.assert_close(shifted[:, 0], original[:, 11])


if __name__ == "__main__":
    unittest.main()
