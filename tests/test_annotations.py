import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from data.annotations import load_annotations
from data.load_data import find_input_annotation, find_songs
from src.music import normalize_chord


class ChordNormalizationTests(unittest.TestCase):
    def test_supported_spellings_and_unsupported_extensions(self) -> None:
        self.assertEqual(normalize_chord("Db:maj"), "C#maj")
        self.assertEqual(normalize_chord("Bbm"), "A#min")
        self.assertEqual(normalize_chord("F"), "Fmaj")
        self.assertEqual(normalize_chord("C7"), "N")

    def test_arff_duplicates_and_intervals(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "0001_beatinfo.arff"
            path.write_text(
                "@RELATION chords\n0.0,1,1,'C'\n0.5,1,2,'Dm'\n"
                "0.5,1,2,'F'\n1.0,1,3,'Bb:min'\n",
                encoding="utf-8",
            )
            annotations = load_annotations(path)
            np.testing.assert_array_equal(annotations.labels, ["Cmaj", "N", "A#min"])
            starts, ends, labels = annotations.intervals(1.4)
            np.testing.assert_allclose(starts, [0.0, 0.5, 1.0])
            np.testing.assert_allclose(ends, [0.5, 1.0, 1.4])
            np.testing.assert_array_equal(labels, annotations.labels)


class DiscoveryTests(unittest.TestCase):
    def test_pairing_and_input_annotation_names(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            annotation_dir = root / "annotations"
            audio_dir.mkdir()
            annotation_dir.mkdir()
            audio = audio_dir / "0001_mix.wav"
            audio.touch()
            (audio_dir / "unpaired.wav").touch()
            annotation = annotation_dir / "0001_beatinfo.arff"
            annotation.touch()
            songs = find_songs(audio_dir, annotation_dir)
            self.assertEqual(list(songs), ["0001"])
            self.assertEqual(songs["0001"].audio, audio)

            local_annotation = audio_dir / "0001_beatinfo.arff"
            local_annotation.touch()
            self.assertEqual(find_input_annotation(audio), local_annotation)


if __name__ == "__main__":
    unittest.main()
