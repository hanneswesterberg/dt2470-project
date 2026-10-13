"""Tests for interactive prompt parsing and script-level selections."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from data.load_data import SongFiles
from main import _parse_cloud_request
from src.cli import ask_choice, ask_many
from src.models import component_path


class PromptTests(unittest.TestCase):
    @patch("builtins.input", return_value="2")
    def test_numbered_choice(self, _) -> None:
        result = ask_choice("Choose", (("a", "First"), ("b", "Second")))
        self.assertEqual(result, "b")

    @patch("builtins.input", return_value="1,3-4")
    def test_multiple_selection_with_range(self, _) -> None:
        items = [(value, str(value)) for value in range(1, 5)]
        self.assertEqual(ask_many("Choose", items), [1, 3, 4])


class ScriptSelectionTests(unittest.TestCase):
    def test_cloud_ids_and_random_selection(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cloud = {
                song_id: SongFiles(
                    song_id, root / f"{song_id}.wav", root / f"{song_id}.arff"
                )
                for song_id in ("0001", "0002", "0003")
            }
            self.assertEqual(
                [song.song_id for song in _parse_cloud_request("0001,0003", cloud, 1)],
                ["0001", "0003"],
            )
            self.assertEqual(len(_parse_cloud_request("random 2", cloud, 1)), 2)

    def test_component_artifacts_use_separate_folders(self) -> None:
        self.assertEqual(
            component_path(Path("models"), "emission", "experiment"),
            Path("models/emissions/experiment.pkl"),
        )
        self.assertEqual(
            component_path(Path("models"), "transition", "experiment"),
            Path("models/transitions/experiment.pkl"),
        )


if __name__ == "__main__":
    unittest.main()
