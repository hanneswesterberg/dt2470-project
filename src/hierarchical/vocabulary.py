"""The root-quality-seventh state space saved with the hierarchical model."""

from __future__ import annotations

from dataclasses import dataclass

ROOT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")


@dataclass(frozen=True)
class ChordVocabulary:
    """Ordered checkpoint vocabulary for the 12 x quality x seventh states."""

    quality: tuple[str, ...]
    seventh: tuple[str, ...]
    extensions: tuple[str, ...] = ()

    @property
    def state_count(self) -> int:
        return 12 * len(self.quality) * len(self.seventh)

    def encode(self, root: int, quality: int, seventh: int) -> int:
        if not 0 <= root < 12:
            raise ValueError(f"Root index is outside 0..11: {root}")
        if not 0 <= quality < len(self.quality):
            raise ValueError(f"Quality index is invalid: {quality}")
        if not 0 <= seventh < len(self.seventh):
            raise ValueError(f"Seventh index is invalid: {seventh}")
        return root * len(self.quality) * len(self.seventh) + quality * len(
            self.seventh
        ) + seventh

    def decode(self, state: int) -> tuple[int, int, int]:
        if not 0 <= int(state) < self.state_count:
            raise ValueError(f"Chord state is outside the vocabulary: {state}")
        group = len(self.quality) * len(self.seventh)
        state = int(state)
        return (
            state // group,
            (state // len(self.seventh)) % len(self.quality),
            state % len(self.seventh),
        )

    def chord_name(self, state: int) -> str:
        """Format a state without pretending that unsupported extensions exist."""

        root, quality_index, seventh_index = self.decode(state)
        quality = self.quality[quality_index]
        seventh = self.seventh[seventh_index]
        quality_suffix = {
            "maj": "",
            "min": "m",
            "dim": "dim",
            "aug": "aug",
            "sus2": "sus2",
            "sus4": "sus4",
        }.get(quality, quality)
        seventh_suffix = {
            "none": "",
            "b7": "7",
            "maj7": "maj7",
            "dim7": "dim7",
        }.get(seventh, seventh)
        return ROOT_NAMES[root] + quality_suffix + seventh_suffix
