"""Cached local audio-to-timeline pipeline for the hierarchical model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.hierarchical.artifacts import HierarchicalArtifacts
from src.hierarchical.audio import CQT72Config, ContinuousCQT72Extractor
from src.hierarchical.decoder import DecoderConfig, HarmonicLanguageModel
from src.hierarchical.model import HierarchicalCNNRecognizer
from src.inference.pipeline import ChordTimeline

PIPELINE_VERSION = "cqt72-hierarchical-hsmm-v1"


class HierarchicalRecognitionPipeline:
    """Use saved CNN/LM artifacts to infer complex chord events from audio."""

    def __init__(
        self,
        artifacts: HierarchicalArtifacts,
        *,
        device: str | None = None,
        top_roots: int = 4,
        batch_size: int = 64,
        hsmm_order: int = 2,
        frontend_config: CQT72Config | None = None,
        decoder_config: DecoderConfig | None = None,
        cache_dir: Path | str = ".cache/chordrec",
    ) -> None:
        self.artifacts = artifacts.validate()
        self.recognizer = HierarchicalCNNRecognizer(artifacts.cnn_checkpoint, device)
        self._validate_split_manifest()
        self.language = HarmonicLanguageModel(artifacts.language_model, self.recognizer.vocabulary)
        self.extractor = ContinuousCQT72Extractor(frontend_config)
        self.decoder_config = decoder_config or DecoderConfig(step=self.extractor.config.step)
        if self.decoder_config.step != self.extractor.config.step:
            raise ValueError("Frontend and decoder observation steps must match.")
        self.top_roots = top_roots
        self.batch_size = batch_size
        self.hsmm_order = hsmm_order
        self.cache_dir = Path(cache_dir)

    def _validate_split_manifest(self) -> None:
        manifest = json.loads(self.artifacts.split_manifest.read_text(encoding="utf-8"))
        splits = manifest.get("splits", {})
        sizes = tuple(len(splits.get(name, ())) for name in ("train", "val", "test"))
        if sizes != (70, 15, 15):
            raise ValueError(
                f"Expected the frozen 70/15/15 split manifest, found {sizes}."
            )
        checkpoint_manifest = self.recognizer.metadata.get("source_split_manifest")
        if checkpoint_manifest is not None and checkpoint_manifest.get("splits") != splits:
            raise ValueError("Checkpoint and split_manifest.json contain different splits.")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _cache_path(self, audio_path: Path) -> Path:
        identity = {
            "pipeline": PIPELINE_VERSION,
            "audio_sha256": self._sha256(audio_path),
            "cnn_sha256": self._sha256(self.artifacts.cnn_checkpoint),
            "frontend": self.extractor.config.cache_values(),
            "top_roots": self.top_roots,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        return self.cache_dir / f"{audio_path.stem}-{key}.npz"

    def posteriors(self, audio_path: Path | str) -> tuple[np.ndarray, np.ndarray, float, bool]:
        path = Path(audio_path)
        cache_path = self._cache_path(path)
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["pipeline_version"]) == PIPELINE_VERSION:
                    return (
                        cached["centers"],
                        cached["probabilities"],
                        float(cached["duration"]),
                        True,
                    )
        centers, windows, duration = self.extractor.transform_file(path)
        probabilities = self.recognizer.predict_posteriors(
            windows, batch_size=self.batch_size, top_roots=self.top_roots
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            centers=centers,
            probabilities=probabilities,
            duration=np.asarray(duration),
            step=np.asarray(self.extractor.config.step),
            pipeline_version=np.asarray(PIPELINE_VERSION),
        )
        temporary.replace(cache_path)
        return centers, probabilities, duration, False

    def process(self, audio_path: Path | str) -> tuple[ChordTimeline, bool]:
        _, probabilities, duration, cache_hit = self.posteriors(audio_path)
        segments = self.language.decode(
            probabilities, order=self.hsmm_order, config=self.decoder_config
        )
        starts = np.asarray(
            [segment.start_step * self.decoder_config.step for segment in segments], dtype=np.float64
        )
        ends = np.asarray(
            [min(segment.end_step * self.decoder_config.step, duration) for segment in segments], dtype=np.float64
        )
        labels = np.asarray(
            [self.recognizer.vocabulary.chord_name(segment.state) for segment in segments], dtype="<U24"
        )
        return ChordTimeline(starts, ends, labels), cache_hit
