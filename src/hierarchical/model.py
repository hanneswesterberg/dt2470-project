"""Exact hierarchical CNN architecture and joint-posterior inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from src.hierarchical.vocabulary import ChordVocabulary


class Encoder72(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(72, 96, 5, padding=2),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.30),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class HierarchicalCNN72(nn.Module):
    """Network definition whose parameter names match ``best_model.pt``."""

    def __init__(self, quality_count: int = 6, seventh_count: int = 4, extension_count: int = 7) -> None:
        super().__init__()
        self.root_encoder = Encoder72()
        self.rel_encoder = Encoder72()
        self.root_head = nn.Linear(128, 12)
        self.quality_head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, quality_count))
        self.q_embed = nn.Embedding(quality_count, 16)
        self.s_embed = nn.Embedding(seventh_count, 12)
        self.seventh_head = nn.Sequential(nn.Linear(144, 128), nn.GELU(), nn.Linear(128, seventh_count))
        self.extension_head = nn.Sequential(nn.Linear(156, 128), nn.GELU(), nn.Linear(128, extension_count))

    def root_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.root_head(self.root_encoder(inputs))

    def rel_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.rel_encoder(inputs)

    def quality_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.quality_head(features)

    def seventh_logits(self, features: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        return self.seventh_head(torch.cat([features, self.q_embed(quality)], dim=1))

    def extension_logits(self, features: torch.Tensor, quality: torch.Tensor, seventh: torch.Tensor) -> torch.Tensor:
        return self.extension_head(torch.cat([features, self.q_embed(quality), self.s_embed(seventh)], dim=1))

    def forward(self, absolute: torch.Tensor, relative: torch.Tensor, quality: torch.Tensor, seventh: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.rel_features(relative)
        return (
            self.root_logits(absolute),
            self.quality_logits(features),
            self.seventh_logits(features, quality),
            self.extension_logits(features, quality, seventh),
        )


def rotate_to_root(raw: torch.Tensor, roots: torch.Tensor) -> torch.Tensor:
    """Reindex six octaves so each candidate root becomes relative pitch zero."""

    batch, bins, frames = raw.shape
    if bins != 72:
        raise ValueError(f"Expected 72 frequency bins, got {bins}.")
    reshaped = raw.reshape(batch, 6, 12, frames)
    indices = (roots[:, None] + torch.arange(12, device=raw.device)[None, :]) % 12
    return reshaped.gather(
        2, indices[:, None, :, None].expand(-1, 6, -1, frames)
    ).reshape(batch, 72, frames)


class HierarchicalCNNRecognizer:
    """Loaded checkpoint plus exhaustive/approximate joint posterior inference."""

    def __init__(self, checkpoint: Path | str, device: str | None = None) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.vocabulary = ChordVocabulary(
            tuple(payload["quality"]), tuple(payload["seventh"]), tuple(payload["extensions"])
        )
        if (len(self.vocabulary.quality), len(self.vocabulary.seventh)) != (6, 4):
            raise ValueError("Checkpoint must use the saved 6-quality/4-seventh ordering.")
        self.mean = float(payload["mean"])
        self.std = float(payload["std"])
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("Checkpoint normalization standard deviation is invalid.")
        self.metadata = {
            key: payload.get(key)
            for key in (
                "epoch",
                "score",
                "feature",
                "seed",
                "source_split_manifest",
            )
        }
        self.model = HierarchicalCNN72(
            len(self.vocabulary.quality),
            len(self.vocabulary.seventh),
            len(self.vocabulary.extensions),
        ).to(self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()

    @torch.no_grad()
    def predict_posteriors(
        self,
        windows: NDArray[np.floating],
        *,
        batch_size: int = 64,
        top_roots: int = 4,
    ) -> NDArray[np.float32]:
        inputs = np.asarray(windows, dtype=np.float32)
        if inputs.ndim != 3 or inputs.shape[1:] != (72, 32):
            raise ValueError(f"Expected windows with shape (N, 72, 32), got {inputs.shape}.")
        if not 1 <= top_roots <= 12:
            raise ValueError("top_roots must be between 1 and 12.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        outputs: list[NDArray[np.float32]] = []
        q_count = len(self.vocabulary.quality)
        s_count = len(self.vocabulary.seventh)
        for start in range(0, len(inputs), batch_size):
            raw = torch.as_tensor(inputs[start : start + batch_size], device=self.device)
            batch = raw.shape[0]
            root_probability = torch.softmax(
                self.model.root_logits((raw - self.mean) / self.std), dim=1
            )
            roots_by_probability = root_probability.topk(top_roots, dim=1).indices
            posterior = torch.zeros((batch, self.vocabulary.state_count), device=self.device)
            for root_rank in range(top_roots):
                roots = roots_by_probability[:, root_rank]
                relative = rotate_to_root(raw, roots)
                features = self.model.rel_features((relative - self.mean) / self.std)
                quality_probability = torch.softmax(self.model.quality_logits(features), dim=1)
                for quality in range(q_count):
                    qualities = torch.full((batch,), quality, dtype=torch.long, device=self.device)
                    seventh_probability = torch.softmax(
                        self.model.seventh_logits(features, qualities), dim=1
                    )
                    joint = (
                        root_probability.gather(1, roots[:, None])
                        * quality_probability[:, quality : quality + 1]
                        * seventh_probability
                    )
                    for seventh in range(s_count):
                        states = roots * (q_count * s_count) + quality * s_count + seventh
                        posterior.scatter_(1, states[:, None], joint[:, seventh : seventh + 1])
            posterior /= posterior.sum(dim=1, keepdim=True).clamp_min(1e-12)
            outputs.append(posterior.cpu().numpy())
        if not outputs:
            return np.empty((0, self.vocabulary.state_count), dtype=np.float32)
        return np.concatenate(outputs, axis=0)
